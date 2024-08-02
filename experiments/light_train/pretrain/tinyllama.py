import glob
import math
import sys
import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import pdb
import wandb
from transformers import  LlamaTokenizer, AutoModelForCausalLM

your_api_key = os.environ["WANDB_API_KEY"]
wandb.login(key=your_api_key)


global_batch_size = 512
learning_rate = 4e-4
log_step_interval = 10 #10
eval_iters =  None

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

gradient_accumulation_steps = -1
log_iter_interval = -1

train_data_config = [
    ("train", 1.0),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

def format_filename(filename):
    pattern = re.compile(r'(\d+)M-V(\d+)')
    
    match = pattern.search(filename)
    if match:
        m_number = match.group(1)
        v_number = match.group(2)
        
        formatted_m_number = m_number.zfill(7)
        formatted_v_number = v_number.zfill(7)
        
        new_filename = pattern.sub(f"{formatted_m_number}M-V{formatted_v_number}", filename)
        return new_filename
    else:
        return filename

def compute_eval_steps(max_steps, evals_per_interval=20):
    '''
    max_steps: the max training steps.
    evals_per_interval: the number of evaluation during the training.
    It return the steps that evaluate the model.
    '''
    eval_intervals = []
    eval_steps = []

    intervals = [0, max_steps]
    for i in range(len(intervals)-1):
        start = intervals[i]
        end = intervals[i+1]
        step_size = (end - start) / evals_per_interval
        for j in range(1, evals_per_interval+1):
            eval_step = start + j * step_size
            eval_steps.append(eval_step)
    eval_steps = [round(step) for step in eval_steps if step <= max_steps]
    return eval_steps


def setup(
    num_of_device = 1,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    micro_batch_size_para: int = 8,
    precision: Optional[str] = 'bf16-mixed',
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    resume_step: Optional[str] = None,
    model_name: str ='tinyllama',
    max_step: int  = 25000,
    warmup_steps: int = 2000,
    save_step_interval: int  = 500,
    eval_step_interval: int = 500,
    exp_name: str = '',
    eval_only: bool = False,
    path_to_tokenid_probabilities,
) -> None:
    
    batch_size = global_batch_size // num_of_device
    global micro_batch_size
    global gradient_accumulation_steps
    global log_iter_interval
    global eval_steps
    micro_batch_size = micro_batch_size_para

    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0
    log_iter_interval = log_step_interval * gradient_accumulation_steps

    eval_steps = compute_eval_steps(max_step)
    # Doc for precision
    # https://lightning.ai/docs/fabric/stable/fundamentals/precision.html

    exp_fullname = format_filename(str(model_name + exp_name))
    out_dir = Path("out") / exp_fullname
    logger = step_csv_logger("out", exp_fullname, flush_logs_every_n_steps=log_iter_interval)
    os.makedirs(out_dir, exist_ok=True)
    wandb_logger = WandbLogger(project='slim-V-scaling_laws', name=exp_fullname, save_dir=out_dir)

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if num_of_device > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            num_of_device = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=num_of_device, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    # fabric.print(hparams)

    max_iters = max_step * gradient_accumulation_steps
    warmup_iters = warmup_steps * gradient_accumulation_steps
    fabric.print(f"eval_steps {eval_steps}")
    fabric.print(f"max_iters {max_iters}\nsave_step_interval {save_step_interval}.")
    main(fabric, train_data_dir, val_data_dir, resume, resume_step, model_name, 
        exp_fullname, max_iters, warmup_iters, save_step_interval, eval_steps,
        eval_only, path_to_tokenid_probabilities)

def main(fabric, train_data_dir, val_data_dir, resume, resume_step, model_name, 
        exp_fullname, max_iters, warmup_iters, save_step_interval,  eval_steps,
        eval_only, path_to_tokenid_probabilities):
    out_dir = Path("out") / exp_fullname
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))


    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    
    with open(path_to_tokenid_probabilities, 'rb') as f:
        tokenid_probabilities = json.load(f) # a dict

        
    max_key = max(tokenid_probabilities.keys())
    lookup_probabilities = torch.empty(max_key + 1).to(fabric.device)
    for k, v in tokenid_probabilities.items():
        lookup_probabilities[k] = v 

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        if resume_step is not None:
            resume = out_dir / ("step-"+f"{int(resume_step):06d}"+"-ckpt.pth")
        else:
            if sorted(out_dir.glob("*.pth")) == []:
                resume = False # no .pth file to resume
            else:
                resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, out_dir, max_iters, warmup_iters, 
        save_step_interval, eval_steps, model_name, lookup_probabilities, eval_only)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, out_dir, max_iters, warmup_iters, 
        save_step_interval, eval_steps,  model_name, lookup_probabilities, eval_only):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None and eval_only:
        val_loss = validate(fabric, model, val_dataloader, max_iter=eval_iters)  # sanity check
        val_lossu = validate_pplu(fabric, model, val_dataloader, lookup_probabilities, max_iter=eval_iters)  # sanity check
        fabric.print(f"val_loss is {val_loss.item()}\nval_ppl is { math.exp(val_loss.item())}")
        lmloss_results_file = out_dir / f"step-{state['step_count']:06d}-ckpt-lmloss.txt"
        with open(lmloss_results_file, "w", encoding="utf-8") as f:
            f.write(f"{val_loss.item()}")
        exit(0)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")

        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = FusedCrossEntropyLoss()
    for  train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], lr_decay_iters=max_iters, warmup_iters=warmup_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
                
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        if val_dataloader is not None:
             if (not is_accumulating and state["step_count"] in eval_steps):  
                t0 = time.perf_counter()
                # val_loss = validate(fabric, model, val_dataloader)
                val_lossu = validate_pplu(fabric, model, val_dataloader, lookup_probabilities)
                t1 = time.perf_counter() - t0
                monitor.eval_end(t1)
                fabric.print(f"step {state['iter_num']}: val lossu {val_lossu:.8f}, val time: {t1 * 1000:.2f}ms")
                fabric.log_dict({"metric/val_lossu": val_lossu.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.log_dict({"metric/val_pplu":  math.exp(val_lossu.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.barrier()

                pplu_results_file = out_dir / f"step-{state['step_count']:06d}-ckpt.txt"
                with open(pplu_results_file, "w", encoding="utf-8") as f:
                    f.write(f"{math.exp(val_lossu.item())}")

        if (not is_accumulating and state["step_count"] in eval_steps):
            # delete previous checkpoints to save space.
            if fabric.global_rank == 0:
                for file_path in out_dir.glob('*.pth'):
                    file_path.unlink()
                    fabric.print(f"Deleted: {file_path}")            
            fabric.barrier()

            checkpoint_path = out_dir / f"step-{state['step_count']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
            
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()

        if state["iter_num"] % log_iter_interval == 0:
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    # print days as well
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )
                


        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, max_iter=None) -> torch.Tensor:
    fabric.print("Validating ppl ...")
    model.eval()

    losses = [] # fabric.device
    first_input_ids = None
    for k, val_data in tqdm(enumerate(val_dataloader)):
        if max_iter is not None and k >= max_iter:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        if first_input_ids is None:
            first_input_ids = input_ids
        else:
            if torch.eq(first_input_ids, input_ids).all():
                break

        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses.append(loss.item())
        
    losses = torch.tensor(losses, device=fabric.device)
    out = losses.mean()
    model.train()
    return out

@torch.no_grad()
def validate_pplu(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, lookup_probabilities, max_iter=None) -> torch.Tensor:
    fabric.print("Validating pplu...")
    model.eval()

    losses = [] # fabric.device
    first_input_ids = None
    for k, val_data in enumerate(val_dataloader):
        if max_iter is not None and k >= max_iter:
            break

        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        if first_input_ids is None:
            first_input_ids = input_ids
        else:
            if torch.eq(first_input_ids, input_ids).all():
                break

        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)# unnormalized
        probabilities = lookup_probabilities[targets].unsqueeze(2) # [B,S,1]
        normalized_logits = torch.nn.functional.softmax(logits, dim=-1) / probabilities
        normalized_logits = normalized_logits.reshape(-1, normalized_logits.shape[-1])
        targets = targets.reshape(-1)
        loss = torch.nn.functional.nll_loss(torch.log(normalized_logits), targets)
        # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses.append(loss.item())
    losses = torch.tensor(losses, device=fabric.device)
    out = losses.mean()
    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    # pdb.set_trace()
    for prefix, _ in data_config:
        # filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        filenames = sorted(glob.glob(str(data_dir / "*.bin" )))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size ,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/slimpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=False,
        seed=seed,
        split="train"
    )# shuffle=False for the data-constraint experiments.
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, lr_decay_iters, warmup_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    return learning_rate
    

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_device', type=int, default=8)
    parser.add_argument('--train_data_dir', type=str, default="")
    parser.add_argument('--val_data_dir', type=str, default="")
    parser.add_argument('--micro_batch_size_para', type=int, default=8)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--tpu', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_step', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="tinyllama")
    parser.add_argument('--max_step', type=int, default=25000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--save_step_interval', type=int, default=500)
    parser.add_argument('--eval_step_interval', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--path_to_tokenid_probabilities', type=str, default='')
    args = parser.parse_args()
    setup(
        args.num_of_device,
        Path(args.train_data_dir),
        Path(args.val_data_dir),
        args.micro_batch_size_para,
        args.precision,
        args.tpu,
        args.resume,
        args.resume_step,
        args.model_name,
        args.max_step,
        args.warmup_steps,
        args.save_step_interval,
        args.eval_step_interval,
        args.exp_name,
        args.eval_only,
        args.path_to_tokenid_probabilities,
    )

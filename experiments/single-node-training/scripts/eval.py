import os
from pathlib import Path
import pdb
import re

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

def eval(model_name='tiny_LLaMA_120M-V1K', exp_name='', batch_size='8'):
    
    out_dir = Path("out_slim") / format_filename(model_name + exp_name)
    
    # convert to hf version and save to out_dir
    for checkpoint_path in out_dir.glob("step-*-ckpt.pth"):
        step_count = checkpoint_path.name
        checkpoint_path = out_dir / step_count
        hf_out_dir = out_dir / step_count[:-9]
        hf_out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = step_count
        if not (hf_out_dir/'pytorch_model.bin').exists():
            convert_cmd = f'python scripts/convert_lit_checkpoint.py \
                --out_dir {hf_out_dir} \
                --checkpoint_name {checkpoint_name} \
                --model_name {model_name} --model_only False'
            os.system(convert_cmd)
        vocab_size = model_name.split('-')[-1][1:]
        train_tokenizier_name = 'hf_slimpajama-6B-' + vocab_size + '-BPE'
        copy_tokenizer_cmd = f'cp train_tokenizier_slim/{train_tokenizier_name}/* {hf_out_dir}'
        os.system(copy_tokenizer_cmd)

    hf_out_dirs = [d for d in out_dir.glob("step*") if d.is_dir()]
    for hf_out_dir in sorted(hf_out_dirs):
        print(f"Eval on {hf_out_dir}...")
        acc_results_file = hf_out_dir / 'acc'
        accnorm_results_file = hf_out_dir / 'accnorm'
        if acc_results_file.exists() and accnorm_results_file.exists():
            continue
        eval_cmd = f'python lm-evaluation-harness/lm_eval/__main__.py \
            --model hf-auto \
            --model_args pretrained={hf_out_dir},dtype="float" \
            --acc_results_file {acc_results_file}\
            --accnorm_results_file {accnorm_results_file}\
            --tasks hellaswag,arc_easy,piqa,openbookqa,winogrande,arc_challenge,boolq\
            --batch_size auto'
        os.system(eval_cmd)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(eval)
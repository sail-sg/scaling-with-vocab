import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count, Process
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer
import pdb


def prepare_func(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int,
    filenames_sample=[], process_id=0
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    tokens_count = 0
    for name in filenames_sample:

        filepath = source_path / name
        prefix, _ = os.path.splitext(name.name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix+'_'+str(process_id),
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
        print(f"vocab_size {tokenizer.vocab_size}")
        print(f"Processing {name}")
        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                tokens_count += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
           

def prepare(
    source_path = "",
    checkpoint_dir = "",
    destination_path = "",
) -> None:
    source_path, checkpoint_dir, destination_path = Path(source_path), Path(checkpoint_dir), Path(destination_path)

    filenames = list(source_path.glob('*.jsonl'))
    num_processes = cpu_count() 
    chunked_filenames = np.array_split(filenames, num_processes)
    processes = []
    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_func, args=(source_path, checkpoint_dir, destination_path, 2049*1024, "", list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', default=4096, type=int)
    args = parser.parse_args()
    vocab = args.vocab

    source_path = "../slimpajama-train-sampled.jsonl"
    checkpoint_dir = f'../trained_tokenizers/hf_slimpajama-6B-{vocab}-BPE'
    destination_path = f'prepared_dataset/train/vocab_{vocab}'
    prepare(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path)

    source_path = "../slimpajama-validation.jsonl"
    checkpoint_dir = f'../trained_tokenizers/hf_slimpajama-6B-{vocab}-BPE'
    destination_path = f'prepared_dataset/validation/vocab_{vocab}'
    prepare(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path)

  
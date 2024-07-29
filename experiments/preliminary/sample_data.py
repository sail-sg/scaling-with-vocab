'''
The code to sample the corpus ```slimpajama.jsonl``` in case the original slimpajama-627B 
(https://huggingface.co/datasets/cerebras/SlimPajama-627B) to too large to do experiments.
'''

import os, random
from tqdm import tqdm
import json

def merge_sample_subsets(sample_rate=0.1, 
        sample_input_path = 'slimpajama-train.jsonl',
        ):
    sample_output_path = f'slimpajama-train-sampled.jsonl'
    counter = 0
    with open(sample_input_path, 'r', encoding='utf-8') as lines:
        for line in tqdm(lines):
            if random.random() < sample_rate:
                sample_output_path.write(json.dumps({"text": json.loads(line)["text"]}, ensure_ascii=False) + "\n")
                counter += 1
                if counter % 10000 == 0:
                    print("Counter: {}".format(counter))
    
    sample_output_path.close()

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(merge_sample_subsets)        
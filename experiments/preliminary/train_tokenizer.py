'''
The code for training the tokenizer with different vocabulary sizes. In details,
we train the tokenizer on an uniformly sampled 6B version 
corpus slimpajama (https://huggingface.co/datasets/cerebras/SlimPajama-627B).
'''
import time

import sentencepiece as spm
import argparse
import os
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='', type=str)
parser.add_argument('--vocab_size', default='4096', type=str)
parser.add_argument('--model_type', default='BPE', type=str)
args = parser.parse_args()

vocab_size = args.vocab_size
start_time = time.time()
spm.SentencePieceTrainer.train(
    input=args.dataset_path,
    model_prefix=f'slimpajama-6B-{vocab_size}-{args.model_type}', 
    shuffle_input_sentence=False, 
    train_extremely_large_corpus=True,
    input_sentence_size=100000,
    max_sentence_length=10000, 
    pad_id=3,
    model_type=args.model_type,
    vocab_size=vocab_size,
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
)
end_time = time.time()
print(end_time - start_time)

if args.model_type == 'BPE':
  sp_model_file = f'slimpajama-6B-{vocab_size}-{args.model_type}.model'
  dirname =  sp_model_file.replace('.model', '')
  output_hf_dir = './' + 'hf_' + dirname
  os.makedirs(output_hf_dir, exist_ok=True)

  print(f"output_hf_dir is {output_hf_dir}")
  tokenizer = LlamaTokenizer(vocab_file=sp_model_file, legacy=False)
  tokenizer.save_pretrained(output_hf_dir)
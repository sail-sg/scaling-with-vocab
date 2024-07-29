'''
The code is used to compute the frequency of word in the tokenized corpus given the tokenizer with
vocabulary size, which is used to compute the unigram-normalized loss. 
We store the frequency in a pickle file.

It requires the path of corpus ```corpus_for_tokenizer_training```
and the directory path of the trained tokenzier.
'''

from transformers import  LlamaTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
import pickle
import pdb
import json
from transformers import AutoTokenizer
import sentencepiece as spm

sp = spm.SentencePieceProcessor()

def token_count(
        corpus_for_tokenizer_training='slimpajama-sampled.jsonl',
        vocab_size='4096',
        prefix='BPE'):
    
    tokenizer_path = f'../trained_tokenizers/hf_slimpajama-6B-{vocab_size}-{prefix}'
    if prefix == '':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        sp_path = tokenizer_path.replace('hf_','')
        sp_path = sp_path + '.model'
        sp.load(sp_path)

    counter = None
    log_step_interval = 1
    fname_prefix = corpus_for_tokenizer_training.replace('/', '-')
    print('fname_prefix is ', fname_prefix)
    with open(corpus_for_tokenizer_training, encoding='utf-8') as f:
        cnt = 0 
        for row in tqdm(f):
            cnt += 1
            if cnt % log_step_interval == 0:
                print(f'current line: {cnt}')
            text = json.loads(row)["text"]
            if prefix=='':
                text_ids = tokenizer.encode(text)
            else:
                text_ids = sp.encode_as_ids(text)
            if counter is None:
                counter = Counter(text_ids)
            else:
                counter.update(Counter(text_ids))


    # # Avoid zero probabilities for the rare tokens
    for i in range(int(vocab_size)):
        counter[i] = counter.get(i, 1)

    total_tokenid = sum(counter.values())
    tokenid_probabilities = {tokenid: count / total_tokenid for tokenid, count in counter.items()}

    save_name = f'../token_lookup_probabilities/tokenid_probabilities_{vocab_size}.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(tokenid_probabilities, f)
    
    # test load
    with open(save_name, 'rb') as f:
        tokenid_probabilities = pickle.load(f)



if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(token_count)


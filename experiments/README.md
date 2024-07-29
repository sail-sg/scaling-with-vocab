### Installation:
```
pip install -r requirements.txt
```

###  Steps to pre-train the model with different vocabulary sizes:
1. **Preliminary**:
    - 1.1 Download the [slimpajama-627B dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B). After downloading the dataset, we merge the chunks into a single jsonl file for easy operation. We get two files, ```slimpajama-train.jsonl``` and ```slimpajama-validation.jsonl```. 
    
    Sample the traning set from the original slimpajama-627B in case the original slimpajama-627B  to too large to do experiments. Then we get the sampled traning set ```slimpajama-train-sampled.jsonl```.
     Take vocabulary size = 4096 as an example:

        
        
        cd preliminary

        python sample_data.py --sample_input_path slimpajama-train.jsonl
        
        
 
    - 1.2 Train the tokenizer on the corpus with a certain vocabulary size. We have provided the trained tokenziers in the directory ```trained_tokenizers/```.
    
        ```
        python train_tokenizer.py --dataset_path slimpajama-train-sampled.jsonl --vocab_size 4096
        ```

    - 1.3 Compute the frequency of each word in the corpus given the tokenizer, which is used for computing unigram-normalized loss. The computed frequency files are stored in pkl file.  The pkl file is a dictionary that records the  the frequency of each word in the tokenized corpus, given the tokenizer with vocabulary size V. We have provided these files in the directory ```token_lookup_probabilities/```.

        ```
        python generate_lookup_probabilities.py --vocab_size 4096
        ```

    - 1.4  Fit the tokenization function that maps from training characters (H) to tokens (D), used to data fitting after the experiments. The fitting technique is robust to various tokenizers, for example, BPE tokenizer, unigram-based tokenizer, word-based tokenizer.

        ```
        python fit_tokenization_func.py
        ```

2. **Single-Node Training**
    - 2.1 Pre-process the corpus from text to IDs, and the pre-processed files are stored in ```single-node-training/prepared_dataset/train``` and ```single-node-training/prepared_dataset/validation``` 
        ```
        cd single-node-training

        python scripts/prepare_data_parallel.py --vocab 4096```

    - 2.2 Pre-training the model by ```scripts/run.sh'''. 
    Here we compare the difference between the compute of traditional perplexity (PPL) and the unigram-normalized perplexity (PPLu), where log(PPLu) is unigram-normalized language modeling loss.
        ```
        # compute the traditional perplexity (PPL) 
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
                losses.append(loss.item())
                
            losses = torch.tensor(losses, device=fabric.device)
            out = losses.mean()
            model.train()
            return out

        # compute the unigram-normalized  perplexity (PPLu) 
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
                losses.append(loss.item())
            losses = torch.tensor(losses, device=fabric.device)
            out = losses.mean()
            model.train()
            return out
        ```

    - 2.3 We use the unigram-normalized loss to evaluate the pre-trained models. In addition, we use [lighteval-0.3.0](https://github.com/huggingface/lighteval/releases/tag/v0.3.0) to evaluate the pre-trained models on downstream tasks. 

3. **Multi-Node Training**

    - 3.1 For multi-node training, we use the [Megatron framework](https://github.com/epfLLM/Megatron-LLM). In the directory [multi-node-training](multi-node-training/), we provide the code snippet of Unigram-normalized language modeling loss.  We can quickly replace the original language modeling loss  with the unigram-normalized version by the replacement of  3 files, ```Megatron-LLM/finetune.py```, ```Megatron-LLM/megatron/model/gpt_model.py```
    and  ```Megatron-LLM/megatron/core/tensor_parallel/cross_entropy.py```.


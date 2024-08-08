max_step=54800
warmup_steps=5480
save_step_interval=54800



for vocab in '4096'
train_data_dir=prepared_dataset/train/vocab_$vocab
val_data_dir=prepared_dataset/validation/vocab_$vocab
path_to_tokenid_probabilities=../token_lookup_probabilities_json/tokenid_probabilities_$vocab.json
do
    model_name=tiny_LLaMA_335M-V$vocab
    lightning run model \
        --node-rank=0  \
        --main-address=127.0.01  \
        --accelerator=cuda \
        --num-nodes=1 \
        --devices=8 \
        pretrain/tinyllama.py \
        --num_of_device 8 \
        --model_name $model_name \
        --exp_name 'IsoFLOPs'\
        --max_step $max_step\
        --warmup_steps $warmup_steps\
        --save_step_interval $save_step_interval \
        --micro_batch_size_para 8\
        --resume True\
        --train_data_dir $train_data_dir \
        --val_data_dir $val_data_dir \
        --path_to_tokenid_probabilities $path_to_tokenid_probabilities        

done
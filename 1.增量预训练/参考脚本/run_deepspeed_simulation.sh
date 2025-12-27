#!/bin/bash

# DeepSpeed ZeRO-3 分布式训练模拟
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed \
    --num_gpus 4 \
    --master_port 29501 \
    pretraining.py \
    --model_name_or_path ../model/Qwen/Qwen3-0.6B \
    --train_file_dir ../data/pretrain \
    --validation_file_dir ../data/pretrain \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft False \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 0.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir ../output/pretrain_deepspeed \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --deepspeed zero3.json \
    --dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ../output/cache
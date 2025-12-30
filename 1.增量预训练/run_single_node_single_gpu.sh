#!/bin/bash


# 单GPU训练脚本 - 适用于Windows+WSL2环境
# dataset_config_name - 数据需要有text字段，没有的化需要转换

CUDA_VISIBLE_DEVICES=0 python pretraining.py \
    --model_name_or_path ../model/Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path ../output/tokenizers_merge \
    --train_file_dir ../data/pretrain \
    --validation_file_dir ../data/pretrain \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
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
    --preprocessing_num_workers 4 \
    --block_size 512 \
    --group_by_length True \
    --output_dir ../output/pretrain \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ../output/cache
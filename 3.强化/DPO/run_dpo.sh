#!/bin/bash

# DPO训练脚本 - 修复SwanLab超时问题
# 主要修复：
# 1. 禁用report_to避免SwanLab超时
# 2. 添加超时和重试机制
# 3. 优化日志记录

CUDA_VISIBLE_DEVICES=0 python dpo_training.py \
    --model_name_or_path ../../model/Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path ../../output/tokenizers_merge \
    --template_name qwen \
    --train_file_dir ../../data/reward \
    --validation_file_dir ../../data/reward \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 100 \
    --max_eval_samples 10 \
    --max_steps 100 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir ../../output/dpo_adapter \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --device_map auto \
    --report_to none \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ../../output/cache \
    --logging_steps 1
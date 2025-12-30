#!/bin/bash
# 单GPU训练脚本，避免多GPU冲突
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN

python sft.py \
    --model_name_or_path ../model/Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path ../output/tokenizers_merge \
    --train_file_dir ../data/finetune \
    --validation_file_dir ../data/finetune \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 100 \
    --max_eval_samples 10 \
    --model_max_length 2048 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir ../output/sft_adapter \
    --overwrite_output_dir \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dtype bfloat16 \
    --bf16 \
    --dataloader_num_workers 1 \
    --remove_unused_columns True \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ../output/cache \
    --flash_attn True
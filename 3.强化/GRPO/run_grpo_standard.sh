#!/bin/bash

# GRPO标准格式训练脚本 - 支持一个问题多个有序答案的数据格式
# 针对32k长文本的配置
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 grpo_training_standard.py \
    --model_name_or_path ../../model/Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path ../../output/tokenizers_merge \
    --train_file_path ../../data/grpo_standard/train.json \
    --train_samples -1 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 13 \
    --output_dir ../../output/grpo_standard_adapter \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    \
    `# 标准GRPO数据格式相关配置` \
    --use_standard_rewards True \
    --max_responses_per_prompt 8 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    \
    `# 显存优化配置` \
    `# GRPO算法要求 generation_batch_size （即 per_device_train_batch_size ）必须能被 num_generations 整除，` \
    `# 因为每次需要生成多个响应来进行比较和学习` \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    \
    `# 答案生成相关配置` \
    `# - num_generations: 每个prompt生成4个响应用于GRPO比较学习` \
    `# - max_prompt_length: 输入prompt最大16384 tokens` \
    `# - max_completion_length: 生成答案最大512 tokens` \
    --num_generations 4 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 16384 \
    --max_completion_length 512

echo "训练完成!"

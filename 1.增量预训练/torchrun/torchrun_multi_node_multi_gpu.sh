#!/bin/bash

# 多节点多GPU分布式训练脚本 - 基于torchrun
# 使用方法：
# 单节点多GPU：直接运行此脚本
# 多节点多GPU：在主节点运行，其他节点需要修改MASTER_ADDR和NODE_RANK

# ======== 分布式配置参数 ========
# 主节点IP地址（多节点训练时需要设置为实际的主节点IP）
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# 主节点端口
export MASTER_PORT=${MASTER_PORT:-"29500"}
# 当前节点的排名（0为主节点，1,2,3为其他节点）
export NODE_RANK=${NODE_RANK:-"0"}
# 总节点数
export WORLD_SIZE=${WORLD_SIZE:-"1"}
# 每个节点的GPU数
export NPROC_PER_NODE=${NPROC_PER_NODE:-"4"}

# ======== 训练参数 ========
MODEL_PATH="../model/Qwen/Qwen3-0.6B"
TOKENIZERS_PATH="../output/tokenizers_merge"
TRAIN_FILE_DIR="../data/pretrain"
VALIDATION_FILE_DIR="../data/pretrain"
OUTPUT_DIR="../output/pretrain"
CACHE_DIR="../output/cache"

# ======== torchrun分布式训练命令 ========
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretraining.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZERS_PATH \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $VALIDATION_FILE_DIR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
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
    --gradient_accumulation_steps 32 \
    --preprocessing_num_workers 4 \
    --block_size 512 \
    --group_by_length True \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --ddp_find_unused_parameters False \
    --logging_first_step True \
    --dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir $CACHE_DIR \
    --deepspeed ds_config.json
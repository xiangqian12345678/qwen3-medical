#!/bin/bash

# DeepSpeed 多节点多GPU分布式训练脚本
# 支持 ZeRO-1, ZeRO-2, ZeRO-3 等训练模式
# 使用方法: bash deepspeed_muti_node_muti_gpu.sh [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]

set -e

# 默认配置参数
ZERO_STAGE=${1:-"3"}                    # ZeRO阶段: 1, 2, 3
NUM_NODES=${2:-"1"}                      # 节点数量
GPUS_PER_NODE=${3:-"4"}                  # 每个节点的GPU数量
MASTER_ADDR=${4:-"localhost"}            # 主节点地址
MASTER_PORT=${5:-"29501"}                # 主节点端口
NODE_RANK=${NODE_RANK:-"0"}              # 当前节点排名 (通过环境变量设置)

# 模型和数据配置
MODEL_PATH="../model/Qwen/Qwen3-0.6B"
TRAIN_DATA_DIR="../data/pretrain"
VALID_DATA_DIR="../data/pretrain"
OUTPUT_DIR="../output/pretrain_deepspeed_zero${ZERO_STAGE}"

# 训练超参数
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
NUM_TRAIN_EPOCHS=0.5
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
MAX_TRAIN_SAMPLES=10000
MAX_EVAL_SAMPLES=10

# DeepSpeed 配置文件选择
case $ZERO_STAGE in
    1)
        DEEPSPEED_CONFIG="zero1.json"
        echo "使用 ZeRO-1 训练模式"
        ;;
    2)
        DEEPSPEED_CONFIG="zero2.json"
        echo "使用 ZeRO-2 训练模式"
        ;;
    3)
        DEEPSPEED_CONFIG="zero3.json"
        echo "使用 ZeRO-3 训练模式"
        ;;
    *)
        echo "错误: 不支持的 ZeRO 阶段: $ZERO_STAGE. 请选择 1, 2, 或 3"
        exit 1
        ;;
esac

# 检查 DeepSpeed 配置文件是否存在
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "错误: DeepSpeed 配置文件 $DEEPSPEED_CONFIG 不存在"
    exit 1
fi

# 计算总进程数
TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

# 显示配置信息
echo "=== DeepSpeed 分布式训练配置 ==="
echo "ZeRO 阶段: $ZERO_STAGE"
echo "节点数量: $NUM_NODES"
echo "每节点GPU数量: $GPUS_PER_NODE"
echo "总进程数: $TOTAL_PROCESSES"
echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "当前节点排名: $NODE_RANK"
echo "DeepSpeed 配置文件: $DEEPSPEED_CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))
export NODE_RANK=$NODE_RANK
export WORLD_SIZE=$TOTAL_PROCESSES

# DeepSpeed 训练命令
export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=0
deepspeed \
    --num_nodes=$NUM_NODES \
    --num_gpus=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    --module pretraining \
    --model_name_or_path "$MODEL_PATH" \
    --train_file_dir "$TRAIN_DATA_DIR" \
    --validation_file_dir "$VALID_DATA_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --do_train \
    --do_eval \
    --seed 42 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir "../output/cache" \
    --save_on_each_node False \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --ignore_data_skip True

echo "训练完成！结果保存在: $OUTPUT_DIR"

# 单节点多GPU运行示例
# bash deepspeed_muti_node_muti_gpu.sh 3 1 4 localhost 29501

# 多节点多GPU运行示例:
# 节点0 (主节点):
# NODE_RANK=0 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501
# 
# 节点1:
# NODE_RANK=1 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501

# 支持的ZeRO模式说明:
# ZeRO-1: 仅优化器状态分片
# ZeRO-2: 优化器状态 + 梯度分片
# ZeRO-3: 优化器状态 + 梯度 + 模型参数分片 (最节省显存)
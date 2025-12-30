#!/bin/bash
set -e

# =========================================================
# torchrun + DeepSpeed 多节点多 GPU PPO 训练脚本
# 支持 ZeRO-1 / ZeRO-2 / ZeRO-3
#
# 使用方法：
# bash torchrun_multi_node_multi_gpu.sh [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
# =========================================================

# ================== 分布式参数 ==================
ZERO_STAGE=${1:-"2"}                 # ZeRO 阶段: 1 / 2 / 3
NUM_NODES=${2:-"1"}                  # 节点数量
GPUS_PER_NODE=${3:-"1"}              # 每节点 GPU 数
MASTER_ADDR=${4:-"localhost"}        # 主节点地址
MASTER_PORT=${5:-"29502"}            # 主节点端口
NODE_RANK=${NODE_RANK:-"0"}          # 当前节点 rank（通过环境变量传入）

WORLD_SIZE=$NUM_NODES
NPROC_PER_NODE=$GPUS_PER_NODE

# ================== 模型 & 数据 ==================
SFT_MODEL_PATH="../../output/sft_merge"
REWARD_MODEL_PATH="../../output/rm_merge"
TOKENIZER_PATH="../../output/tokenizers_merge"
TRAIN_FILE_DIR="../../data/finetune"
VALID_FILE_DIR="../../data/finetune"
OUTPUT_DIR="../../output/ppo_adapter_distributed"

# ================== 训练超参 ==================
TOTAL_EPISODES=8000
NUM_TRAIN_EPOCHS=1
EVAL_STEPS=100
MAX_SOURCE_LENGTH=256
RESPONSE_LENGTH=128
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING=True
REPORT_TO=none
EVAL_STRATEGY=steps
MISSING_EOS_PENALTY=1.0
DTYPE=bfloat16
TEMPLATE_NAME="qwen"

# ================== DeepSpeed 配置 ==================
case $ZERO_STAGE in
  1)
    DS_CONFIG="zero1.json"
    echo ">>> 使用 ZeRO-1"
    ;;
  2)
    DS_CONFIG="zero2.json"
    echo ">>> 使用 ZeRO-2"
    ;;
  3)
    DS_CONFIG="zero3.json"
    echo ">>> 使用 ZeRO-3"
    ;;
  *)
    echo "❌ 不支持的 ZeRO 阶段: $ZERO_STAGE"
    exit 1
    ;;
esac

if [ ! -f "$DS_CONFIG" ]; then
    echo "❌ DeepSpeed 配置文件不存在: $DS_CONFIG"
    exit 1
fi

# ================== 环境变量 ==================
export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))

# ================== 打印配置 ==================
echo "================================================="
echo " PPO 分布式训练配置"
echo "-------------------------------------------------"
echo " ZeRO Stage        : $ZERO_STAGE"
echo " Num Nodes         : $NUM_NODES"
echo " GPUs / Node       : $GPUS_PER_NODE"
echo " World Size        : $WORLD_SIZE"
echo " Node Rank         : $NODE_RANK"
echo " Master Addr       : $MASTER_ADDR"
echo " Master Port       : $MASTER_PORT"
echo " DeepSpeed Config  : $DS_CONFIG"
echo " Output Dir        : $OUTPUT_DIR"
echo "================================================="

mkdir -p "$OUTPUT_DIR"

# ================== torchrun 启动 ==================
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ppo_training.py \
    --sft_model_path $SFT_MODEL_PATH \
    --reward_model_path $REWARD_MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --template_name $TEMPLATE_NAME \
    --dtype $DTYPE \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $VALID_FILE_DIR \
    --max_source_length $MAX_SOURCE_LENGTH \
    --response_length $RESPONSE_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --do_train \
    --total_episodes $TOTAL_EPISODES \
    --output_dir $OUTPUT_DIR \
    --missing_eos_penalty $MISSING_EOS_PENALTY \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --report_to $REPORT_TO \
    --deepspeed $DS_CONFIG

echo "✅ 训练完成，输出目录：$OUTPUT_DIR"

#!/bin/bash
set -e

# =========================================================
# torchrun + DeepSpeed 多节点多 GPU ORPO 训练脚本
# 支持 ZeRO-1 / ZeRO-2 / ZeRO-3 (ZeRO-3 可选 CPU Offload)
#
# 使用方法：
# bash torchrun_multi_node_multi_gpu.sh [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port] [cpu_offload]
# =========================================================

# ================== 分布式参数 ==================
ZERO_STAGE=${1:-"3"}                 # ZeRO 阶段: 1 / 2 / 3
NUM_NODES=${2:-"1"}                  # 节点数量
GPUS_PER_NODE=${3:-"1"}              # 每节点 GPU 数
MASTER_ADDR=${4:-"localhost"}        # 主节点地址
MASTER_PORT=${5:-"29500"}            # 主节点端口
CPU_OFFLOAD=${6:-"false"}            # 是否启用 CPU Offload (仅 ZeRO-3 有效)
NODE_RANK=${NODE_RANK:-"0"}          # 当前节点 rank（通过环境变量传入）

WORLD_SIZE=$NUM_NODES
NPROC_PER_NODE=$GPUS_PER_NODE

# ================== 路径配置 ==================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="$PROJECT_ROOT/model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="$PROJECT_ROOT/output/tokenizers_merge"
TRAIN_FILE_DIR="$PROJECT_ROOT/data/reward"
OUTPUT_DIR="$PROJECT_ROOT/output/orpo_zero${ZERO_STAGE}"
CACHE_DIR="$PROJECT_ROOT/output/cache"

# ================== 训练超参 ==================
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

MAX_TRAIN_SAMPLES=1000
MAX_EVAL_SAMPLES=50
MAX_STEPS=100
EVAL_STEPS=20
SAVE_STEPS=50

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
    if [ "$CPU_OFFLOAD" = "true" ]; then
        echo ">>> ZeRO-3 启用 CPU Offload"
    fi
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
echo " torchrun + DeepSpeed ORPO 分布式训练配置"
echo "-------------------------------------------------"
echo " ZeRO Stage        : $ZERO_STAGE"
echo " Num Nodes         : $NUM_NODES"
echo " GPUs / Node       : $GPUS_PER_NODE"
echo " Node Rank         : $NODE_RANK"
echo " Master Addr       : $MASTER_ADDR"
echo " Master Port       : $MASTER_PORT"
echo " DeepSpeed Config  : $DS_CONFIG"
echo " CPU Offload       : $CPU_OFFLOAD"
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
  orpo_training.py \
  --model_name_or_path $MODEL_PATH \
  --tokenizer_name_or_path $TOKENIZER_PATH \
  --template_name qwen \
  --train_file_dir $TRAIN_FILE_DIR \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples $MAX_TRAIN_SAMPLES \
  --max_eval_samples $MAX_EVAL_SAMPLES \
  --max_steps $MAX_STEPS \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --output_dir $OUTPUT_DIR \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bf16 True \
  --fp16 False \
  --dtype bfloat16 \
  --gradient_checkpointing True \
  --remove_unused_columns False \
  --report_to none \
  --cache_dir $CACHE_DIR \
  --deepspeed $DS_CONFIG

echo "✅ ORPO 训练完成，输出目录：$OUTPUT_DIR"

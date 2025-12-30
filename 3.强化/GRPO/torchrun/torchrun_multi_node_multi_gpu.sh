#!/bin/bash
set -e

# =========================================================
# GRPO 多节点多 GPU 分布式训练脚本
# 基于 torchrun + DeepSpeed ZeRO1/ZeRO2/ZeRO3
# =========================================================

# ================== 分布式参数 ==================
ZERO_STAGE=${1:-"0"}                 # ZeRO阶段: 0(关闭) / 1 / 2 / 3
NUM_NODES=${2:-"1"}                  # 节点数量
GPUS_PER_NODE=${3:-"1"}              # 每节点 GPU 数量
MASTER_ADDR=${4:-"localhost"}        # 主节点地址
MASTER_PORT=${5:-"29501"}            # 主节点端口
NODE_RANK=${NODE_RANK:-"0"}          # 当前节点 rank（通过环境变量传入）

WORLD_SIZE=$NUM_NODES
NPROC_PER_NODE=$GPUS_PER_NODE

# ================== 模型 & 数据 ==================
MODEL_PATH="../../model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="../../model/Qwen/Qwen3-0.6B"
TRAIN_FILE_DIR="../../data/grpo"
OUTPUT_DIR="../../output/grpo_adapter_distributed_new"

# ================== 训练超参 ==================
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5.0e-7
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=13
LOGGING_STEPS=10
NUM_GENERATIONS=4

# ================== DeepSpeed 配置 ==================
DS_CONFIG=""
case "$ZERO_STAGE" in
    1)
        DS_CONFIG="zero1.json"
        ;;
    2)
        DS_CONFIG="zero2.json"
        ;;
    3)
        DS_CONFIG="zero3.json"
        ;;
    0)
        echo "❌ ZeRO关闭，不使用 DeepSpeed"
        ;;
    *)
        echo "❌ 不支持的 ZERO_STAGE: $ZERO_STAGE"
        exit 1
        ;;
esac

if [[ "$ZERO_STAGE" != "0" && ! -f "$DS_CONFIG" ]]; then
    echo "❌ DeepSpeed 配置文件不存在: $DS_CONFIG"
    exit 1
fi

# ================== 显示配置 ==================
echo "================================================="
echo " GRPO 分布式训练配置"
echo "-------------------------------------------------"
echo " ZeRO Stage         : $ZERO_STAGE"
echo " Num Nodes          : $NUM_NODES"
echo " GPUs / Node        : $GPUS_PER_NODE"
echo " World Size         : $WORLD_SIZE"
echo " Node Rank          : $NODE_RANK"
echo " Master Addr        : $MASTER_ADDR"
echo " Master Port        : $MASTER_PORT"
echo " Output Dir         : $OUTPUT_DIR"
if [[ "$ZERO_STAGE" != "0" ]]; then
    echo " DeepSpeed Config   : $DS_CONFIG"
else
    echo " DeepSpeed          : 未启用"
fi
echo "================================================="

mkdir -p "$OUTPUT_DIR"
export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE

if [[ "$NUM_NODES" -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))
fi

# ================== torchrun 启动 ==================
TORCHRUN_CMD="torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

DEEPSPEED_ARGS=""
if [[ "$ZERO_STAGE" != "0" ]]; then
    DEEPSPEED_ARGS="--deepspeed $DS_CONFIG"
fi

eval "$TORCHRUN_CMD grpo_training.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --train_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --num_generations $NUM_GENERATIONS \
    --report_to none \
    --output_dir $OUTPUT_DIR \
    $DEEPSPEED_ARGS"

echo "✅ 训练完成，输出目录: $OUTPUT_DIR"

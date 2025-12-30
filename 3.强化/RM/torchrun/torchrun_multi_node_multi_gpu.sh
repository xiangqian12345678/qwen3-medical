#!/bin/bash
set -e

# =========================================================
# RM 多节点多 GPU 分布式训练脚本
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="$PROJECT_ROOT/model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="$PROJECT_ROOT/output/tokenizers_merge"
TRAIN_FILE_DIR="$PROJECT_ROOT/data/reward"
VAL_FILE_DIR="$PROJECT_ROOT/data/reward"
OUTPUT_DIR="$PROJECT_ROOT/output/rm_adapter"
CACHE_DIR="$PROJECT_ROOT/output/cache"

# ================== 训练超参 ==================
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10

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
echo " RM 分布式训练配置"
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

# ================== 构建训练参数 ==================
# DeepSpeed + LoRA 组合时不支持 gradient_checkpointing，会导致梯度重复归约错误
if [[ "$ZERO_STAGE" != "0" ]]; then
    GRADIENT_CHECKPOINTING="False"
    echo "🔧 DeepSpeed 模式: 禁用 gradient_checkpointing (避免 LoRA 参数梯度重复归约)"
else
    GRADIENT_CHECKPOINTING="True"
fi

TRAIN_ARGS="--model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $VAL_FILE_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --report_to none \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --dtype bfloat16 \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING"

# ================== DeepSpeed 模式下移除 device_map ==================
if [[ "$ZERO_STAGE" != "0" ]]; then
    # 移除 --device_map 参数，因为 DeepSpeed 会自动管理设备分配
    TRAIN_ARGS=$(echo "$TRAIN_ARGS" | sed -e 's/--device_map auto//g' -e 's/--device_map=auto//g')
fi

# ================== 启动训练 ==================
if [[ "$ZERO_STAGE" != "0" ]]; then
    echo "使用DeepSpeed启动训练"
    TRAIN_CMD="deepspeed --num_gpus=$GPUS_PER_NODE \
        --num_nodes=$NUM_NODES \
        --master_port=$MASTER_PORT \
        --module reward_modeling \
        $TRAIN_ARGS \
        --deepspeed $DS_CONFIG"
else
    echo "使用torchrun启动训练"
    TRAIN_CMD="torchrun \
        --nnodes=$WORLD_SIZE \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        reward_modeling.py \
        $TRAIN_ARGS"
fi

echo "执行训练命令: $TRAIN_CMD"
eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成，输出目录: $OUTPUT_DIR"
    echo "训练日志: ${OUTPUT_DIR}/trainer_state.json"
    echo "模型权重: ${OUTPUT_DIR}/adapter_model.safetensors"
    echo "训练配置: ${OUTPUT_DIR}/adapter_config.json"
else
    echo ""
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi

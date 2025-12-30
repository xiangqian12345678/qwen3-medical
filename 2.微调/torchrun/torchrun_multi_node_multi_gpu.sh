#!/bin/bash
set -e

# =========================================================
# torchrun + DeepSpeed 多节点多 GPU SFT 微调脚本（LoRA）
# 支持 ZeRO-1 / ZeRO-2 / ZeRO-3
#
# 使用方法：
# bash torchrun_multi_node_multi_gpu.sh \
#     [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
# =========================================================

# ================== 分布式参数 ==================
ZERO_STAGE=${1:-"2"}                 # ZeRO 阶段: 1 / 2 / 3
NUM_NODES=${2:-"1"}                  # 节点数量
GPUS_PER_NODE=${3:-"4"}              # 每节点 GPU 数
MASTER_ADDR=${4:-"localhost"}        # 主节点地址
MASTER_PORT=${5:-"29500"}            # 主节点端口
NODE_RANK=${NODE_RANK:-"0"}           # 当前节点 rank（通过环境变量传入）

WORLD_SIZE=$NUM_NODES
NPROC_PER_NODE=$GPUS_PER_NODE

# ================== 模型 & 数据 ==================
MODEL_PATH="../model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="../output/tokenizers_merge"

TRAIN_FILE_DIR="../data/finetune"
VALID_FILE_DIR="../data/finetune"

OUTPUT_DIR="../output/sft_zero${ZERO_STAGE}"
CACHE_DIR="../output/cache"

# ================== 训练超参 ==================
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.05

MAX_TRAIN_SAMPLES=1000
MAX_EVAL_SAMPLES=10
MODEL_MAX_LENGTH=2048

# ================== LoRA 参数 ==================
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="all"

# ================== DeepSpeed 配置选择 ==================
case $ZERO_STAGE in
  1)
    DS_CONFIG="zero1.json"
    echo ">>> 使用 ZeRO-1（Optimizer Sharding）"
    ;;
  2)
    DS_CONFIG="zero2.json"
    echo ">>> 使用 ZeRO-2（Optimizer + Gradient Sharding）"
    ;;
  3)
    DS_CONFIG="zero3.json"
    echo ">>> 使用 ZeRO-3（Optimizer + Gradient + Parameter Sharding）"
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
echo " torchrun + DeepSpeed SFT 分布式训练配置"
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
  sft.py \
  --model_name_or_path $MODEL_PATH \
  --tokenizer_name_or_path $TOKENIZER_PATH \
  --train_file_dir $TRAIN_FILE_DIR \
  --validation_file_dir $VALID_FILE_DIR \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
  --do_train \
  --do_eval \
  --template_name qwen \
  --use_peft True \
  --target_modules $TARGET_MODULES \
  --lora_rank $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --max_train_samples $MAX_TRAIN_SAMPLES \
  --max_eval_samples $MAX_EVAL_SAMPLES \
  --model_max_length $MODEL_MAX_LENGTH \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 13 \
  --preprocessing_num_workers 4 \
  --ddp_timeout 30000 \
  --ddp_find_unused_parameters False \
  --logging_first_step True \
  --dtype bfloat16 \
  --bf16 \
  --gradient_checkpointing True \
  --flash_attn True \
  --report_to tensorboard \
  --cache_dir $CACHE_DIR \
  --remove_unused_columns True \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --deepspeed $DS_CONFIG

echo "✅ SFT 训练完成，输出目录：$OUTPUT_DIR"

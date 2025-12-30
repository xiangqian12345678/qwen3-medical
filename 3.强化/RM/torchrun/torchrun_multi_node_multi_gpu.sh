#!/bin/bash

# RM分布式训练脚本 - 基于torchrun实现多节点多GPU训练
# 
# 使用方法:
# 1. 单节点多GPU训练:
#    bash torchrun_multi_node_multi_gpu.sh
#
# 2. 多节点多GPU训练 (在主节点执行):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --master_port 29500
#
# 3. 多节点多GPU训练 (在工作节点执行):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --master_port 29500
#
# 4. 使用DeepSpeed ZeRO-2训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 
#
# 5. 使用DeepSpeed ZeRO-3训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4  --deepspeed_config ds_config_zero3.json
#
# 6. 使用DeepSpeed ZeRO-3 + CPU Offload:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4  --deepspeed_config ds_config_zero3_cpu_offload.json
#
# 环境变量说明:
# - CUDA_VISIBLE_DEVICES: 指定可见的GPU设备
# - WORLD_SIZE: 总进程数
# - RANK: 当前进程的全局排名
# - LOCAL_RANK: 当前进程在节点内的本地排名
# - MASTER_ADDR: 主节点IP地址
# - MASTER_PORT: 主节点端口号

# 默认参数设置
NNODES=${NNODES:-1}                    # 节点数量，默认为1（单节点）
NPROC_PER_NODE=${NPROC_PER_NODE:-1}     # 每个节点的进程数（GPU数量），默认为1
NODE_RANK=${NODE_RANK:-0}               # 当前节点的排名，默认为0（主节点）
MASTER_ADDR=${MASTER_ADDR:-localhost}   # 主节点地址，默认为本地
MASTER_PORT=${MASTER_PORT:-29501}       # 主节点端口，默认为29501（避免与DPO端口冲突）
USE_DEEPSPEED=${USE_DEEPSPEED:-false}   # 是否使用DeepSpeed，默认为false
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"ds_config.json"}  # DeepSpeed配置文件路径

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --use_deepspeed)
            USE_DEEPSPEED="$2"
            shift 2
            ;;
        --deepspeed_config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --nnodes NNODES         节点数量 (默认: 1)"
            echo "  --nproc_per_node NPROC  每个节点的进程数/GPU数量 (默认: 1)"
            echo "  --node_rank RANK        当前节点排名 (默认: 0)"
            echo "  --master_addr ADDR      主节点地址 (默认: localhost)"
            echo "  --master_port PORT      主节点端口 (默认: 29501)"
            echo "  --use_deepspeed BOOL    是否使用DeepSpeed (默认: false)"
            echo "  --deepspeed_config PATH DeepSpeed配置文件路径 (默认: ds_config.json)"
            echo "  -h, --help              显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 单节点4GPU训练"
            echo "  $0 --nproc_per_node 4"
            echo ""
            echo "  # 多节点训练 (主节点)"
            echo "  $0 --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100"
            echo ""
            echo "  # 多节点训练 (工作节点)"
            echo "  $0 --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100"
            echo ""
            echo "  # 使用DeepSpeed ZeRO-2训练"
            echo "  $0 --nproc_per_node 4 "
            echo ""
            echo "  # 使用DeepSpeed ZeRO-3训练"
            echo "  $0 --nproc_per_node 4  --deepspeed_config ds_config_zero3.json"
            echo ""
            echo "  # 使用DeepSpeed ZeRO-3 + CPU Offload"
            echo "  $0 --nproc_per_node 4  --deepspeed_config ds_config_zero3_cpu_offload.json"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 计算总进程数
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# 设置GPU可见性（自动选择前NPROC_PER_NODE个GPU）
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "使用用户指定的CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    # 自动检测可用GPU数量
    AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$AVAILABLE_GPUS" -lt "$NPROC_PER_NODE" ]; then
        echo "错误: 检测到 $AVAILABLE_GPUS 个GPU，但请求使用 $NPROC_PER_NODE 个GPU"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE-1)))
    export CUDA_VISIBLE_DEVICES
    echo "自动设置CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# 显示分布式训练配置
echo "=== RM分布式训练配置 ==="
echo "节点数量: $NNODES"
echo "每个节点进程数: $NPROC_PER_NODE"
echo "总进程数: $WORLD_SIZE"
echo "当前节点排名: $NODE_RANK"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "使用DeepSpeed: $USE_DEEPSPEED"
if [ "$USE_DEEPSPEED" = "true" ]; then
    echo "DeepSpeed配置: $DEEPSPEED_CONFIG"
fi
echo "======================"

# 基础训练参数 - 使用绝对路径避免WSL路径解析问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="$PROJECT_ROOT/output/tokenizers_merge"
TRAIN_DATA_DIR="$PROJECT_ROOT/data/reward"
VAL_DATA_DIR="$PROJECT_ROOT/data/reward"
OUTPUT_DIR="$PROJECT_ROOT/output/rm_adapter"
CACHE_DIR="$PROJECT_ROOT/output/cache"

# 验证关键路径是否存在
echo "检查关键路径..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "错误: Tokenizer路径不存在: $TOKENIZER_PATH"
    exit 1
fi
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "错误: 训练数据路径不存在: $TRAIN_DATA_DIR"
    exit 1
fi
echo "路径检查完成"

# 分布式训练参数调整
# 在分布式训练中，通常需要调整batch size等参数以获得最佳性能
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

# 计算有效batch size
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE))
echo "有效训练batch size: $EFFECTIVE_BATCH_SIZE"

# 构建训练命令参数
TRAIN_ARGS="--model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --train_file_dir $TRAIN_DATA_DIR \
    --validation_file_dir $VAL_DATA_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --dtype bfloat16 \
    --device_map auto \
    --report_to none \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir $CACHE_DIR"

# 根据是否使用DeepSpeed选择不同的启动命令
if [ "$USE_DEEPSPEED" = "true" ]; then
    # 检查DeepSpeed配置文件是否存在
    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo "错误: DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
        exit 1
    fi
    echo "使用DeepSpeed启动命令"
    # 使用deepspeed命令启动训练（不带.py扩展名）
    TRAIN_CMD="deepspeed --num_gpus=$NPROC_PER_NODE \
        --num_nodes=$NNODES \
        --master_port=$MASTER_PORT \
        --module reward_modeling \
        $TRAIN_ARGS"
    if [ "$NNODES" -gt 1 ]; then
        # 多节点训练需要设置master_addr
        export MASTER_ADDR=$MASTER_ADDR
    fi
else
    echo "使用torchrun启动命令"
    # 使用torchrun命令启动训练
    TRAIN_CMD="torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        reward_modeling.py \
        $TRAIN_ARGS"
fi

# 启动训练
echo "执行训练命令: $TRAIN_CMD"
eval $TRAIN_CMD

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "模型保存至: $OUTPUT_DIR"
    echo "可查看以下文件:"
    echo "  - 训练日志: ${OUTPUT_DIR}/trainer_state.json"
    echo "  - 模型权重: ${OUTPUT_DIR}/adapter_model.safetensors"
    echo "  - 训练配置: ${OUTPUT_DIR}/adapter_config.json"
    echo "==============="
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查错误信息并调整参数"
    echo "==============="
    exit 1
fi

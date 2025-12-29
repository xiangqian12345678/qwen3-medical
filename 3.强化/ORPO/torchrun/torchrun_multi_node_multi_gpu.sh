#!/bin/bash

# ORPO分布式训练脚本 - 基于torchrun实现多节点多GPU训练
# 支持DeepSpeed Zero2和Zero3模式，Zero3支持CPU Offload
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
# 4. 使用DeepSpeed Zero2训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --deepspeed_stage 2
#
# 5. 使用DeepSpeed Zero3训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --deepspeed_stage 3
#
# 6. 使用DeepSpeed Zero3 + CPU Offload:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --deepspeed_stage 3 --cpu_offload true
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
MASTER_PORT=${MASTER_PORT:-29500}       # 主节点端口，默认为29500
USE_DEEPSPEED=${USE_DEEPSPEED:-false}   # 是否使用DeepSpeed，默认为false
DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-3}  # DeepSpeed ZeRO阶段，默认为3
CPU_OFFLOAD=${CPU_OFFLOAD:-false}      # 是否启用CPU Offload，默认为false

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
        --deepspeed_stage)
            DEEPSPEED_STAGE="$2"
            USE_DEEPSPEED="true"
            shift 2
            ;;
        --cpu_offload)
            CPU_OFFLOAD="$2"
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
            echo "  --master_port PORT      主节点端口 (默认: 29500)"
            echo "  --use_deepsPEED BOOL    是否使用DeepSpeed (默认: false)"
            echo "  --deepspeed_stage STAGE DeepSpeed ZeRO阶段 (2或3，默认: 3)"
            echo "  --cpu_offload BOOL      是否启用CPU Offload (默认: false)"
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
            echo "  $0 --nproc_per_node 4 --deepspeed_stage 2"
            echo ""
            echo "  # 使用DeepSpeed ZeRO-3 + CPU Offload训练"
            echo "  $0 --nproc_per_node 4 --deepspeed_stage 3 --cpu_offload true"
            echo ""
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证DeepSpeed参数
if [ "$USE_DEEPSPEED" = "true" ]; then
    if [ "$DEEPSPEED_STAGE" != "2" ] && [ "$DEEPSPEED_STAGE" != "3" ]; then
        echo "错误: DeepSpeed阶段必须是2或3"
        exit 1
    fi
    if [ "$DEEPSPEED_STAGE" != "3" ] && [ "$CPU_OFFLOAD" = "true" ]; then
        echo "警告: CPU Offload仅在ZeRO-3阶段有效"
    fi
fi

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
echo "=== ORPO分布式训练配置 ==="
echo "节点数量: $NNODES"
echo "每个节点进程数: $NPROC_PER_NODE"
echo "总进程数: $WORLD_SIZE"
echo "当前节点排名: $NODE_RANK"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "使用DeepSpeed: $USE_DEEPSPEED"
if [ "$USE_DEEPSPEED" = "true" ]; then
    echo "DeepSpeed ZeRO阶段: $DEEPSPEED_STAGE"
    echo "CPU Offload: $CPU_OFFLOAD"
fi
echo "=========================="

# 基础训练参数 - 使用绝对路径避免WSL路径解析问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="$PROJECT_ROOT/output/tokenizers_merge"
TRAIN_DATA_DIR="$PROJECT_ROOT/data/reward"
OUTPUT_DIR="$PROJECT_ROOT/output/orpo"
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
GRADIENT_ACCUMULATION_STEPS=4

# 计算有效batch size
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE))
echo "有效训练batch size: $EFFECTIVE_BATCH_SIZE"

# 生成DeepSpeed配置文件
if [ "$USE_DEEPSPEED" = "true" ]; then
    DS_CONFIG_FILE="ds_config_zero${DEEPSPEED_STAGE}.json"
    
    # 根据不同阶段生成配置
    if [ "$DEEPSPEED_STAGE" = "2" ]; then
        cat > "$DS_CONFIG_FILE" << EOF
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true
}
EOF
    elif [ "$DEEPSPEED_STAGE" = "3" ]; then
        if [ "$CPU_OFFLOAD" = "true" ]; then
            cat > "$DS_CONFIG_FILE" << EOF
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "cpu_offload": true,
        "cpu_offload_params": {
            "pin_memory": true
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true
}
EOF
        else
            cat > "$DS_CONFIG_FILE" << EOF
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true
}
EOF
        fi
    fi
    
    echo "生成DeepSpeed配置文件: $DS_CONFIG_FILE"
fi

# 构建torchrun命令参数数组（只包含torchrun原生参数）
TORCHRUN_ARGS=(
    "--nnodes=$NNODES"
    "--nproc_per_node=$NPROC_PER_NODE"
    "--node_rank=$NODE_RANK"
    "--master_addr=$MASTER_ADDR"
    "--master_port=$MASTER_PORT"
)

# 构建训练脚本参数数组
TRAINING_ARGS=(
    "$SCRIPT_DIR/orpo_training.py"
    "--model_name_or_path=$MODEL_PATH"
    "--tokenizer_name_or_path=$TOKENIZER_PATH"
    "--template_name=qwen"
    "--train_file_dir=$TRAIN_DATA_DIR"
    "--per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
    "--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    "--per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE"
    "--do_train"
    "--do_eval"
    "--use_peft=True"
    "--max_train_samples=1000"
    "--max_eval_samples=50"
    "--max_steps=100"
    "--eval_steps=20"
    "--save_steps=50"
    "--max_source_length=1024"
    "--max_target_length=512"
    "--output_dir=$OUTPUT_DIR"
    "--target_modules=all"
    "--lora_rank=8"
    "--lora_alpha=16"
    "--lora_dropout=0.05"
    "--dtype=bfloat16"
    "--bf16=True"
    "--fp16=False"
    "--device_map=auto"
    "--report_to=none"
    "--remove_unused_columns=False"
    "--gradient_checkpointing=True"
    "--orpo_beta=0.1"
    "--cache_dir=$CACHE_DIR"
    "--logging_steps=1"
)

# 如果使用DeepSpeed，添加deepspeed参数到训练脚本参数中
if [ "$USE_DEEPSPEED" = "true" ]; then
    TRAINING_ARGS+=("--deepspeed=$DS_CONFIG_FILE")
fi

# 启动训练
echo "执行训练命令: torchrun ${TORCHRUN_ARGS[*]} ${TRAINING_ARGS[*]}"
torchrun "${TORCHRUN_ARGS[@]}" "${TRAINING_ARGS[@]}"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "模型保存至: $OUTPUT_DIR"
    echo "可查看以下文件:"
    echo "  - 训练日志: ${OUTPUT_DIR}/training_log.txt"
    echo "  - 模型权重: ${OUTPUT_DIR}/adapter_model.bin"
    echo "  - 训练配置: ${OUTPUT_DIR}/adapter_config.json"
    if [ "$USE_DEEPSPEED" = "true" ]; then
        echo "  - DeepSpeed配置: $DS_CONFIG_FILE"
    fi
    echo "==============="
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查错误信息并调整参数"
    echo "==============="
    exit 1
fi
#!/bin/bash

# PPO分布式训练脚本 - 基于torchrun实现多节点多GPU训练
# 支持DeepSpeed ZeRO2和ZeRO3模式，ZeRO3支持CPU Offload
# 
# 使用方法:
# 1. 单节点多GPU训练:
#    bash torchrun_multi_node_multi_gpu.sh
#
# 2. 多节点多GPU训练 (在主节点执行):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --master_port 29502
#
# 3. 多节点多GPU训练 (在工作节点执行):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --master_port 29502
#
# 4. 使用DeepSpeed ZeRO-2训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 2
#
# 5. 使用DeepSpeed ZeRO-3训练:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 3
#
# 6. 使用DeepSpeed ZeRO-3 + CPU Offload:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 3 --cpu_offload true
#
# 环境变量说明:
# - CUDA_VISIBLE_DEVICES: 指定可见的GPU设备
# - WORLD_SIZE: 总进程数
# - RANK: 当前进程的全局排名
# - LOCAL_RANK: 当前进程在节点内的本地排名
# - MASTER_ADDR: 主节点IP地址
# - MASTER_PORT: 主节点端口号

set -e

# ======== 默认配置参数 ========
# 分布式配置
NNODES=${NNODES:-1}                           # 节点数量，默认为1（单节点）
NPROC_PER_NODE=${NPROC_PER_NODE:-1}           # 每个节点的进程数（GPU数量），默认为1
NODE_RANK=${NODE_RANK:-0}                     # 当前节点的排名，默认为0（主节点）
MASTER_ADDR=${MASTER_ADDR:-localhost}         # 主节点地址，默认为本地
MASTER_PORT=${MASTER_PORT:-29502}             # 主节点端口，默认为29502

# DeepSpeed配置
ZERO_STAGE=${ZERO_STAGE:-0}                   # ZeRO阶段: 0(关闭), 2(ZeRO2), 3(ZeRO3)
CPU_OFFLOAD=${CPU_OFFLOAD:-false}             # CPU卸载(仅ZeRO3支持)

# 模型和数据配置
SFT_MODEL_PATH="../../output/sft_merge"
REWARD_MODEL_PATH="../../output/rm_merge"
TOKENIZER_NAME_OR_PATH="../../output/tokenizers_merge"
TRAIN_FILE_DIR="../../data/finetune"
VALIDATION_FILE_DIR="../../data/finetune"
OUTPUT_DIR="../../output/ppo_adapter_distributed"

# 训练超参数
TOTAL_EPISODES=8000
NUM_TRAIN_EPOCHS=1
EVAL_STEPS=100
MAX_SOURCE_LENGTH=256
RESPONSE_LENGTH=128

# 数据类型和精度
DTYPE="bfloat16"
TEMPLATE_NAME="qwen"

# PPO算法参数
MISSING_EOS_PENALTY=1.0

# 分布式训练参数调整
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING="False"
REPORT_TO="none"
EVAL_STRATEGY="steps"

# ======== 解析命令行参数 ========
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
        --zero_stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --cpu_offload)
            CPU_OFFLOAD="$2"
            shift 2
            ;;
        --sft_model_path)
            SFT_MODEL_PATH="$2"
            shift 2
            ;;
        --reward_model_path)
            REWARD_MODEL_PATH="$2"
            shift 2
            ;;
        --tokenizer_name_or_path)
            TOKENIZER_NAME_OR_PATH="$2"
            shift 2
            ;;
        --train_file_dir)
            TRAIN_FILE_DIR="$2"
            shift 2
            ;;
        --validation_file_dir)
            VALIDATION_FILE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --total_episodes)
            TOTAL_EPISODES="$2"
            shift 2
            ;;
        --max_source_length)
            MAX_SOURCE_LENGTH="$2"
            shift 2
            ;;
        --response_length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "分布式配置:"
            echo "  --nnodes N                节点数量 (默认: 1)"
            echo "  --nproc_per_node N        每节点GPU数量 (默认: 1)"
            echo "  --node_rank N             当前节点排名 (默认: 0)"
            echo "  --master_addr ADDR        主节点地址 (默认: localhost)"
            echo "  --master_port PORT        主节点端口 (默认: 29502)"
            echo ""
            echo "DeepSpeed配置:"
            echo "  --zero_stage N            ZeRO阶段: 0(关闭), 2(ZeRO2), 3(ZeRO3) (默认: 0)"
            echo "  --cpu_offload BOOL        CPU卸载 (仅ZeRO3) (默认: false)"
            echo ""
            echo "模型配置:"
            echo "  --sft_model_path PATH     SFT模型路径"
            echo "  --reward_model_path PATH  奖励模型路径"
            echo "  --tokenizer_name_or_path PATH  Tokenizer路径"
            echo "  --train_file_dir PATH     训练数据路径"
            echo "  --validation_file_dir PATH 评估数据路径"
            echo "  --output_dir PATH         输出目录"
            echo ""
            echo "训练配置:"
            echo "  --per_device_train_batch_size N  批次大小 (默认: 1)"
            echo "  --gradient_accumulation_steps N  梯度累积步数 (默认: 4)"
            echo "  --total_episodes N        总回合数 (默认: 8000)"
            echo "  --max_source_length N     源文本最大长度 (默认: 256)"
            echo "  --response_length N       响应最大长度 (默认: 128)"
            echo ""
            echo "示例:"
            echo "  # 单节点多GPU训练:"
            echo "  bash $0 --nproc_per_node 4"
            echo ""
            echo "  # 使用DeepSpeed ZeRO-2:"
            echo "  bash $0 --nproc_per_node 4 --zero_stage 2"
            echo ""
            echo "  # 多节点多GPU训练 (主节点):"
            echo "  bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --zero_stage 2"
            echo ""
            echo "  # 多节点多GPU训练 (工作节点):"
            echo "  bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --zero_stage 2"
            echo ""
            echo "  # 使用DeepSpeed ZeRO-2:"
            echo "  bash $0 --nproc_per_node 4 --zero_stage 2"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ======== 验证参数 ========
if [[ ! "$ZERO_STAGE" =~ ^[023]$ ]]; then
    echo "错误: zero_stage 必须是 0, 2, 或 3"
    exit 1
fi

if [[ "$CPU_OFFLOAD" != "true" && "$CPU_OFFLOAD" != "false" ]]; then
    echo "错误: cpu_offload 必须是 true 或 false"
    exit 1
fi

if [ "$ZERO_STAGE" != "3" ] && [ "$CPU_OFFLOAD" = "true" ]; then
    echo "警告: CPU Offload仅在ZeRO-3阶段有效"
fi

# ======== 创建输出目录 ========
mkdir -p "$OUTPUT_DIR"

# ======== 设置GPU可见性 ========
if [ "$NNODES" -eq 1 ]; then
    # 单节点情况，设置可见的GPU
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
else
    # 多节点情况，通常每个节点都能看到所有GPU，由torchrun自动处理
    echo "多节点训练，将使用torchrun自动分配GPU"
fi

# ======== 生成DeepSpeed配置文件 ========
DEEPSPEED_CONFIG=""
if [ "$ZERO_STAGE" != "0" ]; then
    DEEPSPEED_CONFIG="zero${ZERO_STAGE}_config.json"
    echo "生成DeepSpeed配置文件: $DEEPSPEED_CONFIG"
    
    # 转换CPU_OFFLOAD为JSON布尔值
    if [ "$CPU_OFFLOAD" = "true" ]; then
        CPU_OFFLOAD_JSON="true"
    else
        CPU_OFFLOAD_JSON="false"
    fi
    
    # 计算总batch size
    TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))
    
    # 生成DeepSpeed配置
    cat > "$DEEPSPEED_CONFIG" << EOF
{
    "steps_per_print": 10,
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true,
    "zero_force_ds_cpu_optimizer": false,
    
    "bf16": {
        "enabled": true
    },
    
    "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
    "gradient_clipping": 1.0,
    
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
EOF

    if [ "$ZERO_STAGE" == "2" ]; then
        cat >> "$DEEPSPEED_CONFIG" << EOF
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "cpu_offload": $CPU_OFFLOAD_JSON
    }
}
EOF
    elif [ "$ZERO_STAGE" == "3" ]; then
        cat >> "$DEEPSPEED_CONFIG" << EOF
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "cpu_offload": $CPU_OFFLOAD_JSON,
        "stage3_max_live_parameters": 1000000000,
        "stage3_max_reuse_distance": 1000000000,
        "stage3_param_persistence_threshold": 1000000000,
        "stage3_gather_16bit_weights_on_model_save": true,
        "sub_group_size": 1000000000
    }
}
EOF
    fi
    echo "DeepSpeed配置文件已生成: $DEEPSPEED_CONFIG"
else
    echo "不使用DeepSpeed ZeRO优化"
fi

# ======== 显示配置信息 ========
echo "=== PPO 分布式训练配置 ==="
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "节点数: $NNODES"
echo "当前节点排名: $NODE_RANK"
echo "每节点GPU数量: $NPROC_PER_NODE"
echo "SFT模型路径: $SFT_MODEL_PATH"
echo "奖励模型路径: $REWARD_MODEL_PATH"
echo "Tokenizer路径: $TOKENIZER_NAME_OR_PATH"
echo "训练数据路径: $TRAIN_FILE_DIR"
echo "输出目录: $OUTPUT_DIR"

if [ "$ZERO_STAGE" != "0" ]; then
    echo "DeepSpeed ZeRO-${ZERO_STAGE}: 启用"
    echo "CPU Offload: $CPU_OFFLOAD"
    echo "DeepSpeed配置文件: $DEEPSPEED_CONFIG"
else
    echo "DeepSpeed: 未启用"
fi

echo "批次大小: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "梯度累积步数: $GRADIENT_ACCUMULATION_STEPS"
echo "有效批次大小: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))"
echo "总回合数: $TOTAL_EPISODES"
echo "最大源长度: $MAX_SOURCE_LENGTH"
echo "响应长度: $RESPONSE_LENGTH"
echo "================================"

# ======== 构建torchrun命令 ========
TORCHRUN_CMD="torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

# ======== 构建DeepSpeed参数 ========
DEEPSPEED_ARGS=""
if [ "$ZERO_STAGE" != "0" ]; then
    DEEPSPEED_ARGS="--deepspeed $DEEPSPEED_CONFIG"
fi

# ======== 设置分布式训练环境变量 ========
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ======== 执行训练 ========
echo "执行训练命令..."
eval "$TORCHRUN_CMD ../PPO/ppo_training.py \
    --sft_model_path \"$SFT_MODEL_PATH\" \
    --reward_model_path \"$REWARD_MODEL_PATH\" \
    --tokenizer_name_or_path \"$TOKENIZER_NAME_OR_PATH\" \
    --template_name $TEMPLATE_NAME \
    --dtype $DTYPE \
    --train_file_dir \"$TRAIN_FILE_DIR\" \
    --validation_file_dir \"$VALIDATION_FILE_DIR\" \
    --max_source_length $MAX_SOURCE_LENGTH \
    --response_length $RESPONSE_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --do_train \
    --total_episodes $TOTAL_EPISODES \
    --output_dir \"$OUTPUT_DIR\" \
    --missing_eos_penalty $MISSING_EOS_PENALTY \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --report_to $REPORT_TO \
    $DEEPSPEED_ARGS"

# ======== 检查训练结果 ========
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "模型保存至: $OUTPUT_DIR"
    echo "==============="
else
    echo ""
    echo "=== 训练失败 ==="
    echo "请检查错误信息并调整参数"
    echo "==============="
    exit 1
fi

# ======== 清理临时文件 ========
if [ "$ZERO_STAGE" != "0" ] && [ -f "$DEEPSPEED_CONFIG" ]; then
    echo "保留DeepSpeed配置文件: $DEEPSPEED_CONFIG"
fi

# ======== 使用说明 ========
echo ""
echo "=== 使用说明 ==="
echo "1. 单节点多GPU训练:"
echo "   bash $0 --nproc_per_node 4"
echo ""
echo "2. 使用DeepSpeed ZeRO-2:"
echo "   bash $0 --nproc_per_node 4 --zero_stage 2"
echo ""
echo "3. 使用DeepSpeed ZeRO-3 + CPU Offload:"
echo "   bash $0 --nproc_per_node 4 --zero_stage 3 --cpu_offload true"
echo ""
echo "4. 多节点多GPU训练 (主节点):"
echo "   bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --zero_stage 2"
echo ""
echo "5. 多节点多GPU训练 (工作节点):"
echo "   bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --zero_stage 2"
echo ""
echo "=== 注意事项 ==="
echo "1. ZeRO-2: 优化器状态分片，减少显存使用"
echo "2. ZeRO-3: 参数、梯度、优化器状态全部分片，最大显存节省"
echo "3. CPU Offload: 将参数/优化器转移到CPU内存，进一步节省显存（仅ZeRO-3）"
echo "4. 使用ZeRO-3时，建议开启cpu_offload=true以获得最佳显存优化"
echo "5. DeepSpeed需要安装: pip install deepspeed"

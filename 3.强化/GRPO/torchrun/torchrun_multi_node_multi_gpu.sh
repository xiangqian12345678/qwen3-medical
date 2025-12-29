#!/bin/bash

# GRPO 多节点多GPU分布式训练脚本 - 支持DeepSpeed ZeRO2/ZeRO3
# 基于 torchrun 实现，支持 DeepSpeed ZeRO优化
# 
# 使用方法:
# 1. 单节点多GPU训练:
#    bash torchrun_multi_node_multi_gpu.sh
#
# 2. 多节点多GPU训练 (主节点):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100
#
# 3. 多节点多GPU训练 (工作节点):
#    bash torchrun_multi_node_multi_gpu.sh --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100
#
# 4. 使用DeepSpeed ZeRO-2:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 2
#
# 5. 使用DeepSpeed ZeRO-3:
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 3
#
# 6. 启用CPU Offload (仅ZeRO-3):
#    bash torchrun_multi_node_multi_gpu.sh --nproc_per_node 4 --zero_stage 3 --cpu_offload true

set -e

# ======== 默认配置参数 ========
# 分布式配置
NNODES=${NNODES:-1}                           # 节点数量
NPROC_PER_NODE=${NPROC_PER_NODE:-1}           # 每节点GPU数量
NODE_RANK=${NODE_RANK:-0}                     # 当前节点排名
MASTER_ADDR=${MASTER_ADDR:-localhost}         # 主节点地址
MASTER_PORT=${MASTER_PORT:-29501}             # 主节点端口

# DeepSpeed配置
ZERO_STAGE=${ZERO_STAGE:-0}                   # ZeRO阶段: 0(关闭), 2(ZeRO2), 3(ZeRO3)
CPU_OFFLOAD=${CPU_OFFLOAD:-false}             # CPU卸载(仅ZeRO3支持)
OFFLOAD_PARAM_DEVICE=${OFFLOAD_PARAM_DEVICE:-none}  # 参数卸载设备: none/cpu/nvme
OFFLOAD_OPTIMIZER_DEVICE=${OFFLOAD_OPTIMIZER_DEVICE:-none}  # 优化器卸载设备

# 模型和数据配置
MODEL_PATH="../../model/Qwen/Qwen3-0.6B"
TOKENIZER_PATH="../../model/Qwen/Qwen3-0.6B"
TRAIN_FILE_DIR="../../data/grpo"
OUTPUT_DIR="../../output/grpo_adapter_distributed_new"

# 训练超参数
TRAIN_SAMPLES=-1
MAX_STEPS=-1
NUM_TRAIN_EPOCHS=1
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=13

# 数据类型和精度
DTYPE="bfloat16"
BF16="True"

# GRPO算法参数
BETA=0.001
LEARNING_RATE=5.0e-7
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.03
USE_VLLM="False"
LOGGING_STEPS=10

# QLoRA配置
USE_PEFT="True"
QLORA="True"
LOAD_IN_4BIT="True"
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 显存优化配置
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
NUM_GENERATIONS=4
GRADIENT_ACCUMULATION_STEPS=1
MAX_PROMPT_LENGTH=16384
MAX_COMPLETION_LENGTH=512

# 其他配置
REMOVE_UNUSED_COLUMNS="False"
GRADIENT_CHECKPOINTING="False"
REPORT_TO="tensorboard"

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
        --offload_param_device)
            OFFLOAD_PARAM_DEVICE="$2"
            shift 2
            ;;
        --offload_optimizer_device)
            OFFLOAD_OPTIMIZER_DEVICE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --train_file_dir)
            TRAIN_FILE_DIR="$2"
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
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --zero_mode)
            ZERO_STAGE="$2"
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
            echo "  --master_port PORT        主节点端口 (默认: 29501)"
            echo ""
            echo "DeepSpeed配置:"
            echo "  --zero_stage N            ZeRO阶段: 0(关闭), 2(ZeRO2), 3(ZeRO3) (默认: 0)"
            echo "  --cpu_offload BOOL        CPU卸载 (仅ZeRO3) (默认: false)"
            echo "  --offload_param_device DEVICE   参数卸载设备: none/cpu/nvme (默认: none)"
            echo "  --offload_optimizer_device DEVICE 优化器卸载设备: none/cpu/nvme (默认: none)"
            echo "  --zero_mode N             ZeRO模式别名 (兼容旧版本)"
            echo ""
            echo "模型配置:"
            echo "  --model_path PATH         模型路径"
            echo "  --train_file_dir PATH     训练数据路径"
            echo "  --output_dir PATH         输出目录"
            echo "  --per_device_train_batch_size N  批次大小 (默认: 4)"
            echo "  --learning_rate N         学习率 (默认: 5.0e-7)"
            echo ""
            echo "示例:"
            echo "  # 单节点4GPU，不使用DeepSpeed"
            echo "  $0 --nproc_per_node 4"
            echo ""
            echo "  # 单节点4GPU，使用ZeRO-2"
            echo "  $0 --nproc_per_node 4 --zero_stage 2"
            echo ""
            echo "  # 单节点4GPU，使用ZeRO-3 + CPU Offload"
            echo "  $0 --nproc_per_node 4 --zero_stage 3 --cpu_offload true"
            echo ""
            echo "  # 多节点训练 (主节点)"
            echo "  $0 --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --zero_stage 3"
            echo ""
            echo "  # 多节点训练 (工作节点)"
            echo "  $0 --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --zero_stage 3"
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

# ======== DeepSpeed ZeRO-3 兼容性检查 ========
# 当启用 DeepSpeed ZeRO-3 时，禁用量化以确保兼容性
if [ "$ZERO_STAGE" == "3" ]; then
    if [ "$LOAD_IN_4BIT" == "True" ] || [ "$QLORA" == "True" ]; then
        echo "警告: DeepSpeed ZeRO-3 与量化不兼容，自动禁用量化"
        echo "  - 禁用 4-bit 量化 (LOAD_IN_4BIT: True -> False)"
        echo "  - 禁用 QLoRA (QLORA: True -> False)"
        LOAD_IN_4BIT="False"
        QLORA="False"
    fi
fi

# ======== 创建DeepSpeed配置文件 ========
DEEPSPEED_CONFIG="zero${ZERO_STAGE}_config.json"
if [ "$ZERO_STAGE" != "0" ]; then
    echo "生成DeepSpeed配置文件: $DEEPSPEED_CONFIG"
    
    # 计算总batch size
    TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))
    
    # 转换CPU_OFFLOAD为JSON布尔值
    if [ "$CPU_OFFLOAD" = "true" ]; then
        CPU_OFFLOAD_JSON="true"
    else
        CPU_OFFLOAD_JSON="false"
    fi
    
    # 生成DeepSpeed配置
    cat > "$DEEPSPEED_CONFIG" << EOF
{
    "steps_per_print": $LOGGING_STEPS,
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true,
    "zero_force_ds_cpu_optimizer": false,
    
    "bf16": {
        "enabled": $(echo "$BF16" | tr '[:upper:]' '[:lower:]')
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
echo "=== GRPO 分布式训练配置 ==="
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "节点数: $NNODES"
echo "当前节点排名: $NODE_RANK"
echo "每节点GPU数量: $NPROC_PER_NODE"
echo "模型路径: $MODEL_PATH"
echo "训练数据路径: $TRAIN_FILE_DIR"
echo "输出目录: $OUTPUT_DIR"

if [ "$ZERO_STAGE" != "0" ]; then
    echo "DeepSpeed ZeRO-${ZERO_STAGE}: 启用"
    echo "CPU Offload: $CPU_OFFLOAD"
    echo "参数卸载设备: $OFFLOAD_PARAM_DEVICE"
    echo "优化器卸载设备: $OFFLOAD_OPTIMIZER_DEVICE"
    echo "DeepSpeed配置文件: $DEEPSPEED_CONFIG"
else
    echo "DeepSpeed: 未启用"
fi

echo "QLoRA启用: $QLORA"
echo "4bit量化: $LOAD_IN_4BIT"
echo "批次大小: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "梯度累积步数: $GRADIENT_ACCUMULATION_STEPS"
echo "有效批次大小: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE * NNODES))"
echo "================================"

# ======== 创建输出目录 ========
mkdir -p "$OUTPUT_DIR"

# ======== 设置可见的GPU ========
if [ "$NNODES" -eq 1 ]; then
    # 单节点情况，设置可见的GPU
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE-1)))
else
    # 多节点情况，通常每个节点都能看到所有GPU，由torchrun自动处理
    echo "多节点训练，将使用torchrun自动分配GPU"
fi

# ======== 构建torchrun命令 ========
TORCHRUN_CMD="torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

# ======== 执行训练 ========
# 构建 DeepSpeed 参数传递给训练脚本
DEEPSPEED_ARGS=""
if [ "$ZERO_STAGE" != "0" ]; then
    DEEPSPEED_ARGS="--deepspeed $DEEPSPEED_CONFIG"
fi

eval "$TORCHRUN_CMD grpo_training.py \\
    --model_name_or_path \"$MODEL_PATH\" \\
    --tokenizer_name_or_path \"$TOKENIZER_PATH\" \\
    --train_file_dir \"$TRAIN_FILE_DIR\" \\
    --train_samples $TRAIN_SAMPLES \\
    --max_steps $MAX_STEPS \\
    --num_train_epochs $NUM_TRAIN_EPOCHS \\
    --save_steps $SAVE_STEPS \\
    --save_strategy steps \\
    --save_total_limit $SAVE_TOTAL_LIMIT \\
    --output_dir \"$OUTPUT_DIR\" \\
    --dtype $DTYPE \\
    --bf16 $BF16 \\
    --report_to $REPORT_TO \\
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \\
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \\
    --beta $BETA \\
    --learning_rate $LEARNING_RATE \\
    --lr_scheduler_type $LR_SCHEDULER_TYPE \\
    --warmup_ratio $WARMUP_RATIO \\
    --use_vllm $USE_VLLM \\
    --logging_steps $LOGGING_STEPS \\
    $DEEPSPEED_ARGS \\
    --use_peft $USE_PEFT \\
    --qlora $QLORA \\
    --load_in_4bit $LOAD_IN_4BIT \\
    --lora_target_modules $LORA_TARGET_MODULES \\
    --lora_r $LORA_R \\
    --lora_alpha $LORA_ALPHA \\
    --lora_dropout $LORA_DROPOUT \\
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \\
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \\
    --num_generations $NUM_GENERATIONS \\
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
    --max_prompt_length $MAX_PROMPT_LENGTH \\
    --max_completion_length $MAX_COMPLETION_LENGTH"

echo "训练完成! 结果保存在: $OUTPUT_DIR"

# ======== 清理临时文件 ========
if [ "$ZERO_STAGE" != "0" ] && [ -f "$DEEPSPEED_CONFIG" ]; then
    echo "清理临时DeepSpeed配置文件: $DEEPSPEED_CONFIG"
    # 保留配置文件用于调试，如果需要删除请取消下面这行的注释
    # rm "$DEEPSPEED_CONFIG"
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
echo "4. 多节点训练 (主节点):"
echo "   bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 0 --master_addr 192.168.1.100 --zero_stage 3"
echo ""
echo "5. 多节点训练 (工作节点):"
echo "   bash $0 --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr 192.168.1.100 --zero_stage 3"
echo ""
echo "=== 注意事项 ==="
echo "1. ZeRO-2: 优化器状态分片，减少显存使用"
echo "2. ZeRO-3: 参数、梯度、优化器状态全部分片，最大显存节省"
echo "3. CPU Offload: 将参数/优化器转移到CPU内存，进一步节省显存"
echo "4. 使用ZeRO-3时，建议开启cpu_offload=true以获得最佳显存优化"
echo "5. DeepSpeed需要安装: pip install deepspeed"
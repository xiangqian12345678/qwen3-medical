# 多节点多GPU分布式训练指南

## 脚本说明

`torchrun_multi_node_multi_gpu.sh` 是基于 PyTorch torchrun 的多节点多GPU分布式训练脚本，支持：

- 单节点多GPU训练
- 多节点多GPU训练
- DeepSpeed 优化
- 混合精度训练 (BF16)

## 数据源配置

### 1.只采用本地数据

    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$WORLD_SIZE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        pretraining.py \
        --model_name_or_path $MODEL_PATH \
        --train_file_dir $TRAIN_FILE_DIR \
        ......

### 2. 采用本地数据和huggingface数据

    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$WORLD_SIZE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        pretraining.py \
        --model_name_or_path $MODEL_PATH \
        --dataset_name wikitext,Linly-AI/Chinese-pretraining-dataset \
        --dataset_config_name wikitext-2-raw-v1,none \
        --train_file_dir "$TRAIN_DATA_DIR" \
        ......

## 使用方法

### 1. 单节点多GPU训练

```bash
# 使用4个GPU训练
export NPROC_PER_NODE=4
bash torchrun_multi_node_multi_gpu.sh

# 或直接设置GPU数量
NPROC_PER_NODE=4 bash torchrun_multi_node_multi_gpu.sh
```

### 2. 多节点多GPU训练

假设有2个节点，每个节点4个GPU：

#### 2.1 主节点 (NODE_RANK=0)：

```bash
export MASTER_ADDR="192.168.1.100"  # 主节点IP
export MASTER_PORT="29500"
export NODE_RANK="0"
export WORLD_SIZE="2"
export NPROC_PER_NODE="4"

bash torchrun_multi_node_multi_gpu.sh
```

#### 2.2 从节点 (NODE_RANK=1)：

```bash
export MASTER_ADDR="192.168.1.100"  # 主节点IP
export MASTER_PORT="29500"
export NODE_RANK="1"
export WORLD_SIZE="2"
export NPROC_PER_NODE="4"

bash torchrun_multi_node_multi_gpu.sh
```

### 3. 环境变量说明

| 变量名              | 说明      | 默认值       |
|------------------|---------|-----------|
| `MASTER_ADDR`    | 主节点IP地址 | localhost |
| `MASTER_PORT`    | 主节点端口   | 29500     |
| `NODE_RANK`      | 当前节点排名  | 0         |
| `WORLD_SIZE`     | 总节点数    | 1         |
| `NPROC_PER_NODE` | 每节点GPU数 | 4         |

## 参数调整建议

### 批量大小配置

- `per_device_train_batch_size`: 建议设置为1-2（根据GPU内存调整）
- `gradient_accumulation_steps`: 总批量大小 = per_device_batch_size × num_gpus × gradient_accumulation_steps
- 例如：4个GPU，每GPU批量大小1，累积步数32，总批量大小为128

### DeepSpeed配置

- 使用ZeRO Stage 2进行内存优化
- 开启CPU Offload减少GPU内存占用
- 支持梯度检查点节省内存

## 注意事项

1. **网络配置**：多节点训练确保节点间网络连通，防火墙开放主节点端口
2. **环境一致性**：所有节点需要相同的Python环境、依赖版本和代码
3. **数据访问**：确保所有节点都能访问相同的训练数据路径
4. **GPU内存**：根据GPU内存调整批量大小和模型精度
5. **存储权限**：输出目录需要所有节点都有写入权限

## 故障排查

1. **连接超时**：检查网络连接和防火墙设置
2. **内存不足**：减少批量大小或启用更多DeepSpeed优化
3. **数据加载错误**：确认数据路径在所有节点上可访问
4. **版本不匹配**：确保所有节点的PyTorch和Transformers版本一致

## 性能优化

1. **数据预处理**：使用多进程数据加载 (`preprocessing_num_workers`)
2. **梯度累积**：适当调整梯度累积步数平衡内存和计算效率
3. **混合精度**：使用BF16减少内存使用并加速训练
4. **梯度检查点**：启用梯度检查点节省内存（会增加计算时间）
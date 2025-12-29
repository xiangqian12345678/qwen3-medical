# 微调分布式训练指南 (基于torchrun)

## 脚本说明

`torchrun_multi_node_multi_gpu.sh` 是基于 PyTorch torchrun 的多节点多GPU分布式微调训练脚本，支持：

- 单节点多GPU训练
- 多节点多GPU训练
- DeepSpeed 优化
- LoRA 微调
- 混合精度训练 (BF16)
- Flash Attention 加速

## 主要特性

### 1. 分布式训练支持
- 支持 DDP (DistributedDataParallel) 分布式训练
- 自动处理多节点通信
- 支持梯度累积和优化

### 2. 内存优化
- DeepSpeed ZeRO Stage 2 优化
- CPU Offload 减少 GPU 内存占用
- 梯度检查点节省内存
- BF16 混合精度训练

### 3. 模型微调
- LoRA 参数高效微调
- 支持自定义 LoRA 配置
- 自动保存微调适配器

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

#### 主节点 (NODE_RANK=0)：
```bash
export MASTER_ADDR="192.168.1.100"  # 主节点IP
export MASTER_PORT="29500"
export NODE_RANK="0"
export WORLD_SIZE="2"
export NPROC_PER_NODE="4"

bash torchrun_multi_node_multi_gpu.sh
```

#### 从节点 (NODE_RANK=1)：
```bash
export MASTER_ADDR="192.168.1.100"  # 主节点IP
export MASTER_PORT="29500"
export NODE_RANK="1"
export WORLD_SIZE="2"
export NPROC_PER_NODE="4"

bash torchrun_multi_node_multi_gpu.sh
```

### 3. 环境变量说明

| 变量名              | 说明          | 默认值       |
|------------------|-------------|-----------|
| `MASTER_ADDR`    | 主节点IP地址    | localhost |
| `MASTER_PORT`    | 主节点端口      | 29500     |
| `NODE_RANK`      | 当前节点排名     | 0         |
| `WORLD_SIZE`     | 总节点数       | 1         |
| `NPROC_PER_NODE` | 每节点GPU数    | 4         |

## 参数配置

### 1. 模型配置
- `model_name_or_path`: 基础模型路径
- `tokenizer_name_or_path`: 分词器路径
- `model_max_length`: 最大序列长度 (2048)

### 2. LoRA配置
- `lora_rank`: LoRA 等级 (8)
- `lora_alpha`: LoRA alpha 值 (16)
- `lora_dropout`: LoRA dropout (0.05)
- `target_modules`: 目标模块 (all)

### 3. 训练配置
- `learning_rate`: 学习率 (2e-5)
- `num_train_epochs`: 训练轮数 (1)
- `per_device_train_batch_size`: 每设备批量大小 (1)
- `gradient_accumulation_steps`: 梯度累积步数 (8)

### 4. 优化配置
- `bf16`: 启用 BF16 精度
- `flash_attn`: 启用 Flash Attention
- `gradient_checkpointing`: 启用梯度检查点
- `deepspeed`: DeepSpeed 配置文件

## 性能优化建议

### 1. 批量大小调整
```bash
# 增加批量大小提高训练效率
--per_device_train_batch_size 2
--gradient_accumulation_steps 4  # 总批量大小 = 2 * 4 * GPU数
```

### 2. 数据加载优化
```bash
# 增加数据加载进程数
--preprocessing_num_workers 8
--dataloader_num_workers 4
```

### 3. 内存优化
```bash
# 如果内存不足，可以：
# 1. 减少 LoRA rank
--lora_rank 4
# 2. 启用更多 DeepSpeed 优化
# 3. 减少批量大小
```

## 故障排查

### 1. 连接问题
```bash
# 检查网络连通性
ping $MASTER_ADDR
# 检查端口是否开放
telnet $MASTER_ADDR $MASTER_PORT
```

### 2. 内存问题
```bash
# 监控GPU使用情况
nvidia-smi -l 1
# 如果内存不足，减少批量大小或启用更多优化
```

### 3. 数据问题
```bash
# 确保所有节点都能访问数据目录
ls -la $TRAIN_FILE_DIR
# 检查数据格式是否正确
```

## 扩展功能

### 1. 自定义DeepSpeed配置
可以修改 `ds_config.json` 文件来调整DeepSpeed优化策略：

```json
{
    "zero_optimization": {
        "stage": 3,  // 使用 ZeRO Stage 3 进一步节省内存
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

### 2. 模型量化
```bash
# 添加量化参数
--load_in_8bit True
--quantization_type 8bit
```

### 3. 多任务微调
```bash
# 指定多个训练数据集
--train_file_dir "$TRAIN_DIR1,$TRAIN_DIR2"
--validation_file_dir "$VAL_DIR1,$VAL_DIR2"
```

## 注意事项

1. **环境一致性**: 确保所有节点的Python环境、依赖版本完全一致
2. **数据同步**: 确保所有节点都能访问相同的训练数据路径
3. **输出权限**: 输出目录需要所有节点都有写入权限
4. **防火墙设置**: 确保节点间网络端口开放
5. **GPU驱动**: 确保所有节点的NVIDIA驱动版本兼容

## 与单GPU脚本对比

| 特性              | 单GPU脚本 | torchrun脚本 |
|------------------|---------|-------------|
| GPU利用率         | 1 GPU   | 多个GPU      |
| 训练速度          | 较慢     | 显著提升      |
| 内存优化          | 基础     | DeepSpeed优化 |
| 分布式支持        | 无       | 完整支持      |
| 配置复杂度        | 简单     | 中等         |

通过使用torchrun分布式训练脚本，可以充分利用多GPU资源，显著提升微调训练效率。
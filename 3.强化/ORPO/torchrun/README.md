# ORPO 多节点多 GPU 分布式训练文档

基于 torchrun + DeepSpeed，多节点/多GPU大模型训练，支持 ZeRO-1/2/3。

---

## 1. 基本用法

```bash
bash torchrun_multi_node_multi_gpu.sh \
    [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
```

参数说明：

| 参数            | 含义                  |
| ------------- | ------------------- |
| zero_stage    | ZeRO 优化阶段，1 / 2 / 3 |
| num_nodes     | 节点数量                |
| gpus_per_node | 每节点 GPU 数量          |
| master_addr   | 主节点 IP 地址           |
| master_port   | 主节点端口号              |

---

## 2. 使用示例

### 单节点多GPU（4卡，ZeRO-3）

```bash
bash torchrun_multi_node_multi_gpu.sh 3 1 4 localhost 29500
```

### 多节点多GPU（2节点 × 4 GPU）

**节点0（主节点）**

```bash
NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29500
```

**节点1（工作节点）**

```bash
NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29500
```

> 注意：所有节点 MASTER_ADDR 与 MASTER_PORT 必须一致，NODE_RANK 唯一。

---

## 3. ZeRO 优化阶段说明

| ZeRO 阶段 | 特点                                  |
| ------- | ----------------------------------- |
| 1       | Optimizer 分片，显存优化有限，通信开销低           |
| 2       | Optimizer + Grad 分片，显存/通信综合最优       |
| 3       | Optimizer + Grad + 参数分片，显存最低，通信开销最大 |

配置文件对应关系：

* ZeRO-1 → zero1.json
* ZeRO-2 → zero2.json
* ZeRO-3 → zero3.json

脚本直接加载本地 JSON 配置文件，无自动生成。

---

## 4. 模型与数据路径说明

| 参数             | 默认路径                               |
| -------------- | ---------------------------------- |
| MODEL_PATH     | `../model/Qwen/Qwen3-0.6B`         |
| TOKENIZER_PATH | `../output/tokenizers_merge`       |
| TRAIN_FILE_DIR | `../data/reward`                   |
| OUTPUT_DIR     | `../output/orpo_zero${ZERO_STAGE}` |
| CACHE_DIR      | `../output/cache`                  |

---

## 5. 训练超参数（默认值）

| 参数                          | 默认值  |
| --------------------------- | ---- |
| per_device_train_batch_size | 1    |
| per_device_eval_batch_size  | 1    |
| gradient_accumulation_steps | 4    |
| effective batch size        | 自动计算 |
| max_train_samples           | 1000 |
| max_eval_samples            | 50   |
| max_steps                   | 100  |
| eval_steps                  | 20   |
| save_steps                  | 50   |
| lora_rank                   | 8    |
| lora_alpha                  | 16   |
| lora_dropout                | 0.05 |
| bf16                        | True |
| gradient_checkpointing      | True |

---

## 6. 注意事项

* 多节点训练时，MASTER_ADDR / MASTER_PORT 必须一致。
* 每个节点 NODE_RANK 唯一（从0开始）。
* ZeRO-3 显存占用最低，但通信压力最大。
* BF16 需要 GPU 硬件支持。
* 输出目录存在时会覆盖已有内容。

---

## 7. 工程实践建议

| 模型规模  | 建议                               |
| ----- | -------------------------------- |
| ≤ 13B | ZeRO-2 + Activation Checkpoint   |
| ≈ 32B | ZeRO-2 + CPU Offload（推荐）         |
| ≥ 65B | ZeRO-3 + CPU Offload + 通信 Bucket |

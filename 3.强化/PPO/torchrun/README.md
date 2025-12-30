# PPO 分布式训练脚本与使用说明

## 1️⃣ 模式支持

PPO不支持zero3模式

## 2️⃣ 使用文档

### 基本用法

```bash
bash torchrun_multi_node_multi_gpu.sh \
    [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
```

* `zero_stage`         ZeRO 优化阶段，可选 1 / 2 / 3
* `num_nodes`          训练节点数量
* `gpus_per_node`      每个节点 GPU 数量
* `master_addr`        主节点 IP 地址
* `master_port`        主节点端口号

### 使用示例

#### 单节点多 GPU（4 卡，ZeRO-2）

```bash
bash torchrun_multi_node_multi_gpu.sh 2 1 4 localhost 29502
```

#### 多节点多 GPU（2 节点 × 4 GPU）

节点 0（主节点）：

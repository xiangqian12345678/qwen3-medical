# GRPO 多节点多 GPU 分布式训练使用说明

本文档说明如何使用脚本：
`torchrun_multi_node_multi_gpu.sh`

支持单节点 / 多节点训练，兼容 DeepSpeed ZeRO-2 / ZeRO-3 优化。

---

## 1. 命令格式

```bash
bash torchrun_multi_node_multi_gpu.sh \
    [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
```

参数说明：

| 参数            | 含义                     | 默认值       |
| ------------- | ---------------------- | --------- |
| zero_stage    | ZeRO 优化阶段: 0(关闭), 2, 3 | 0         |
| num_nodes     | 训练节点数量                 | 1         |
| gpus_per_node | 每节点 GPU 数量             | 1         |
| master_addr   | 主节点地址                  | localhost |
| master_port   | 主节点端口                  | 29501     |

---

## 2. 使用示例

### 单节点多 GPU（4 卡，ZeRO-2）

```bash
bash torchrun_multi_node_multi_gpu.sh 2 1 4 localhost 29501
```

### 多节点多 GPU（2 节点 × 4 GPU，ZeRO-3 + CPU Offload）

节点 0（主节点）：

```bash
NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29501
```

节点 1（工作节点）：

```bash
NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29501
```

注意：

* 所有节点 MASTER_ADDR 与 MASTER_PORT 必须一致
* 每节点 NODE_RANK 唯一（从 0 开始）

---

## 3. ZeRO 优化阶段说明

* **ZeRO-2**：优化器状态 + 梯度分片，显存/通信均衡，常用
* **ZeRO-3**：优化器状态 + 梯度 + 模型参数分片，显存最低，通信开销大
* **CPU Offload**：仅 ZeRO-3 支持，将参数/优化器迁移到 CPU 内存，进一步节省显存

---

## 4. 注意事项

1. 多节点训练时，各节点 MASTER_ADDR / MASTER_PORT 必须相同
2. ZeRO-3 建议开启 CPU Offload 以获得最佳显存优化
3. DeepSpeed 配置文件自动生成在当前目录
4. 输出目录存在时会被覆盖

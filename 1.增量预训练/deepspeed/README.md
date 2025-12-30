# DeepSpeed 多节点多GPU分布式训练指南

## 1. 基本用法

使用 Bash 脚本启动 DeepSpeed 分布式训练：

```bash
bash deepspeed_muti_node_muti_gpu.sh [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]
```

- zero_stage：ZeRO 优化阶段（1, 2, 3）
- num_nodes：训练节点数量
- gpus_per_node：每个节点的 GPU 数量
- master_addr：主节点 IP 地址
- master_port：主节点端口号

示例：

单节点多 GPU：

```bash
bash deepspeed_muti_node_muti_gpu.sh 3 1 4 localhost 29501
```

多节点多 GPU（2 节点，每节点 4 GPU）：

```bash
# 节点0（主节点）
NODE_RANK=0 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501
```

```bash
# 节点1
NODE_RANK=1 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501
```

## 2. 参数说明

### 2.1 ZeRO 阶段

    | 阶段 | 描述 |
    | --- | --------------------------------------- |
    | 1 | 仅优化器状态分片 |
    | 2 | 优化器状态 + 梯度分片 |
    | 3 | 优化器状态 + 梯度 + 模型参数分片（最节省显存） |

### 2.2 模型与数据路径

    MODEL_PATH：预训练模型路径（如 Qwen3-0.6B）
    TRAIN_DATA_DIR：训练数据目录
    VALID_DATA_DIR：验证数据目录
    OUTPUT_DIR：训练结果输出目录

### 2.3 训练超参数

    | 参数                           | 默认值   | 描述                    |
    | ----------------------------- | ----- | --------------------- |
    | `PER_DEVICE_TRAIN_BATCH_SIZE` | 1     | 每个 GPU 的训练 batch size |
    | `PER_DEVICE_EVAL_BATCH_SIZE`  | 1     | 每个 GPU 的评估 batch size |
    | `GRADIENT_ACCUMULATION_STEPS` | 16    | 梯度累积步数                |
    | `NUM_TRAIN_EPOCHS`            | 0.5   | 训练轮数                  |
    | `LEARNING_RATE`               | 2e-4  | 学习率                   |
    | `WEIGHT_DECAY`                | 0.01  | 权重衰减                  |
    | `WARMUP_RATIO`                | 0.05  | 学习率 warmup 比例         |
    | `MAX_TRAIN_SAMPLES`           | 10000 | 最大训练样本数               |
    | `MAX_EVAL_SAMPLES`            | 10    | 最大评估样本数               |

### 2.4 DeepSpeed 配置文件

    根据 ZeRO 阶段选择配置文件
    | 阶段 | 配置文件   |
    | -- | ---------- |
    | 1  | zero1.json |
    | 2  | zero2.json |
    | 3  | zero3.json |

## 3.三种模式分析

    32B模型进行训练，8卡，精度：FP16 / BF16（2 Bytes）
    1.模型显存分析
        | 项目         | 大小             |
        | ----------- | ---------------- |
        | Parameters  | 64 GB            |
        | Gradients   | 64 GB            |
        | Optimizer   | 128 GB           |
        | **合计**     | **256 GB / GPU** |

    2.ZeRO-1模式 只切 Optimizer
        | 项目         | 公式      | 每卡              |
        | ----------- | --------- | ---------------- |
        | Parameters  | 64        | 64 GB            |
        | Gradients   | 64        | 64 GB            |
        | Optimizer   | 128 / 8   | 16 GB            |
        | **合计**     |           | **144 GB / GPU** |

    3.ZeRO-2：切 Optimizer + Gradients 
        | 项目        | 公式     | 每卡             |
        | ---------- | ------- | --------------- |
        | Parameters | 64      | 64 GB           |
        | Gradients  | 64 / 8  | 8 GB            |
        | Optimizer  | 128 / 8 | 16 GB           |
        | **合计**    |         | **88 GB / GPU** |
        
        优化 1：Activation Checkpoint
            activation 通常 = 参数的 20~40%
            32B 下：≈ 12~25GB
            checkpoint 后可降到 3~8GB

        优化 2：CPU Offload Optimizer：
            optimizer → CPU memory
            | 项目        | GPU             |
            | ---------- | --------------- |
            | Parameters | 64              |
            | Gradients  | 8               |
            | Optimizer  | 0               |
            | **合计**    | **72 GB / GPU** |

        优化 3：BF16 + reduce bucket
            控制 reduce_bucket_size
            避免通信峰值显存
        工程结论： 
            ZeRO-2 + activation checkpoint + optimizer offload
            是 32B 在 8 卡下最稳妥、性价比最高方案

    4.ZeRO-3：切Optimizer + Gradients + Parameters
        | 项目        | 公式     | 每卡            |
        | ---------- | ------- | --------------- |
        | Parameters | 64 / 8  | 8 GB            |
        | Gradients  | 64 / 8  | 8 GB            |
        | Optimizer  | 128 / 8 | 16 GB           |
        | **合计**    |         | **32 GB / GPU** |
    5.ZeRO-3: 切Optimizer + Gradients + Parameters  + CPU offload
        梯度（一般不 offload）: 64 GB / 8 = 8 GB
        | 项目        | 公式     | 每卡            |
        | ---------- | ------- | --------------- |
        | Parameters | 64 / 8  | 8 GB            |
        | Optimizer  | 128 / 8 | 16 GB           |
        | Optimizer  | 0       |
        | **合计**    |         | **24 GB / GPU** |

## 4. 训练流程说明

    - 脚本根据输入参数设置 ZeRO 阶段和 DeepSpeed 配置
    - 创建输出目录，并设置 CUDA_VISIBLE_DEVICES
    - 计算总进程数并打印训练配置信息
    - 执行 DeepSpeed 命令启动训练：
        包含训练和评估 (--do_train --do_eval)
        支持梯度累积、混合精度（bfloat16）
        日志记录和模型保存按步数触发
        使用分布式训练环境变量 NODE_RANK 和 WORLD_SIZE

## 5. 注意事项

    - 确保 DeepSpeed 配置文件存在于脚本目录
    - 多节点训练需保证各节点 MASTER_ADDR 和 MASTER_PORT 一致
    - ZeRO-3 可以显著节省显存，但可能增加通信开销
    - 脚本默认采用 bfloat16 混合精度训练，需 GPU 支持
    - 输出目录可使用 --overwrite_output_dir 覆盖已有结果

============================================================
torchrun + DeepSpeed 多节点多 GPU DPO 训练使用说明
============================================================

本文档说明如何使用脚本：

    torchrun_multi_node_multi_gpu.sh

在单节点或多节点环境下，基于 torchrun + DeepSpeed
进行 DPO（Direct Preference Optimization）分布式训练，
支持 ZeRO-1 / ZeRO-2 / ZeRO-3，并结合 LoRA 微调。

------------------------------------------------------------
1. 基本用法
------------------------------------------------------------

启动命令格式：

    bash torchrun_multi_node_multi_gpu.sh \
        [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]

参数说明：

    zero_stage     ZeRO 优化阶段，可选 1 / 2 / 3
    num_nodes      训练节点数量
    gpus_per_node  每个节点的 GPU 数量
    master_addr    主节点 IP 地址
    master_port    主节点端口号

NODE_RANK 通过环境变量指定。

------------------------------------------------------------
2. 使用示例
------------------------------------------------------------

【示例 1】单节点单卡（ZeRO-2，推荐）

    bash torchrun_multi_node_multi_gpu.sh 2 1 1 localhost 29500


【示例 2】单节点多 GPU（4 卡）

    bash torchrun_multi_node_multi_gpu.sh 2 1 4 localhost 29500


【示例 3】多节点多 GPU（2 节点 × 4 GPU）

节点 0（主节点）：

    NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh \
        2 2 4 192.168.1.100 29500

节点 1：

    NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh \
        2 2 4 192.168.1.100 29500

------------------------------------------------------------
3. ZeRO 优化阶段说明
------------------------------------------------------------

ZeRO-1：
    - 仅切分 Optimizer 状态
    - 通信最少，显存优化有限

ZeRO-2（推荐）：
    - 切分 Optimizer + Gradients
    - 显存与通信的最佳平衡
    - DPO / SFT 最常用方案

ZeRO-3：
    - 切分 Optimizer + Gradients + Parameters
    - 显存占用最低
    - 通信压力最大

对应配置文件：

    ZeRO-1 -> zero1.json
    ZeRO-2 -> zero2.json
    ZeRO-3 -> zero3.json

------------------------------------------------------------
4. 模型与数据路径
------------------------------------------------------------

脚本中使用的主要路径：

    MODEL_PATH        基座模型路径（如 Qwen3-0.6B）
    TOKENIZER_PATH    tokenizer 路径
    TRAIN_FILE_DIR    DPO 偏好数据目录
    VALID_FILE_DIR    验证数据目录
    OUTPUT_DIR        LoRA Adapter 输出目录
    CACHE_DIR         HuggingFace / 数据缓存目录

------------------------------------------------------------
5. 训练特性
------------------------------------------------------------

    - torchrun 原生分布式启动
    - DeepSpeed ZeRO 优化
    - BF16 混合精度
    - LoRA 参数高效微调
    - Gradient Accumulation
    - Gradient Checkpointing
    - 多节点多 GPU 同步训练

------------------------------------------------------------
6. 工程建议
------------------------------------------------------------

    模型规模 ≤ 13B：
        ZeRO-2 + LoRA

    模型规模 ≈ 32B：
        ZeRO-2 + Gradient Checkpoint（推荐）

    模型规模 ≥ 65B：
        ZeRO-3 + CPU Offload（需通信调优）

------------------------------------------------------------
7. 注意事项
------------------------------------------------------------

    - 多节点训练时，MASTER_ADDR / MASTER_PORT 必须一致
    - NODE_RANK 必须从 0 连续递增
    - 确保 zero*.json 配置文件存在
    - OUTPUT_DIR 会被 overwrite 覆盖
    - BF16 需要硬件支持

============================================================

============================================================
torchrun + DeepSpeed 多节点多 GPU 分布式训练使用说明
============================================================

本文档说明如何使用脚本：
    torchrun_multi_node_multi_gpu.sh

在单节点或多节点环境下，基于 torchrun + DeepSpeed
进行大模型分布式训练，支持 ZeRO-1 / ZeRO-2 / ZeRO-3。

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


------------------------------------------------------------
2. 使用示例
------------------------------------------------------------

【示例 1】单节点多 GPU（4 卡，ZeRO-3）

    bash torchrun_multi_node_multi_gpu.sh 3 1 4 localhost 29500

含义说明：
    - 使用 ZeRO-3
    - 单节点
    - 每节点 4 张 GPU


【示例 2】多节点多 GPU（2 节点 × 4 GPU）

节点 0（主节点）：

    NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh \
        3 2 4 192.168.1.100 29500

节点 1：

    NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh \
        3 2 4 192.168.1.100 29500

注意事项：
    - 所有节点的 MASTER_ADDR 和 MASTER_PORT 必须保持一致
    - 每个节点的 NODE_RANK 必须唯一（从 0 开始）


------------------------------------------------------------
3. ZeRO 优化阶段说明
------------------------------------------------------------

ZeRO-1：
    - 仅对 Optimizer 状态进行分片
    - 显存优化有限，通信开销最低

ZeRO-2：
    - 对 Optimizer 状态 + 梯度进行分片
    - 显存 / 通信综合性价比最好
    - 工程中最常用

ZeRO-3：
    - 对 Optimizer + 梯度 + 模型参数进行分片
    - 显存占用最低
    - 通信开销最大


ZeRO 阶段与配置文件对应关系：

    ZeRO-1  ->  zero1.json
    ZeRO-2  ->  zero2.json
    ZeRO-3  ->  zero3.json

脚本会根据 zero_stage 参数自动选择配置文件。


------------------------------------------------------------
4. 模型与数据路径说明
------------------------------------------------------------

脚本中使用的主要路径参数：

    MODEL_PATH        预训练模型路径（如 Qwen3-0.6B）
    TOKENIZER_PATH    tokenizer 路径
    TRAIN_FILE_DIR    训练数据目录
    VALID_FILE_DIR    验证数据目录
    OUTPUT_DIR        模型与日志输出目录
    CACHE_DIR         HuggingFace / 数据缓存目录


------------------------------------------------------------
5. 训练超参数（默认值）
------------------------------------------------------------

    per_device_train_batch_size    = 1
    per_device_eval_batch_size     = 1
    gradient_accumulation_steps    = 16
    num_train_epochs               = 0.5
    learning_rate                  = 2e-4
    warmup_ratio                   = 0.05
    weight_decay                   = 0.01
    max_train_samples              = 10000
    max_eval_samples               = 10


------------------------------------------------------------
6. 显存分析（32B 模型 / 8 卡）
------------------------------------------------------------

假设条件：
    - 模型规模：32B
    - 精度：FP16 / BF16（2 Bytes）


【无 ZeRO 情况】

    Parameters        64  GB
    Gradients         64  GB
    Optimizer States  128 GB
    --------------------------------
    合计              256 GB / GPU


【ZeRO-1（仅切 Optimizer）】

    Parameters        64  GB
    Gradients         64  GB
    Optimizer         16  GB
    --------------------------------
    合计              144 GB / GPU


【ZeRO-2（切 Optimizer + Gradients）】

    Parameters        64  GB
    Gradients          8  GB
    Optimizer         16  GB
    --------------------------------
    合计               88 GB / GPU


ZeRO-2 推荐优化组合：

    1. Activation Checkpoint
        - Activation ≈ 参数的 20% ~ 40%
        - 32B 模型约 12 ~ 25 GB
        - Checkpoint 后可降至 3 ~ 8 GB

    2. Optimizer CPU Offload
        - Optimizer 状态迁移到 CPU 内存
        - GPU 显存可降至约 72 GB / GPU

工程结论：
    ZeRO-2 + Activation Checkpoint + Optimizer Offload
    是 32B / 8 卡下最稳妥、性价比最高方案


【ZeRO-3（切 Optimizer + Gradients + Parameters）】

    Parameters         8  GB
    Gradients          8  GB
    Optimizer         16  GB
    --------------------------------
    合计              32 GB / GPU


【ZeRO-3 + CPU Offload】

    Parameters         8  GB
    Gradients          8  GB
    Optimizer          0  GB
    --------------------------------
    合计             ≈24 GB / GPU


------------------------------------------------------------
7. 训练流程说明
------------------------------------------------------------

整体流程如下：

    1. 解析输入参数，确定 ZeRO 阶段与分布式规模
    2. 自动选择 DeepSpeed 配置文件
    3. 设置 torchrun 分布式环境变量：
        - MASTER_ADDR
        - MASTER_PORT
        - NODE_RANK
        - WORLD_SIZE
    4. 使用 torchrun 启动 pretraining.py
    5. 训练过程中支持：
        - BF16 混合精度
        - 梯度累积
        - 梯度检查点
        - TensorBoard 日志
        - 多节点多 GPU 同步训练


------------------------------------------------------------
8. 注意事项
------------------------------------------------------------

    - 多节点训练时，所有节点的 MASTER_ADDR 和 MASTER_PORT 必须一致
    - 每个节点的 NODE_RANK 不能重复
    - ZeRO-3 显存占用最低，但通信压力最大
    - BF16 需要 GPU 硬件支持
    - 请确保 zero1.json / zero2.json / zero3.json 配置文件存在
    - 输出目录存在时会被 overwrite_output_dir 覆盖


------------------------------------------------------------
9. 工程实践建议
------------------------------------------------------------

    模型规模 ≤ 13B：
        ZeRO-2 + Activation Checkpoint

    模型规模 ≈ 32B：
        ZeRO-2 + Optimizer CPU Offload（推荐）

    模型规模 ≥ 65B：
        ZeRO-3 + CPU Offload + 通信 Bucket 调优


------------------------------------------------------------
10. 说明
------------------------------------------------------------

本脚本基于 torchrun + DeepSpeed：

    - 更贴近 PyTorch 原生分布式训练
    - 便于接入任务调度系统 / 平台化训练
    - 相比 deepspeed launcher 更灵活、更工程化

============================================================

============================================================
RM 分布式训练 - torchrun + DeepSpeed 使用说明
============================================================

本文档说明如何使用脚本：torchrun_multi_node_multi_gpu.sh
支持 ZeRO-1 / ZeRO-2 / ZeRO-3 训练模式，适用于单节点或多节点多 GPU 环境。

------------------------------------------------------------
1. 基本用法
------------------------------------------------------------

启动命令格式：

    bash torchrun_multi_node_multi_gpu.sh \
        [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]

参数说明：

    zero_stage       ZeRO 阶段，可选 1 / 2 / 3
    num_nodes        训练节点数量
    gpus_per_node    每个节点 GPU 数量
    master_addr      主节点 IP 地址
    master_port      主节点端口号

------------------------------------------------------------
2. 使用示例
------------------------------------------------------------

【示例 1】单节点多 GPU（4 卡，ZeRO-3）

    bash torchrun_multi_node_multi_gpu.sh 3 1 4 localhost 29500

说明：
    - 使用 ZeRO-3
    - 单节点
    - 每节点 4 张 GPU

【示例 2】多节点多 GPU（2 节点 × 4 GPU）

节点 0（主节点）：

    NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29500

节点 1：

    NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh 3 2 4 192.168.1.100 29500

注意：
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
    - 显存 / 通信综合性价比最佳
    - 工程中最常用

ZeRO-3：
    - 对 Optimizer + 梯度 + 模型参数进行分片
    - 显存占用最低
    - 通信开销最大

ZeRO 阶段对应配置文件：

    ZeRO-1 -> zero1.json
    ZeRO-2 -> zero2.json
    ZeRO-3 -> zero3.json

------------------------------------------------------------
4. 模型与数据路径
------------------------------------------------------------

脚本中主要路径：

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
    gradient_accumulation_steps    = 8
    num_train_epochs               = 1
    learning_rate                  = 2e-5
    weight_decay                   = 0.001
    warmup_ratio                    = 0.05
    max_train_samples              = 1000
    max_eval_samples               = 10
    lora_rank                       = 8
    lora_alpha                      = 16
    lora_dropout                     = 0.05
    max_source_length              = 1024
    max_target_length              = 256

------------------------------------------------------------
6. 注意事项
------------------------------------------------------------

    - 确保 zero1.json / zero2.json / zero3.json 配置文件存在
    - 输出目录存在时会被 overwrite_output_dir 覆盖
    - ZeRO-3 显存占用最低，但通信压力最大
    - BF16 / bfloat16 需要 GPU 支持
    - 多节点训练时，各节点 NODE_RANK 唯一，MASTER_ADDR / MASTER_PORT 保持一致

------------------------------------------------------------
7. 工程实践建议
------------------------------------------------------------

    模型 ≤ 13B：
        ZeRO-2 + Activation Checkpoint

    模型 ≈ 32B：
        ZeRO-2 + Optimizer CPU Offload（推荐）

    模型 ≥ 65B：
        ZeRO-3 + CPU Offload + 通信 Bucket 调优

============================================================

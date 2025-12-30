============================================================
torchrun + DeepSpeed 多节点多 GPU SFT 微调使用说明
============================================================

本文档说明如何使用脚本：

    torchrun_multi_node_multi_gpu.sh

该脚本基于 torchrun + DeepSpeed，
用于大模型 SFT（监督微调），支持 LoRA + ZeRO-1 / 2 / 3。

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

当前节点 rank 通过环境变量 NODE_RANK 指定。

------------------------------------------------------------
2. 使用示例
------------------------------------------------------------

【示例 1】单节点多 GPU（4 卡，ZeRO-2）

    bash torchrun_multi_node_multi_gpu.sh 2 1 4 localhost 29500

【示例 2】多节点多 GPU（2 节点 × 4 GPU，ZeRO-3）

节点 0（主节点）：

    NODE_RANK=0 bash torchrun_multi_node_multi_gpu.sh \
        3 2 4 192.168.1.100 29500

节点 1：

    NODE_RANK=1 bash torchrun_multi_node_multi_gpu.sh \
        3 2 4 192.168.1.100 29500

注意事项：
    - 所有节点的 master_addr / master_port 必须一致
    - 每个节点的 NODE_RANK 必须唯一（从 0 开始）

------------------------------------------------------------
3. ZeRO 优化阶段说明
------------------------------------------------------------

ZeRO-1：
    - 仅切分 Optimizer 状态
    - 通信开销最小，显存节省有限

ZeRO-2：
    - 切分 Optimizer + Gradient
    - 显存 / 通信综合最优
    - SFT / LoRA 场景最常用（推荐）

ZeRO-3：
    - 切分 Optimizer + Gradient + Parameters
    - 显存占用最低
    - 通信开销最大

对应配置文件：

    ZeRO-1  -> zero1.json
    ZeRO-2  -> zero2.json
    ZeRO-3  -> zero3.json

脚本会根据 zero_stage 自动选择。

------------------------------------------------------------
4. 模型与数据路径说明
------------------------------------------------------------

主要路径参数：

    MODEL_PATH        基座模型路径（如 Qwen3-0.6B）
    TOKENIZER_PATH    tokenizer 路径
    TRAIN_FILE_DIR    SFT 训练数据目录
    VALID_FILE_DIR    SFT 验证数据目录
    OUTPUT_DIR        LoRA 权重输出目录
    CACHE_DIR         HuggingFace / 数据缓存目录

------------------------------------------------------------
5. LoRA 微调参数
------------------------------------------------------------

    use_peft       = True
    target_modules = all
    lora_rank      = 8
    lora_alpha     = 16
    lora_dropout   = 0.05

说明：
    - 仅训练 LoRA Adapter 权重
    - 基座模型参数保持冻结
    - 显存占用远低于全参微调

------------------------------------------------------------
6. 训练超参数（默认值）
------------------------------------------------------------

    per_device_train_batch_size    = 1
    per_device_eval_batch_size     = 1
    gradient_accumulation_steps    = 8
    num_train_epochs               = 1
    learning_rate                  = 2e-5
    warmup_ratio                   = 0.05
    weight_decay                   = 0.05
    model_max_length               = 2048
    max_train_samples              = 1000
    max_eval_samples               = 10

------------------------------------------------------------
7. 训练流程说明
------------------------------------------------------------

整体流程如下：

    1. 解析命令行参数（ZeRO 阶段 / 节点数 / GPU 数）
    2. 自动选择 DeepSpeed 配置文件
    3. 设置 torchrun 分布式环境变量
    4. 启动 sft.py
    5. 开启：
        - BF16 混合精度
        - 梯度累积
        - 梯度检查点
        - Flash Attention
        - TensorBoard 日志
        - DeepSpeed 分布式优化

------------------------------------------------------------
8. 注意事项
------------------------------------------------------------

    - 多节点训练时，网络必须互通
    - BF16 需要 GPU 硬件支持
    - zero1.json / zero2.json / zero3.json 必须存在
    - OUTPUT_DIR 存在时会被 overwrite_output_dir 覆盖
    - ZeRO-3 通信压力较大，需注意网络带宽

------------------------------------------------------------
9. 工程实践建议
------------------------------------------------------------

    SFT / LoRA 场景：
        ZeRO-2 + Gradient Checkpoint（推荐）

    显存极度受限：
        ZeRO-3 + LoRA

    平台化 / 调度系统：
        torchrun + DeepSpeed 作为统一入口

------------------------------------------------------------
10. 说明
------------------------------------------------------------

该脚本与预训练脚本在参数形式与调用方式上保持一致：

    bash torchrun_multi_node_multi_gpu.sh \
        [zero_stage] [num_nodes] [gpus_per_node] [master_addr] [master_port]

便于工程统一、文档统一、平台化接入。

============================================================

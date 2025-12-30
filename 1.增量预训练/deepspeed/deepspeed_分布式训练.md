# 基本用法

    # ZeRO-3 单节点  4GPU
        bash deepspeed_muti_node_muti_gpu.sh 3 1 4 localhost 29501
    
    # ZeRO-2  单节点  4GPU
        bash deepspeed_muti_node_muti_gpu.sh 2 1 4 localhost 29501

# 多节点训练示例:

    # 节点0 (主节点):
        NODE_RANK=0 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501
    
    # 节点1:
        NODE_RANK=1 bash deepspeed_muti_node_muti_gpu.sh 3 2 4 192.168.1.100 29501

# 数据源配置 deepspeed_muti_node_muti_gpu.sh
## 1.只采用本地数据
    deepspeed \
        --num_nodes=$NUM_NODES \
        --num_gpus=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank=$NODE_RANK \
        --module pretraining \
        --model_name_or_path "$MODEL_PATH" \
        --train_file_dir "$TRAIN_DATA_DIR" \
        ...... 
## 2. 采用本地数据和huggingface数据
    deepspeed \
        --num_nodes=$NUM_NODES \
        --num_gpus=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank=$NODE_RANK \
        --module pretraining \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_name wikitext,Linly-AI/Chinese-pretraining-dataset \
        --dataset_config_name wikitext-2-raw-v1,none \
        --train_file_dir "$TRAIN_DATA_DIR" \
        ...... 

# 三种模式分析

    32B模型进行训练，8卡
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

    5.对比
        | 模式              | 每卡显存  | 是否可跑 | 工程评价       |
        | ---------------- | -------- | ------ | ------------- |
        | ZeRO-1           | 144 GB   | ❌     | 直接淘汰       |
        | ZeRO-2           | 88 GB    | ⚠️     | **主流解法**   |
        | ZeRO-2 + offload | ~72 GB   | ✅     | ⭐⭐⭐⭐     |
        | ZeRO-3           | 40~55 GB | ✅     | ⭐⭐（吃网络） |



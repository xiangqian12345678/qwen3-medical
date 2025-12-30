# torchrun分布式训练

    torchrun训练脚本 中的文件拷贝到pretraining.py目录下
    按照torchrun_分布式训练.md进行执行，支持：
    1.单节点单GPU
    2.单节点多GPU
    3.多节点多GPU

# deepspeed分布式训练

    deepspeed训练脚本 中的文件拷贝到pretraining.py目录下
    按照torchrun_分布式训练.md进行执行，支持：
    分布式训练：
        1.单节点单GPU
        2.单节点多GPU
        3.多节点多GPU
    训练模式： zero1模式没有必要支持
        1.zero2模式
        2.zero3模式
        3.zero3+cpu_offload模式

# 训练脚本配置

    1.配置本地训练数据
        --train_file_dir ../data/pretrain \
        --validation_file_dir ../data/pretrain \
    2.配置开源数据集
        --dataset_name wikitext,Linly-AI/Chinese-pretraining-dataset \
        --dataset_config_name wikitext-2-raw-v1,none \
    3.注意数据集合的格式要满足要求，待规范补充
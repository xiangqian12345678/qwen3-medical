# 训练脚本配置

    1.配置本地训练数据
        --train_file_dir ../data/pretrain \
        --validation_file_dir ../data/pretrain \
    2.配置开源数据集
        --dataset_name wikitext,Linly-AI/Chinese-pretraining-dataset \
        --dataset_config_name wikitext-2-raw-v1,none \
    3.注意数据集合的格式要满足要求，待规范补充
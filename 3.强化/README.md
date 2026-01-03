## 训练命令

### 单机训练

```bash 
  sh run_rm.sh 
```

### 分布式训练

[分布式训练文档](./torchrun/README.md)

# 训练脚本配置

    1.配置本地训练数据
        --train_file_dir ../data/pretrain \
        --validation_file_dir ../data/pretrain \
    2.配置开源数据集
        --dataset_name wikitext,Linly-AI/Chinese-pretraining-dataset \
        --dataset_config_name wikitext-2-raw-v1,none \
    3.注意数据集合的格式要满足要求，待规范补充

# 强化学习说明

    Zero3支持：    ORPO，DPO，RM,GRPO
    Zero3不支持：  PPO
    Zero2支持：    GRPO，PPO，ORPO，DPO，RM
    
    GRPO： 理论上支持Zero3，工程上不稳定，不建议采用
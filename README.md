### 作者信息

    联系人： 刘向前  
    微信：   13552482980
    QQ:     1012088761

### 测试机器

    RTX50系列芯片上单卡测试通过

### 单机执行流程

1.分词训练

    对于医疗领域，分词需要专业化

```bash
    cd 0.分词增量训练
    
    sh run_train.sh
    
    sh run_merge.sh
```

[分词训练文档](0.分词增量训练/README.md)

2.增量预训练

    对于医疗领域，通用基座大模型需要再次预训练，得到医疗领域大模型，作为医疗领域的基座大模型

```bash
   cd 1.增量预训练
   
   sh run_pretrain.sh
```

[增量预训练](1.增量预训练/README.md)

3.微调大模型

拥有了医疗基座大模型，需要让模型学习医疗领域的工作方式，例如：问诊，开方等

```bash
  cd 2.微调
  
  sh run_sft.sh
```

[微调大模型](2.微调/)

4.强化学习

如果微调效果达不到要求，需要强化学习，更进一步对齐人类偏好，强化学习有多种方式，任一选择

4.1 DPO

```bash
  cd 3.强化/DPO
  sh run_dpo.sh
  
```

[DPO文档](3.强化/DPO/README.md)

4.2 ORPO

```bash
  cd 3.强化/ORPO
  sh run_orpo.sh
```

[ORPO文档](3.强化/ORPO/README.md)

4.3 GRPO

```bash
  cd 3.强化/GRPO
  sh run_grpo.sh
```

[GRPO文档](3.强化/GRPO/README.md)

4.4 PPO

```bash
# 训练奖励模型
  cd 3.强化/RM
  sh run_rm.sh
  
# 训练PPO模型
  cd 3.强化/PPO
  sh run_ppo.sh
```

[奖励模型训练](3.强化/RM/README.md)

[PPO模型训练](3.强化/PPO/README.md)

5. 量化大模型

```bash
   cd 4.量化/
   sh run_quantiz.sh
```

[量化大模型](4.量化/README.md)

6.蒸馏

```bash
    cd 5.蒸馏/
    sh run_train.sh
```

7.模型评估

```bash
  # 困惑度
  cd 6.评估/
  sh perplexity.sh
```

8.lora合并

微调或强化学习的模型为lora方式，会得到模型的adapter部分
调用模型需要完整的模型，这时候需要merge adapter部分和训练的基础模型

9.服务

    提供了两种架构的服务
    基于fastapi和gradio两种版本

[fastapi服务启动文档](8.服务/fastapi/README.md)
[gradio服务启动文档](8.服务/gradio/README.md)

10.部署

    提供基于vllm的部署脚本和访问脚本

### 分布式训练

    配置环境变量： world_size (总GPU数)
    执行脚本： torch_run目录下有分布式训练脚本

### 部分参考(方便的话帮他们点个赞)

- [Qwen3微调演练平台](https://github.com/lijiayi-ai/Qwen3-FineTuning-Playground) — Qwen3微调演练平台
- [Qwen3医学推理项目](https://github.com/18520339/multi-reward-medical-reasoning) — 医学推理多奖励相关代码
- [Qwen3模型架构](https://zhuanlan.zhihu.com/p/1905976602019464591) — Qwen3模型架构
- [Qwen3增量预训练](https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/148145089) — Qwen3增量预训练
- [Qwen3大模型微调](https://developer.aliyun.com/article/1663178) — Qwen3大模型微调
- [SFT与DPO训练全流程](https://blog.csdn.net/gitblog_00831/article/details/150752889) — SFT与DPO训练全流程
- [医疗大模型](https://github.com/shibing624/MedicalGPT) — 医疗大模型全流程训练

### 📚 Dataset

## 1. 通用数据集合

### 预训练数据集

- Linly-AI/Chinese-pretraining-dataset (16GB中英文无监督、平行语料)
- wikipedia-cn-20230720-filtered (524MB中文维基百科语料)

### 微调数据集

- shibing624/sharegpt_gpt4 (10万条多语言ShareGPT GPT4多轮对话数据集)
- anon8231489123/ShareGPT_Vicuna_unfiltered (9万条英文ShareGPT多轮对话数据集 未确认内容)
- BelleGroup/train_0.5M_CN (50万条中文ChatGPT指令Belle数据集)
- BelleGroup/train_1M_CN (100万条中文ChatGPT指令Belle数据集)
- shibing624/alpaca-zh (2万条中文ChatGPT指令Alpaca数据集)
- Chinese-Vicuna/guanaco_belle_merge_v1.0 (69万条中文指令Guanaco数据集)
- RyokoAI/ShareGPT52K (5万条英文ChatGPT多轮对话数据集)
- BelleGroup/multiturn_chat_0.8M (80万条中文ChatGPT多轮对话数据集)
- FreedomIntelligence/ShareGPT-CN (3.8万条中文ShareGPT多轮对话数据集)
- shibing624/roleplay-zh-sharegpt-gpt4-data (7千条中文角色扮演多轮对话数据集)

### DPO数据集

- shibing624/DPO-En-Zh-20k-Preference (2万条中英文偏好数据集)
- SAGI-1/ultrafeedback_binarized_dpo
- aladinDJ/ultramix-DPO-annotated
- Finnish-NLP/ultrachat_dpo_sft_deepl_kaannetty
- Palash123/dpo_anthropic_hh_rlhf

### PPO数据集

- Dahoas/pythia_125M_ppo_hh_eval_human
-

### RM数据集

- tasksource/oasst1_pairwise_rlhf_reward (2万条多语言oasst1的reward数据集)
- Dahoas/full-hh-rlhf (11万条英文hh-rlhf的reward数据集)
- Dahoas/static-hh (9万条英文reward数据集)
- Dahoas/rm-static (7万条英文Reward数据集)
- yitingxie/rlhf-reward-datasets (7万条英文Reward数据集)
- liyucheng/zhihu_rlhf_3k (3千条中文知乎问答)

## 2. 医疗数据集合

### 预训练数据集

- shibing624/medical (240万条中文医疗数据集，包括预训练、指令微调和奖励数据集)
- MedRAG/pubmed
- suolyer/pile_pubmed-central (海量医学文献摘要与全文)
- hejazizo/mimic-iii (临床笔记、出院摘要、护理记录等)
- raphus/clinical_trials_gov_COMP631_project

### 微调数据集

- qiaojin/PubMedQA
- shibing624/huatuo_medical_qa_sharegpt (22万条中文医疗对话数据集，华佗项目)
- shibing624/medical (240万条中文医疗数据集，包括预训练、指令微调和奖励数据集)
- MedAlpaca (medalpaca/medical_meadow_*)
- MedMCQA (medmcqa，4选1医疗考试题)
- iCliniq (真实在线问诊)
- HealthCareMagic (真实医生回答)
- CBLUE (中文医疗必备，包含医疗问答、诊断推理、实体识别)
- MedQuAD (约47k问答对)
- BigBIO (100+生物医疗NLP任务数据集)
- dthung/med-fact-check-sft-dataset

### DPO数据集

- shibing624/medical (240万条中文医疗数据集，包含奖励数据集)
- FineMed (~33,000 DPO样本，三元组+SFT样本)
- Anthropic (可用于医学prompt微调)
- HealthCareMagic (医生问答，需构造)
- iCliniq (医生问答，需构造)
- liyucheng/zhihu_rlhf_3k (3千条中文知乎问答偏好数据集)

### GRPO数据集

- MedQA (USMLE/CMLE，多选QA，correctness/reasoning)
- MedMCQA (多选QA，医学知识评测，医学正确性)
- PubMedQA (文献QA，循证医学，evidence/factuality)
- 自建医疗错误集 (错误样本，hallucination penalty)

### ORPO数据集

- daqc/medicina-qa-dpo-orpo-format-es

### PPO数据集

- OpenAssistant/oasst1 (原版oasst1数据集)

### RM数据集

- shibing624/medical (240万条中文医疗数据集，包含奖励数据集)
- HealthCareMagic (医生问答，需构造)
- iCliniq (医生问答，需构造)
- MedQA (USMLE/CMLE，多选QA，需构造)
- MedMCQA (多选QA，医学知识评测，需构造)
- PubMedQA (文献QA，循证医学，需构造)
- Anthropic HH (偏好对比，有害性对齐)
- tasksource/oasst1_pairwise_rlhf_reward (2万条多语言oasst1的reward数据集)
- Dahoas/full-hh-rlhf (11万条英文hh-rlhf的reward数据集)
- Dahoas/static-hh (9万条英文reward数据集)
- Dahoas/rm-static (7万条英文Reward数据集)
- yitingxie/rlhf-reward-datasets (7万条英文Reward数据集)
- liyucheng/zhihu_rlhf_3k (3千条中文知乎问答)

### 使用流程

    数据采集 → 脱敏清洗 → 规范化标注 → 模型训练 → 自动 + 人工评测 → 安全策略 + 上线
    1.数据采集
        不同的训练需要不同字段
        确认数据集核心字段是否存在或能够转化

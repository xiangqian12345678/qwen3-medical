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

4.2 ORPO

```bash
  cd 3.强化/ORPO
  sh run_orpo.sh
```

4.3 GRPO

```bash
  cd 3.强化/GRPO
  sh run_grpo.sh
```

4.4 PPO

```bash
# 训练奖励模型
  cd 3.强化/RM
  sh run_rm.sh
  
# 训练PPO模型
  cd 3.强化/PPO
  sh run_ppo.sh
```

5. 量化大模型

   32B大模型部署显存：

| 部署策略              | 显存需求                     |
  |-------------------|--------------------------|
| FP16 / BF16 单卡    | ≥ 80GB（模型 + 激活 + buffer） |
| FP16 + 8K context | 90–100GB                 |
| FP32 单卡           | ≥ 130–140GB              |

    32B量化大模型部署显存：

| 量化类型      | 权重显存    | 激活显存   | 总显存估计   |
  |-----------|---------|--------|---------|
| FP16      | 64GB    | 6–12GB | 70–80GB |
| INT8      | 32GB    | 6–12GB | 38–44GB |
| INT4      | 16GB    | 6–12GB | 22–28GB |
| INT4+GPTQ | 10–11GB | 6–12GB | 16–23GB |

6.蒸馏

这部分暂时不提供，未来补充
[模型蒸馏](https://github.com/xiangqian19831224/qwen3-pretrain-sft-rl-distill-eval/tree/main/4-%E8%92%B8%E9%A6%8F)

7.模型评估

```bash
  # 困惑度

  sh run_quantize.sh
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

#### 医疗数据集

- 240万条中文医疗数据集(
  包括预训练、指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 22万条中文医疗对话数据集(
  华佗项目)：[shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt)
  【本项目支持格式】

#### 通用数据集

##### Pretraining datasets(预训练数据集)

-

16GB中英文无监督、平行语料[Linly-AI/Chinese-pretraining-dataset](https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset)
-
524MB中文维基百科语料[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)

##### Supervised fine-tuning datasets(指令微调数据集)

- 10万条多语言ShareGPT
  GPT4多轮对话数据集：[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) 【本项目支持格式】
-

9万条英文ShareGPT多轮对话数集：[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
【本项目支持格式】

- 50万条中文ChatGPT指令Belle数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- 100万条中文ChatGPT指令Belle数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
-

5万条英文ChatGPT指令Alpaca数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)

- 2万条中文ChatGPT指令Alpaca数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
- 69万条中文指令Guanaco数据集(
  Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
- 5万条英文ChatGPT多轮对话数据集：[RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
-

80万条中文ChatGPT多轮对话数据集：[BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)

- 116万条中文ChatGPT多轮对话数据集：[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)
-

3.8万条中文ShareGPT多轮对话数据集：[FreedomIntelligence/ShareGPT-CN](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-CN)
-
130万条中文微调数据集（汇总）：[zhuangxialie/Llama3-Chinese-Dataset](https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset/dataPeview)
【本项目支持格式】

-

7千条中文角色扮演多轮对话数据集：[shibing624/roleplay-zh-sharegpt-gpt4-data](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)
【本项目支持格式】

#### Preference datasets(偏好数据集)

-

2万条中英文偏好数据集：[shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)
【本项目支持格式】

- 原版的oasst1数据集：[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
-

2万条多语言oasst1的reward数据集：[tasksource/oasst1_pairwise_rlhf_reward](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)

- 11万条英文hh-rlhf的reward数据集：[Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf)
- 9万条英文reward数据集(来自Anthropic's Helpful Harmless
  dataset)：[Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)
- 7万条英文reward数据集（来源同上）：[Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
-

7万条繁体中文的reward数据集（翻译自rm-static）[liswei/rm-static-m2m100-zh](https://huggingface.co/datasets/liswei/rm-static-m2m100-zh)

- 7万条英文Reward数据集：[yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
- 3千条中文知乎问答偏好数据集：[liyucheng/zhihu_rlhf_3k](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)
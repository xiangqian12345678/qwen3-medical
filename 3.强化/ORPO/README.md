## 训练命令

### 单机训练

```bash 
  sh run_orpo.sh 
```

### 分布式训练

[分布式训练文档](./torchrun/README.md)

## 数据处理流程

    本地文件：jsonl
        {
            "system":"",
            "history":[],
            "question":"20个关于新鲜果汁菜单的口号，适用于一家名为\"Dishes\"的餐厅",
            "response_chosen":"
                这里是一个名为“Dishes”的餐厅的20个口号，突出了其新鲜果汁菜单：
                1. “品尝Dishes新鲜果汁，感受不同！”
                2. “新鲜榨取，直达您的餐桌 - Dishes果汁纯享！”
                ...
                20. “Dishes：果汁永远新鲜，味道永远美味！”
                ",
            "response_rejected":"
                1. \"与菜肴一起品尝新鲜！\"
                2. \"菜肴：新鲜果汁，新的开始！\"
                ...
                20. \"菜肴：新鲜始终是你一天的首选\"
            "
        }
    格式处理：
        {
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n20个关于新鲜果汁菜单的口号，适用于一家名为\"Dishes\"的餐厅<|im_end|>\n<|im_start|>assistant\n",
            "chosen": "这里是一个名为“Dishes”的餐厅的20个口号，突出了其新鲜果汁菜单：1. “品尝Dishes新鲜果汁，感受不同！” ... 20. “Dishes：果汁永远新鲜，味道永远美味！”",
            "rejected": "1. \"与菜肴一起品尝新鲜！\" ... 20. \"菜肴：新鲜始终是你一天的首选\""
        }
    说明：
        - prompt 由 Conversation 模板生成：
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
        - chosen / rejected 是对应模型优选与非优回答。
        - 长度过滤保证 prompt+response 不超过 max_source_length + max_target_length。
        - train_dataset 和 eval_dataset 可以直接输入 DPOTrainer。

## 标准数据格式

    {
      "prompt": "用户输入的内容",
      "chosen": "被标记为更好、更优的模型回复",
      "rejected": "被标记为较差、次优的模型回复"
    }

    数据集合只需要转为下面格式就可以适配本程序：
        { 
            "question":" ",
            "response_chosen":" ",
            "response_rejected":" "
        }

## 开源数据使用

    数据集合只需要转为下面格式就可以适配本程序：
    { 
        "question":" ",
        "response_chosen":" ",
        "response_rejected":" "
    }

## 开源医疗数据

| 数据集                                                                      | 样本量            | 数据类型                             | 偏好对来源                                                        | 特点 / 用途                                    | 开源情况                |
|--------------------------------------------------------------------------|----------------|----------------------------------|--------------------------------------------------------------|--------------------------------------------|---------------------|
| ChiMed‑DPO                                                               | ~24,000        | 三元组 (prompt + chosen + rejected) | Zhongjing_rlhf (~20k 医学研究生/医生标注) + MedicalGPT 偏好对 (~4k 医生选择) | 专门用于医学聊天模型的 DPO 训练；强调符合专业医生偏好              | 科研项目数据，开源程度随论文/代码更新 |
| FineMed                                                                  | ~33,000 DPO 样本 | 三元组 + SFT 样本                     | 医学 SFT + 人类偏好标注                                              | 综合医学模型训练数据集；可用于 SFT → DPO 微调流程；强化复杂医学推理能力  | 已公开信息可用，但可能需申请      |
| KoMeP (Korean Medical Preference Dataset)                                | ~5,500         | 偏好对 (chosen vs rejected)         | 五个 LLM 生成回答对 + 人类偏好标注                                        | 韩国医学偏好数据集；可用于 alignment tuning（DPO / ORPO） | 研究成果，关注论文/仓库开源情况    |
| MedicalGPT 项目相关偏好数据                                                      | 不固定            | 偏好对可提取                           | 中文医疗语料 + 多模型生成首选/拒绝对                                         | 用于 SFT + RLHF + DPO 流程；可自行提取奖励数据构建 DPO     | GitHub 项目可获取部分训练资源  |
| 通用 DPO 数据集 (UltraFeedback / UltraMix / UltraChat DPO, Anthropic HH, SHP) | 不同             | 偏好对                              | 通用人类偏好标注                                                     | 可用于医学 prompt 微调，提升通用偏好对齐能力                 | 大多数开源               |






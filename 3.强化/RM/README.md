## 数据处理流程

    输入样例：
        {
            "system": "",
            "history": [],
            "question": "20个关于新鲜果汁菜单的口号，适用于一家名为\"Dishes\"的餐厅",
            "response_chosen": "这里是一个名为\"Dishes\"的餐厅的20个口号，突出了其新鲜果汁菜单：1. \"品尝Dishes新鲜果汁，感受不同！\" 2. \"新鲜榨取，直达您的餐桌 - Dishes果汁纯享！\" ... 20. \"Dishes：果汁永远新鲜，味道永远美味！\"",
            "response_rejected": "1. \"与菜肴一起品尝新鲜！\" 2. \"菜肴：新鲜果汁，新的开始！\" ... 20. \"菜肴：新鲜始终是你一天的首选\""
        }
    步骤一： 构建 Prompt（
    模板：
        Conversation(
            name="qwen",
            system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            roles=("user", "assistant"),
            prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
            sep="\n",
            stop_str="<|im_end|>"
        )
    处理：
        {
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n20个关于新鲜果汁菜单的口号，适用于一家名为\"Dishes\"的餐厅<|im_end|>\n<|im_start|>assistant\n",
            "chosen": "这里是一个名为\"Dishes\"的餐厅的20个口号，突出了其新鲜果汁菜单：1. \"品尝Dishes新鲜果汁，感受不同！\" 2. \"新鲜榨取，直达您的餐桌 - Dishes果汁纯享！\" ... 20. \"Dishes：果汁永远新鲜，味道永远美味！\"",
            "rejected": "1. \"与菜肴一起品尝新鲜！\" 2. \"菜肴：新鲜果汁，新的开始！\" ... 20. \"菜肴：新鲜始终是你一天的首选\""
        }
    步骤二： 分词
        {
            "prompt_input_ids": [
                151644,   # <|im_start|>
                8948,     # system
                198,      # \n
                517,      # You
                ...
                14989,    # assistant
                151645,   # <|im_end|>
                198,      # \n
                151644,   # <|im_start|>
                77091,    # user
                198,      # \n
                20,       # 2
                ...
                585,      # 餐
                292,      # 厅
                151645,   # <|im_end|>
                198,      # \n
                151644,   # <|im_start|>
                77091,    # assistant
                198       # \n
            ],

            "chosen_input_ids": [
                151644, 8948, ..., 198,
                378,      # 这
                268,      # 是
                ...
                30,       # "
                # ... 更多口号内容 ...
                151645    # <|im_end|>
            ],

            "rejected_input_ids": [
                # prompt 部分同上
                151644, 8948, ..., 198,
                48, 236, 32, 30, 332,  # 1. "\""
                338,      # 与
                ...
                33,       # ！
                30,       # "
                # ... 更多内容 ...
                151645    # <|im_end|>
            ]
        }
    步骤三：长度截断与填充
        tokenized_dataset = Dataset([
            {
                "input_ids_chosen": [151644, 8948, 198, ...],
                "attention_mask_chosen": [1, 1, 1, ...],
                "input_ids_rejected": [151644, 8948, 198, ...],
                "attention_mask_rejected": [1, 1, 1, ...]
            },
            # ... 更多样本
        ])

    步骤四：最终结果
        train_dataset = Dataset([
            {
                "input_ids_chosen": [151644, 8948, 198, ...],  # ← 编码结果
                "attention_mask_chosen": [1, 1, 1, ...],
                "input_ids_rejected": [151644, 8948, 198, ...],  # ← 编码结果
                "attention_mask_rejected": [1, 1, 1, ...]
            },
            # ... 保留长度符合要求的样本
        ])
    """

## 标准格式

    1.基础 Pairwise 格式
        {
          "prompt": "感冒了应该吃什么药？",
          "chosen": "感冒可以根据症状服用对乙酰氨基酚或布洛芬缓解发热，多喝水，注意休息。",
          "rejected": "感冒不用管，扛一扛就好了。"
        }
    2.扩展版 Pairwise（工业级常用）
        {
          "prompt": "白带增多发黄是怎么回事？",
          "chosen": "白带发黄可能与阴道炎有关，建议及时就医进行检查，如白带常规。",
          "rejected": "白带发黄很正常，不用管。",
          "source": "medical_annotation",
          "confidence": 0.92,
          "category": "gynecology"
        }
    3.Listwise / Multi-response 格式
        {
          "prompt": "高血压患者饮食注意什么？",
          "responses": [
            { "text": "低盐饮食，控制体重，避免高脂食物", "rank": 1 },
            { "text": "想吃什么吃什么", "rank": 3 },
            { "text": "适当运动，戒烟限酒", "rank": 2 }
          ]
        }
        使用方式： 转成 N 选 2 的 pairwise
    4.直接给 reward 分数
        {
          "prompt": "糖尿病能吃水果吗？",
          "response": "可以适量吃低GI水果，如苹果、梨。",
          "reward": 0.87
        }

## 医疗大模型建议结构

    {
      "prompt": "白带增多发黄怎么办？",
      "chosen": "白带发黄可能提示感染，建议到医院做白带常规检查，明确病因后治疗。",
      "rejected": "这是正常现象，不用担心。",
      "domain": "medical",
      "risk_level": "high"
    }

## 开源数据集

| 序号 | 数据集名称           | 数据类型          | 原始任务       | 是否直接可用于 RM    | 适用训练方式       | 可提供的奖励信号                           | 主要优点                      | 主要问题     |
| -- |----------------------|-----------------|---------------|-------------------| --------------- |------------------------------------------|-----------------------------|-------------|
| 1  | MedPreference        | 偏好对比（A/B）   | 医疗对话偏好    | ✅ 是               | RM / DPO / GRPO | helpfulness / safety / professionalism  | 医疗版偏好数据，最接近真实医生标准| 规模相对有限  |
| 2  | HealthCareMagic      | 医生问答         | 在线医疗咨询    | ⚠️ 需构造            | RM / DPO        | 准确性 / 医生风格                         | 真实医生回答，语言自然          | 年代旧、需脱敏 |
| 3  | iCliniq              | 医生问答         | 医疗咨询       | ⚠️ 需构造            | RM / DPO        | 安全性 / 可信度                            | 高质量医生文本                | 无显式偏好标注 |
| 4  | MedQA (USMLE / CMLE) | 多选 QA         | 医学考试       | ⚠️ 需构造            | RM / GRPO       | correctness / reasoning                  | 标准答案明确                  | 非对话风格    |
| 5  | MedMCQA              | 多选 QA         | 医学知识评测    | ⚠️ 需构造            | RM / GRPO       | 医学正确性                                 | 覆盖面广                     | 缺乏安全信号  |
| 6  | PubMedQA             | 文献 QA         | 循证医学       | ⚠️ 需构造            | RM / GRPO       | evidence / factuality                    | 可做循证 reward              | 语言偏学术    |
| 7  | MIMIC-III            | 临床病历         | 临床决策       | ❌ 不直接             | 专用 RM           | 安全 / 决策一致性                        | 真实临床数据                  | 门槛高、强约束 |
| 8  | MIMIC-IV             | 临床病历         | 临床决策       | ❌ 不直接             | 专用 RM           | 安全 / 风险控制                          | 数据更新                      | 不适合通用助手|
| 9  | Medical Red Team     | 风险对话         | 安全测试       | ⚠️ 需构造            | RM / DPO        | safety / refusal                         | 覆盖高风险场景                 | 不提升专业能力|
| 10 | Anthropic HH         | 偏好对比         | 有害性对齐     | ✅ 部分              | RM / DPO        | harmlessness / safety                    | 偏好信号清晰                   | 医疗深度有限 |
| 11 | 自建医疗错误集         | 错误样本         | 风险控制       | ✅ 是               | RM / GRPO       | hallucination penalty                    | 对齐效果极强                    | 需要专家规则 |

























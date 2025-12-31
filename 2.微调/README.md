# 使用说明

## 1.快捷单机测试

    本地数据            run_sft.sh
    依赖开源数据，输入参数可以仿照增量预训练添加

## 2.分布式训练

- 基于torchrun命令    [使用说明](./torchrun/README.md)

## 3.数据格式

### 3.1 数据处理流程（以对话为例）

    qwen模板：
        Conversation(
            name="qwen",
            system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            messages=[],
            roles=("user", "assistant"),
            prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
            sep="\n",
            stop_str="<|im_end|>",
        )
    程序处理流程
        输入： 
            {"conversations":[
                {"from":"human","value":"你好"},
                {"from":"gpt","value":"你好！我是Qwen助手"},
                {"from":"human","value":"感冒吃什么药？"},
                {"from":"gpt","value":"建议感冒多喝水"}]
            }
        处理： messages = [["你好", "你好！我是Qwen助手"], ["感冒吃什么药？", "建议感冒多喝水"]]
        模板化：
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            你好<|im_end|>
            <|im_start|>assistant
            你好！我是Qwen助手<|im_end|>
            <|im_start|>user
            感冒吃什么药？<|im_end|>
            <|im_start|>assistant
            建议感冒多喝水<|im_end|>
        编码：最终训练数据
            {
                "input_ids": [[1, 234, 567, 890, 123, 456, 789, 2]],  # token序列
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],           # 注意力掩码
                "labels": [[-100, -100, 567, 890, 123, 456, 789, 2]]  # 标签（human部分为-100）
            }

### 3.2 模板类别

    1.对话模板
        Conversation(
            name="qwen",
            system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            messages=[],
            roles=("user", "assistant"),
            prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
            sep="\n",
            stop_str="<|im_end|>",
        )
        
    2.多轮对话模板
        Conversation(
            name="multi_turn",
            system_prompt="<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n",
            messages=[
                {"role": "user", "content": "{query1}"},
                {"role": "assistant", "content": "{response1}"},
                {"role": "user", "content": "{query2}"}
            ],
            roles=("user", "assistant"),
            prompt="<|im_start|>{role}\n{content}<|im_end|>\n",
            sep="\n",
            stop_str="<|im_end|>",
        )
        例如：
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            你好<|im_end|>
            <|im_start|>assistant
            你好！我是Qwen助手<|im_end|>
            <|im_start|>user
            感冒吃什么药？<|im_end|>
            <|im_start|>assistant
            建议感冒多喝水，适当休息，可根据症状考虑感冒药<|im_end|>

    3.指令模板（Instruction/Instruction-tuning）
        Instruction(
            name="qwen_instruction",
            instruction_template="以下是指令：{instruction}\n输入：{input}\n请生成输出。",
            input_field="input",
            output_field="output",
            stop_str="<|im_end|>"
        )
        例如：
            <|im_start|>system
            You are a helpful assistant for executing instructions.<|im_end|>
            <|im_start|>user
            指令：将以下句子翻译成英文
            输入：今天天气很好<|im_end|>
            <|im_start|>assistant
            The weather is nice today.<|im_end|>

    4.问答模板（Question Answering）
        QA(
            name="qwen_qa",
            prompt_template="问题：{question}\n回答：",
            input_field="question",
            output_field="answer",
            stop_str="\n"
        )
        例如：
            <|im_start|>system
            You are a knowledgeable assistant.<|im_end|>
            <|im_start|>user
            问题：中国的首都是哪里？<|im_end|>
            <|im_start|>assistant
            北京<|im_end|>

    5.文本生成/完成模板（Text Completion）
        Completion(
            name="qwen_completion",
            prompt_template="请根据以下提示生成文本：{prompt}\n输出：",
            input_field="prompt",
            output_field="completion",
            stop_str="\n"
        )
        例如：
            <|im_start|>system
            You are a knowledgeable assistant.<|im_end|>
            <|im_start|>user
            问题：中国的首都是哪里？<|im_end|>
            <|im_start|>assistant
            北京<|im_end|>
            
            带上下文：
            <|im_start|>system
            You are a knowledgeable assistant.<|im_end|>
            <|im_start|>user
            上下文：中国的首都是北京，上海是最大城市。
            问题：中国的首都是哪里？<|im_end|>
            <|im_start|>assistant
            北京<|im_end|>


    6.分类/情感分析模板
        Classification(
            name="qwen_classification",
            prompt_template="文本：{text}\n请给出分类标签：",
            input_field="text",
            output_field="label",
            stop_str="\n"
        )
        例如：
            <|im_start|>system
            You are a text classifier.<|im_end|>
            <|im_start|>user
            文本：这部电影真好看！<|im_end|>
            <|im_start|>assistant
            正面<|im_end|>

## 4.开源数据

    1. MIMIC 系列（临床文本与结构化数据）
        MIMIC‑III / MIMIC‑IV：包含数万名 ICU 患者的去识别化临床笔记、出院摘要、护理记录等，适合文本分类、实体识别、摘要等任务 
    2. n2c2 / i2b2 NLP Data Sets
        一系列临床自然语言处理竞赛的真实标注数据，覆盖实体提取、关系抽取、共指消解等任务 
    3. PubMed & PubMedQA
        PubMed 文献摘要：超过数千万条医学文献摘要，可用来做语言理解、科学问答等预训练/微调语料。 
        PubMedQA：针对生物医学科研问答（含问题、背景语境、答案），适合 QA 型微调。  
    4. MedQuAD
        一个结合多源医学问答的开放数据集（约 47k 问答对），常用于 QA 任务微调。  
    5. BigBIO 生态数据集集合
        BigBIO 是一个开源框架和仓库，收录了 100+ 生物医疗 NLP 任务数据集（NER、关系提取、QA 等）。适合做多任务学习或 meta‑dataset 训练。   
    6.中文医疗 NLP 资源
        CMeKG / 中文医学指令集：用于中文医疗问答、对话等微调（在一些开源项目如 ChatGLM‑Medical 中使用）。 
        MedFact：首个面向 LLM 生成内容的中文医学实证事实核查数据集，含问题与声明。 
    7.医学实体与关系数据集
        MedMentions：基于 PubMed 的实体链接数据集，可用于 NER/链接任务。 
        COMETA：医疗实体链接语料（可联系作者获取）。 
    8.PubMed / PubMed Central (PMC)：
        海量医学文献摘要与全文，可作为语料源
    9.ClinicalTrials.gov 
        临床试验 NLP 数据集
    10.UMLS / Metathesaurus
        医学术语与概念本体，可用于构建自定义标注集或知识增强

## 5.开源数据处理

    收集 & 清洗数据（疾病描述、症状、治疗、临床对话等）
    任务定义（QA、摘要、实体识别、对话生成等）
    格式转换（如 instruction → response、Question → Answer）
    微调训练
        使用 LoRA / QLoRA 等高效微调技术节省显存
        结合 domain‑instruction formatting 提升准确性
    评估 & 安全性测试
        使用专门的医疗 benchmark 或事实核查数据集（如 MedFact）
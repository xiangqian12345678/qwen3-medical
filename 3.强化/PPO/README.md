## 数据流程

    输入样例：
        // 输入数据（JSON格式）
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！我是Qwen助手"},
                {"from": "human", "value": "感冒吃什么药？"},
                {"from": "gpt", "value": "建议感冒多喝水"}
            ]
        }

    步骤A：提取messages
        输入：conversations字段
        输出：messages = ["你好", "你好！我是Qwen助手", "感冒吃什么药？", "建议感冒多喝水"]
    步骤B：构建history（问答对）
        输出：history = [
        ["你好", "你好！我是Qwen助手"],
        ["感冒吃什么药？", "建议感冒多喝水"]
    步骤C：应用模板格式化
        dialog = prompt_template.get_dialog(history)
        # 输出dialog列表（偶数索引是问题，奇数索引是答案）：
        [
            # 第1轮对话
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
            "你好！我是Qwen助手",
            
            # 第2轮对话
            "\n<|im_start|>user\n感冒吃什么药？<|im_end|>\n<|im_start|>assistant\n",
            "建议感冒多喝水"
        ]
    步骤D：分词
        # 对dialog[0]和dialog[2]进行分词（只分问题部分）
        tokenized_dialog_0 = tokenizer("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n")
        # 输出：{"input_ids": [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 2610, ...]}
        
        tokenized_dialog_2 = tokenizer("\n<|im_start|>user\n感冒吃什么药？<|im_end|>\n<|im_start|>assistant\n")
        # 输出：{"input_ids": [198, 151644, 872, 198, 23395, 9398, 998, 162, 151645, 198, 151644, 77091, 198, ...]}
    最终：
        {
            "input_ids": [
                # 样本1：第1轮对话的问题部分
                [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 2610, 1234, 151645, 198, 151644, 77091, 198],
                
                # 样本2：第2轮对话的问题部分
                [198, 151644, 872, 198, 23395, 9398, 998, 162, 151645, 198, 151644, 77091, 198]
            ]
        }

## 开源数据集

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

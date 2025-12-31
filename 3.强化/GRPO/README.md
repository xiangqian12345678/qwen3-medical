## 数据处理流程

    本地文件： jsonl
        {"question": "肛门病变可能是什么疾病的症状?", "answer": "食管克罗恩病"}
        {"question": "白带增多发黄怎么个情况？，最近几天，感觉自己的内裤一天下来会有比较多偏黄的粘液痕迹。刚开始以为是尿液比较黄，经过仔细观察发觉是泛黄的白带。医生，您好白带增多泛黄怎么回事？", "answer": "白带增多泛黄应该是患阴道炎了，而且是细菌性阴道炎的可能性大，你可以用妇科洗剂擦洗外阴阴道，同时口服些妇科千金片或者花红颗粒，具体用药谨遵医嘱，如果效果不佳。较好到医院去做白带常规仔细检查，查明病因后必要时再出针对性的用药治疗。平常以清淡饮食为基础，尽量多喝水。白带增多患者可以在理解专业治疗的同时，在生活中用食疗做为辅助治疗，在日常生活饮食中需要有恰当饮食，防止刺激性食物。"}
        {"question": "天雄的药用植物栽培是什么?", "answer": "1.气候土壤：天雄喜温暖、湿润和向阳环境，宜选土层深厚肥沃、土质疏松、排水良好的土壤栽种。2.种植：根块繁殖，选大的子根或野生乌头作种。在冬至前6-10天栽种，行距1尺，株距5寸，种前施足底肥，开2寸深沟，使芽向上栽入沟内，覆土1-2寸，每亩用种根240-300斤，地冻前浇水，并盖土粪一层越冬。等二年苗高1尺左右，摘除顶。一般留6-8片叶，并随时去腋芽。3.田间管理：生长时间，有根腐病和霜霉病害，应加强管理，注意排水通风。霜霉病发病初期可喷射波尔多液或代森铵防治。4.采集加工：于6月底7月初时挖取根部，洗净泥土，晒干。炮制方法1.《雷公炮炙论》：宜炮皴坼后去皮尖底用，不然阴制用并得。2.《药性论》：干姜制用之。3.《日华子本草》：凡丸、散炮去皮脐用，饮药即和皮生使甚佳。4.《纲目》：熟用。一法，每十两，以酒浸七日，掘土坑，用炭半枰煅赤，去火，以醋二升沃之，候干，趁热入天雄在内，小盆合一夜，取出，去脐用之。"}
    格式处理：
        {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': '肛门病变可能是什么疾病的症状?'}
            ],
            'answer': '食管克罗恩病'
        }
    模式输入数据：
        格式处理后的数据
    
    关键配置GRPOConfig: 自动生成多个二回答，根据与标准答案的差距进行打分和排序
        - num_generations: 每个prompt生成的响应数量（在run_grpo.sh中配置为4）
        - max_prompt_length: 输入prompt的最大token长度（16384）
        - max_completion_length: 生成答案的最大token长度（512）
    
    训练：
        sh run_grpo.sh

## 标准的GRPO数据

    1.单论对话
        数据格式：
            {
              "prompt": "A 45-year-old male presents with chest pain radiating to the left arm. What is the most likely diagnosis?",
              "group_id": "case_0001",
              "responses": [
                {
                  "response_id": "r1",
                  "text": "The symptoms are most consistent with acute myocardial infarction.",
                  "reward": 0.92
                },
                {
                  "response_id": "r2",
                  "text": "This could be caused by gastric reflux disease.",
                  "reward": 0.35
                },
                {
                  "response_id": "r3",
                  "text": "Chest pain can have many causes and further tests are needed.",
                  "reward": 0.58
                }
              ]
            }
        格式处理：
            {
                "input": "A 45-year-old male presents with chest pain radiating to the left arm. What is the most likely diagnosis?", 
                "output": "The symptoms are most consistent with acute myocardial infarction.", 
                "reward": 0.92, 
                "group_id": 
                "case_0001", 
                "response_id": "r1"
            }
            {
                "input": "A 45-year-old male presents with chest pain radiating to the left arm. What is the most likely diagnosis?", 
                "output": "This could be caused by gastric reflux disease.", 
                "reward": 0.35, 
                "group_id": 
                "case_0001", 
                "response_id": "r2"
            }
            {
                "input": "A 45-year-old male presents with chest pain radiating to the left arm. What is the most likely diagnosis?", 
                "output": "Chest pain can have many causes and further tests are needed.", 
                "reward": 0.58, 
                "group_id": "case_0001", 
                "response_id": "r3"
            }
    2.多轮对话格式
        输入数据格式:
            {
              "prompt": "问题描述",
              "group_id": "分组ID",
              "responses": [
                {"response_id": "r1", "text": "回答1", "reward": 0.92},
                {"response_id": "r2", "text": "回答2", "reward": 0.35},
                {"response_id": "r3", "text": "回答3", "reward": 0.15},
                {"response_id": "r4", "text": "回答4", "reward": 0.05},
              ]
            }
    
        输出数据格式（每个样本  一个样例会生成多个样本）:
            {
              "prompt": [
                {"role": "system", "content": "A conversation between User and Assistant..."},
                {"role": "user", "content": "问题描述"}
              ],
              "response": "回答1",
              "reward": 0.92,
              "group_id": "分组ID",
              "response_id": "r1"
            }
            ...

    训练：
        # 程序采用多轮对话格式（适配单论对话语料）
        sh run_grpo_standard.sh  

## GRPO开源数据

    MedAlpaca
        HF：medalpaca/medical_meadow_*
        内容：医疗问答（instruction-following）
        覆盖：诊断、用药、检查、解释
        选择原因：
            指令型 QA 非常干净
            可以：
            对同一 prompt 采样多回答
            用规则 / 医疗 LLM 打分
            直接构造成 group
    MedMCQA
        HF：medmcqa
        4 选 1 医疗考试题
        明确正确答案
    PubMedQA
        HF：pubmed_qa
        Yes / No / Maybe + 解释
    iCliniq/HealthCareMagic（真实医生回答）
        HF 上有多个清洗版本（非官方）
        来源：真实在线问诊
    MIMIC-III / MIMIC-IV（临床文本王者）
        出院小结、检查、病程
    CBLUE（中文医疗必备） 包含：
        医疗问答
        诊断推理
        实体识别
    Med-RLHF
        医疗问答 + 偏好
    Health-RLHF
        医疗安全/合规
# 使用说明

## 1.快捷单机测试

    本地数据            run_pretrain.sh
    开源数据+本地数据     run_pretrain2.sh

## 2.分布式训练

- 基于deepspeed命令   [使用说明](./deepspeed/README.md)
- 基于torchrun命令    [使用说明](./torchrun/README.md)

## 3.数据格式

### 本地私有文件

    1.纯文本
        - 格式 纯文本，每行一个样例
        - 例如
            今天天气很好，我们一起去公园散步。
            大模型预训练通常使用大规模无监督文本数据。
            HuggingFace datasets 支持 text/json/jsonl 多种格式。
    2.json或jsonl格式
        - 格式    json/jsonl(必须有一个text字段)
        - 例如：
            {"text": "这是验证集文本。", "source": "eval", "lang": "zh"}
            {"text": "用于评估语言模型的困惑度。", "source": "eval", "lang": "zh"}

### 开源数据集合

    1.纯文本
        - 格式 纯文本，每行一个样例
        - 样例 wikitext/wikitext-103-raw-v1 
        - 例如
            = Plain maskray =
            The plain maskray or brown stingray (Neotrygon annotata) is a species of stingray in the family Dasyatidae. It is found in shallow, soft-bottomed habitats off northern Australia. Reaching 24 cm (9.4 in) in width, this species has a diamond-shaped, grayish green pectoral fin disc. Its short, whip-like tail has alternating black and white bands and fin folds above and below. There are short rows of thorns on the back and the base of the tail, but otherwise the skin is smooth. While this species possesses the dark mask-like pattern across its eyes common to its genus, it is not ornately patterned like other maskrays.
            Benthic in nature, the plain maskray feeds mainly on caridean shrimp and polychaete worms, and to a lesser extent on small bony fishes. It is viviparous, with females producing litters of one or two young that are nourished during gestation via histotroph ("uterine milk"). This species lacks economic value but is caught incidentally in bottom trawls, which it is thought to be less able to withstand than other maskrays due to its gracile build. As it also has a limited distribution and low fecundity, the International Union for Conservation of Nature (IUCN) has listed it as Near Threatened.
            = = Taxonomy and phylogeny = =
            The first scientific description of the plain maskray was authored by Commonwealth Scientific and Industrial Research Organisation (CSIRO) researcher Peter Last in a 1987 issue of Memoirs of the National Museum of Victoria. The specific name annotatus comes from the Latin an ("not") and notatus ("marked"), and refers to the ray's coloration. The holotype is a male 21.2 cm (8.3 in) across, caught off Western Australia; several paratypes were also designated. Last tentatively placed the species in the genus Dasyatis, noting that it belonged to the "maskray" species group that also included the bluespotted stingray (then Dasyatis kuhlii). In 2008, Last and William White elevated the kuhlii group to the rank of full genus as Neotrygon, on the basis of morphological and molecular phylogenetic evidence.
            In a 2012 phylogenetic analysis based on mitochondrial and nuclear DNA, the plain maskray and the Ningaloo maskray (N. ningalooensis) were found to be the most basal members of Neotrygon. The divergence of the N. annotata lineage was estimated to have occurred ~ 54 Ma. Furthermore, the individuals sequenced in the study sorted into two genetically distinct clades, suggesting that N. annotata is a cryptic species complex. The two putative species were estimated to have diverged ~ 4.9 Ma; the precipitating event was likely the splitting of the ancestral population by coastline changes.
    2.json或jsonl格式
        - 格式    json/jsonl(必须有一个text字段)
        - 例如：
            {"text": "这是验证集文本。", "source": "eval", "lang": "zh"}
            {"text": "用于评估语言模型的困惑度。", "source": "eval", "lang": "zh"}

## 4.预训练数据集

### 4.1 开源通用中文数据

- linly-ai/chinese-pretraining-dataset

  汇聚百科、新闻、小说、网页等多来源中文语料，已做基础清洗，适合大模型训练
- THUCNews

  新闻分类语料，较小规模，可做预训练或微调
- WudaoCorpus

  清华/悟道团队收集的超大规模中文语料（非公开，需要授权）
- ChineseGLUE 数据集

  主要用于下游任务评估，但可辅助预训练做语义理解

### 4.2 开源医疗数据

#### 医疗通用语料 / 预训练语料

* **shibing624/medical**
  ：中文医疗大数据集，包括百科、教材文本与中英文问答对，适合中文医疗预训练/微调。([链接](https://huggingface.co/datasets/shibing624/medical))

#### 多语种 / 医疗健康文本

* **nlp‑guild/medical‑data**：社区贡献的医疗 NLP
  数据集合集，包含疾病、症状、药物等文本样例。([链接](https://huggingface.co/datasets/nlp-guild/medical-data))
* **Med‑dataset/Med_Dataset**：集合多个医疗 NLP
  任务的数据，如问答指令对、医患对话、研究问题/答案等。([链接](https://huggingface.co/datasets/Med-dataset/Med_Dataset))
* **kamruzzaman‑asif/Diseases_Dataset**
  ：疾病名称、症状与治疗建议组合的数据，用于分类/生成/信息提取任务。([链接](https://www.selectdataset.com/dataset/0bc1f480d7385ce0e470630d1bb6636e))

#### 临床信息抽取 / 临床 NLP

* **mitclinicalml/clinical‑ie**：与 EMNLP
  研究相关的临床信息抽取数据集（需同意访问条件）。([链接](https://huggingface.co/datasets/mitclinicalml/clinical-ie))

#### 专门整理为医疗预训练任务的合集

* **mkurman/medical‑pre‑training‑datasets**
  ：收集多个适合大模型预训练的英文医学数据集，如临床指南、期刊文章等。([链接](https://huggingface.co/collections/mkurman/medical-pre-training-datasets))
* **NeuML/medical‑and‑scientific‑literature‑datasets**
  ：医学与科学文献语料集合，用于医学语言理解或生物医学知识注入。([链接](https://huggingface.co/collections/NeuML/medical-and-scientific-literature-datasets))

#### 常见的通用医学语料 & 相关数据（可补充预训练语料）

##### 1.公共大规模医学论文与文献

* **PubMed / PubMed Central (PMC)**：生物医学文献数据库，可抓取摘要/全文用于训练。
* **BioASQ / MedQA**：医学问答/多任务数据集，可转换为 Hugging Face 格式。

##### 2.临床电子病历类数据

* **MIMIC 系列（如 MIMIC‑III / MIMIC‑IV / MIMIC‑CXR 报告）**：真实临床数据集，用于临床
  NLP/生成任务（需访问许可）。([链接](https://huggingface.co/foundationmodels/MIMIC-medical-report))

## 5.大模型预训练策略

* **通用语料 + 医疗语料混合预训练**：先用 Common Crawl / RefinedWeb 语料预训练，再用医疗语料做 domain adaptation。
* **任务设计与转换**：将临床问答、指南、医患对话等转换成指令格式或生成格式，适配大语言模型训练。
* **医疗多任务**：NER、分类、实体关系抽取、问答等任务数据集均可用于微调与评估。

---

## 6.Hugging Face 上查找更多医学数据集

```python
from datasets import list_datasets

all_med = [d for d in list_datasets() if "medical" in d]
print(all_med)
```

## 7.数据集总结表

| 类型       | 示例数据集                                                                                       | 典型任务                           |
|----------|---------------------------------------------------------------------------------------------|--------------------------------|
| 中文医疗通用语料 | `shibing624/medical`                                                                        | Continue Pretraining, 微调       |
| 英文医疗语料   | `nlp‑guild/medical‑data`, `Med‑dataset/Med_Dataset`, `kamruzzaman‑asif/Diseases_Dataset`    | Continue Pretraining, QA, 文本生成 |
| 临床信息抽取   | `mitclinicalml/clinical‑ie`                                                                 | NER, 关系抽取                      |
| 医疗文献合集   | `mkurman/medical‑pre‑training‑datasets`, `NeuML/medical‑and‑scientific‑literature‑datasets` | 文献预训练, 知识注入                    |
| 公共文献语料   | PubMed, PMC                                                                                 | 医疗知识注入, 预训练                    |
| 医学问答数据   | BioASQ, MedQA                                                                               | QA, 微调                         |
| 临床电子病历   | MIMIC 系列                                                                                    | 文本生成, 微调, 临床 NLP               |

## 7.数据使用原则

| 数据类型          | 适合预训练     | 适合微调/下游任务 |
|---------------|-----------|-----------|
| 中文医疗百科/教材     | ✅         | ✅         |
| 英文医疗语料（文献、问答） | ✅（文献类大语料） | ✅（问答类）    |
| 临床信息抽取        | ⚠️        | ✅         |
| 医疗文献合集        | ✅         | ✅         |
| 公共文献语料        | ✅         | ✅         |
| 医学问答          | ⚠️        | ✅         |
| 临床电子病历        | ⚠️（需受控）   | ✅         |

## 8.开源数据处理

### 8.1 问答数据

    方法 A：序列拼接 LM 训练
        {
          "text": "[QUESTION] 什么是高血压？ [ANSWER] 高血压是一种以动脉血压升高为特征的慢性病。"
        }
    方法 B：指令式格式 / Instruction Tuning
        {
          "text": "Instruction: 请回答以下问题：什么是高血压？\nOutput: 高血压是一种以动脉血压升高为特征的慢性病。"
        }

    方法 C：多轮对话格式
        {
          "text": "[User] 我最近头痛很严重。\n[Assistant] 你头痛多长时间了？\n[User] 两天了，还有发热。\n[Assistant] 建议去医院检查血压和血常规。"
        }

### 8.2 多字段数据

    例如： nlp‑guild/medical‑data`
    处理：
    1.选择有意义的字段  
        name，desc，category，prvent，cause,sympton,get_way,acompany,cure_departmnt,...
    2.对选择的字段处理  
        如果字段是list结构，需要处理为字符串
        例如: cure_departmnt = [ "药物治疗", "支持性治疗" ]
        转化： cure_departmnt = "治疗方式：药物治疗,支持性治疗"

#### 字段处理原则

| 数据类型      | 处理方式            | 注意事项          |
|-----------|-----------------|---------------|
| str       | 去除空格、特殊符号       | 保留文本内容        |
| list      | 用自然语言符号拼接，或字段标记 | 不保留 `[]`      |
| dict      | 转成自然语言句子或字段标记   | 保留语义          |
| int/float | 转文本，必要时保留单位     | 数值重要性决定是否保留   |
| bool      | 转文本（是/否）或字段标记   |               |
| None/空值   | 跳过              | 避免生成 None 字符串 |

    3.按照一定顺序拼接
        目的： 保持一定的逻辑性
        间隔： 字段之间用句号间隔
        例如： name->desc->category->...
              name = 肺泡蛋白质沉积症
              desc = 肺泡蛋白质沉积症(简称PAP)，又称Rosen-Castle-man-Liebow综合征，是一种罕见疾病。该病以肺泡和细支气管腔
              category = [ "疾病百科", "内科", "呼吸内科" ]
        得到： 名称：肺泡蛋白质沉积症。描述：又称Rosen-Castle-man-Liebow综合征，是一种罕见疾病。该病以肺泡和细支气管腔。类别:疾病百科,内科,呼吸内科



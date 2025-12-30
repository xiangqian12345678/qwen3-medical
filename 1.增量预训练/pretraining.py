"""
增量预训练脚本：对因果语言模型（GPT、GPT-2、CTRL等）进行增量预训练

部分代码参考自 https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import math
import os
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from loguru import logger
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled


@dataclass
class ModelArguments:
    """
    模型相关参数类：定义模型、配置、分词器等相关参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "预训练模型检查点路径，用于权重初始化。如果要从头训练模型，请不要设置此参数。"
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "分词器路径。如果要从头训练模型，请不要设置此参数。"
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "是否以8位模式加载模型"})
    load_in_4bit: bool = field(default=False, metadata={"help": "是否以4位模式加载模型"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "存储从huggingface.co下载的预训练模型的目录"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "要使用的具体模型版本（可以是分支名、标签名或提交ID）"},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "登录Hugging Face Hub的认证令牌"})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用快速分词器（由tokenizers库支持）"},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "覆盖默认的torch.dtype并以该数据类型加载模型。如果传递'auto'，"
                "数据类型将从模型权重自动推导。"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "模型映射到的设备。如果传递'auto'，设备将被自动选择"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "从远程检查点加载模型时是否信任远程代码"},
    )

    def __post_init__(self):
        """参数初始化后的验证"""
        if self.model_name_or_path is None:
            raise ValueError("必须指定有效的model_name_or_path才能运行训练。")


@dataclass
class DataArguments:
    """
    数据相关参数类：定义训练和评估数据的参数
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集名称（通过datasets库）"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集配置名称（通过datasets库）"}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练文本数据文件文件夹"})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "用于评估困惑度的可选输入评估数据文件文件夹"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "为了调试或更快训练，如果设置了此值，将训练样本数量截断到此值。"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "为了调试或更快训练，如果设置了此值，将评估样本数量截断到此值。"
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "启用流模式"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "分词后的可选输入序列长度。"
                "训练数据集将被截断为此大小的块进行训练。"
                "默认为单句输入的模型最大输入长度（考虑特殊标记）。"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "在没有验证分割时，用作验证集的训练集百分比"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数"},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "使用TXT文件时是否保留换行符"}
    )

    def __post_init__(self):
        """参数初始化后的验证"""
        if self.streaming:
            require_version("datasets>=2.0.0", "流模式功能需要datasets>=2.0.0")


def accuracy(predictions, references, normalize=True, sample_weight=None):
    """
    计算准确率
    
    Args:
        predictions: 预测结果
        references: 真实标签
        normalize: 是否归一化
        sample_weight: 样本权重
    
    Returns:
        准确率字典
    """
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }


def compute_metrics(eval_preds):
    """
    计算评估指标
    
    Args:
        eval_preds: 评估预测结果，包含预测和标签
    
    Returns:
        准确率指标
    """
    preds, labels = eval_preds
    # 预测和标签具有相同的形状，经过preprocess_logits_for_metrics计算argmax(-1)后
    # 我们需要对标签进行移位（因为因果语言建模的预测是下一个token）
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    """
    为指标计算预处理logits
    
    Args:
        logits: 模型输出logits
        labels: 标签（未使用）
    
    Returns:
        argmax后的预测结果
    """
    if isinstance(logits, tuple):
        # 训练时：模型可能返回 loss, logits 或 (logits, past_key_values, ...)
        # 推理时：模型通常只返回 logits 或 (logits, past_key_values)
        # 配置影响：当 use_cache=True 时，模型会返回 past_key_values 用于加速生成
        logits = logits[0]
    # logits.shape = (batch_size, sequence_length, vocab_size)
    # logits.argmax(dim=-1)
    # 结果形状：(batch_size, sequence_length)
    # 每个位置的值是预测的下一个token的ID
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    """
    容错数据整理器：将特征列表整理成批次格式

    这个函数用于将单个样本的特征列表转换成适合模型批量处理的张量格式。
    具有容错机制，能够处理多种数据格式和异常情况。

    Args:
        features: 特征列表，每个元素代表一个样本的特征数据

    Returns:
        整理后的批次字典，包含各种张量类型的批次数据
    """
    # 检查第一个特征是否为字典类型，如果不是则转换为字典
    # 这确保了后续处理的一致性
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    # 获取第一个特征作为模板，用于推断批次数据的结构
    first = features[0]
    batch = {}

    # 特殊处理标签数据
    # 标签是监督学习中的关键信息，需要优先处理
    if "label" in first and first["label"] is not None:
        # 从张量中提取数值，如果是张量的话
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        # 根据标签数据类型推断合适的张量数据类型
        # 整数标签通常用于分类任务，浮点标签用于回归任务
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

    # 处理另一种常见的标签格式 label_ids
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            # 如果已经是张量，直接堆叠成批次
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            # 如果是列表或其他格式，先推断数据类型再转换为张量
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # 处理所有其他可能的特征键
    # 遍历第一个样本的所有键值对，构建批次数据
    try:
        for k, v in first.items():
            # 跳过已处理的标签字段和字符串类型字段
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    # 张量类型：直接堆叠所有样本的对应张量
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    # NumPy数组：先堆叠成数组再转换为PyTorch张量
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    # 其他数值类型：直接列表转换为张量
                    batch[k] = torch.tensor([f[k] for f in features])

    except ValueError:  # 容错机制：当正常处理失败时采用快速修复方案
        # 这种情况通常发生在不同样本间特征形状不一致时
        # 通过复制第一个样本的数据来保证批次形状一致
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    # 复制第一个样本的张量，复制批次大小次
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    # 复制第一个样本的NumPy数组
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    # 复制第一个样本的其他数值
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


class SaveModelTrainer(Trainer):
    """
    模型训练器：专门用于保存模型的训练器
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
            _internal_call: 是否内部调用
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """
    保存模型和分词器
    
    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        args: 训练参数
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 处理分布式/并行训练
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """
    为DeepSpeed ZeRO-3保存模型
    参考：https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    
    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        args: 训练参数
        trainer: 训练器实例
    """
    # 创建输出目录，如果目录已存在则不报错
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 将分布在各个GPU节点上的Zero3分区参数整合（consolidate）到一个完整的状态字典中
    # 转换为16位浮点数格式以节省存储空间
    # 返回的是完整的模型参数
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()

    # 获取要保存的模型对象，处理分布式训练时的模型包装情况
    model_to_save = model.module if hasattr(model, "module") else model

    # 保存模型到指定目录，使用Zero3整合后的状态字典
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)

    # 保存tokenizer到输出目录，确保模型的文本处理能力也能被保存
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量
    
    Args:
        model: 要检查的模型
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练参数比例: {100 * trainable_params / all_param}%"
    )


def tokenize_function(tokenizer, examples, block_size):
    """
    分词函数：对文本进行分词并填充到最大长度
    
    Args:
        tokenizer: 分词器
        examples: 包含文本的示例
        block_size: 块大小
    
    Returns:
        分词后的结果，包含input_ids和labels
    """
    # 确保text字段是字符串类型
    texts = examples["text"]
    if not isinstance(texts, list):
        texts = [texts]

    # 过滤掉非字符串的项
    valid_texts = []
    for text in texts:
        if isinstance(text, str):
            valid_texts.append(text)
        elif text is None:
            # 跳过None值
            continue
        else:
            # 尝试转换为字符串
            try:
                valid_texts.append(str(text))
            except Exception:
                # 跳过无法转换的项
                continue

    # 如果没有有效文本，返回空的字典
    if not valid_texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    tokenized_inputs = tokenizer(
        valid_texts,
        truncation=True,
        padding='max_length',
        max_length=block_size
    )
    # 将input_ids复制到labels用于语言建模。这适用于
    # 掩码语言建模（如BERT）或因果语言建模（如GPT）。
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


def tokenize_wo_pad_function(tokenizer, examples):
    """
    不带填充的分词函数：仅进行分词不填充
    
    Args:
        tokenizer: 分词器
        examples: 包含文本的示例
    
    Returns:
        分词后的结果
    """
    # 确保text字段是字符串类型
    texts = examples["text"]
    if not isinstance(texts, list):
        texts = [texts]

    # 过滤掉非字符串的项
    valid_texts = []
    for text in texts:
        if isinstance(text, str):
            valid_texts.append(text)
        elif text is None:
            # 跳过None值
            continue
        else:
            # 尝试转换为字符串
            try:
                valid_texts.append(str(text))
            except Exception:
                # 跳过无法转换的项
                continue

    # 如果没有有效文本，返回空的字典
    if not valid_texts:
        return {"input_ids": [], "attention_mask": []}

    return tokenizer(valid_texts)


def group_text_function(examples, block_size):
    """
    文本分组函数：将连接的文本分割成固定大小的块
    
    Args:
        examples: 分词后的示例
        block_size: 块大小
    
    Returns:
        分组后的结果
    """
    # 检查输入是否为空
    if not examples or not any(examples.values()):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # 连接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # 检查连接后的文本是否为空
    if not concatenated_examples or not concatenated_examples[list(examples.keys())[0]]:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 我们丢弃小的余数，如果模型支持，可以添加填充而不是丢弃，您可以
    # 根据需要自定义这部分。
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按最大长度的块分割
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_local_datasets(data_args, model_args, is_main_process):
    """
    加载本地数据集
    
    Args:
        data_args: 数据参数
        model_args: 模型参数
        is_main_process: 是否为主进程
    
    Returns:
        本地数据集字典，如果不存在则返回空字典
    """
    # 从本地文件加载数据集
    data_files = {}
    dataset_args = {}

    # 加载训练文件
    if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
        train_data_files = glob(f'{data_args.train_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
        if is_main_process:
            logger.info(f"训练文件: {train_data_files}")
        # 训练数据文件必须是相同类型，例如全部txt或全部jsonl
        types = [f.split('.')[-1] for f in train_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"训练文件必须是相同类型，例如全部txt或全部jsonl，但得到 {types}")
        data_files["train"] = train_data_files

    # 加载验证文件
    if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
        eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
        if is_main_process:
            logger.info(f"评估文件: {eval_data_files}")
        data_files["validation"] = eval_data_files
        # 评估数据文件必须是相同类型，例如全部txt或全部jsonl
        types = [f.split('.')[-1] for f in eval_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"评估文件必须是相同类型，例如全部txt或全部jsonl，但得到 {types}")

    # 如果没有找到本地文件，返回空字典
    if not data_files:
        return {}

    # 确定文件扩展名
    extension = "text" if data_files["train"][0].endswith('txt') else 'json'
    if extension == "text":
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

    # 功能: 使用Hugging Face datasets库的load_dataset函数从本地文件加载数据集
    # 支持的文件格式: txt、json、jsonl等
    # 输入样例:
    #   extension = "text"
    #   data_files = {"train": ["data/train.txt", "data/train2.txt"], "validation": ["data/eval.txt"]}
    #   dataset_args = {"keep_linebreaks": True}
    #   cache_dir = "./cache"
    # 输出样例:
    #   raw_datasets = {
    #       "train": Dataset({
    #           features: ['text'],
    #           num_rows: 100000
    #       }),
    #       "validation": Dataset({
    #           features: ['text'],
    #           num_rows: 1000
    #       })
    #   }
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        **dataset_args,
    )

    # 如果没有验证数据，将使用validation_split_percentage来分割数据集
    if "validation" not in raw_datasets.keys():
        # 输入样例:
        #   extension = "text"
        #   data_files = {"train": ["data/train.txt"]}  # 假设 train.txt 有 1000 行
        #   data_args.validation_split_percentage = 10
        #   dataset_args = {"keep_linebreaks": True}
        #   cache_dir = "./cache"
        #   split = "train[:10%]"
        # 输出样例:
        #   raw_datasets["validation"] = Dataset({
        #       features: ['text'],
        #       num_rows: 100  # 前 10% 的数据
        #   })
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )

    return raw_datasets


def load_hub_datasets(data_args, model_args, is_main_process):
    """
    加载多个Hub数据集
    
    Args:
        data_args: 数据参数
        model_args: 模型参数
        is_main_process: 是否为主进程
    
    Returns:
        合并后的Hub数据集字典，如果不存在则返回空字典
    """
    if data_args.dataset_name is None:
        return {}

    # 解析多个数据集名称（以逗号分隔）
    dataset_names = [name.strip() for name in data_args.dataset_name.split(',')]

    # 解析对应的数据集配置名称
    if data_args.dataset_config_name:
        dataset_configs = [
            None if (c := config.strip()) in ("", "None", "none") else c
            for config in data_args.dataset_config_name.split(',')
        ]
    else:
        dataset_configs = [None] * len(dataset_names)

    print(f"dataset_names: {dataset_names}")
    print(f"dataset_config_names: {dataset_configs}")

    # 确保配置名称数量与数据集数量一致
    if len(dataset_configs) != len(dataset_names):
        raise ValueError(f"数据集名称数量({len(dataset_names)})与配置名称数量({len(dataset_configs)})不一致")

    all_train_datasets = []
    all_validation_datasets = []

    for i, dataset_name in enumerate(dataset_names):
        dataset_config = dataset_configs[i]

        if is_main_process:
            config_info = f" (配置: {dataset_config})" if dataset_config else ""
            logger.info(f"正在加载数据集: {dataset_name}{config_info}")

        # 从hub下载并加载数据集
        raw_dataset = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )

        # 处理验证集
        if "validation" in raw_dataset.keys():
            all_train_datasets.append(raw_dataset["train"])
            all_validation_datasets.append(raw_dataset["validation"])
        else:
            # 如果没有验证集，从训练集中分割一部分作为验证集
            train_split = f"train[{data_args.validation_split_percentage}%:]"
            val_split = f"train[:{data_args.validation_split_percentage}%]"

            train_data = load_dataset(
                dataset_name,
                dataset_config,
                split=train_split,
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
            val_data = load_dataset(
                dataset_name,
                dataset_config,
                split=val_split,
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )

            all_train_datasets.append(train_data)
            all_validation_datasets.append(val_data)

    # 合并所有数据集
    if all_train_datasets:
        merged_datasets = {}
        if len(all_train_datasets) == 1:
            merged_datasets["train"] = all_train_datasets[0]
        else:
            merged_datasets["train"] = concatenate_datasets(all_train_datasets)

        if len(all_validation_datasets) == 1:
            merged_datasets["validation"] = all_validation_datasets[0]
        else:
            merged_datasets["validation"] = concatenate_datasets(all_validation_datasets)

        return merged_datasets

    return {}


def load_datasets(data_args, model_args, is_main_process):
    """
    加载数据集 - 融合本地和Hub数据集
    
    Args:
        data_args: 数据参数
        model_args: 模型参数
        is_main_process: 是否为主进程
    
    Returns:
        融合后的原始数据集
    """
    # 加载本地数据集
    local_datasets = load_local_datasets(data_args, model_args, is_main_process)
    # 确保返回的是字典类型
    if local_datasets is None:
        local_datasets = {}

    # 加载Hub数据集
    hub_datasets = load_hub_datasets(data_args, model_args, is_main_process)
    # 确保返回的是字典类型
    if hub_datasets is None:
        hub_datasets = {}

    # 融合数据集
    raw_datasets = {}

    # 处理训练集
    if local_datasets and "train" in local_datasets and hub_datasets and "train" in hub_datasets:
        # 如果两者都有训练集，进行连接
        raw_datasets["train"] = concatenate_datasets([local_datasets["train"], hub_datasets["train"]])
    elif local_datasets and "train" in local_datasets:
        raw_datasets["train"] = local_datasets["train"]
    elif hub_datasets and "train" in hub_datasets:
        raw_datasets["train"] = hub_datasets["train"]

    # 处理验证集
    if local_datasets and "validation" in local_datasets and hub_datasets and "validation" in hub_datasets:
        # 如果两者都有验证集，进行连接
        raw_datasets["validation"] = concatenate_datasets([local_datasets["validation"], hub_datasets["validation"]])
    elif local_datasets and "validation" in local_datasets:
        raw_datasets["validation"] = local_datasets["validation"]
    elif hub_datasets and "validation" in hub_datasets:
        raw_datasets["validation"] = hub_datasets["validation"]

    # 如果没有任何数据集，抛出错误
    if not raw_datasets:
        raise ValueError("既没有找到本地数据文件，也没有指定Hub数据集名称")

    if is_main_process:
        logger.info(f"融合后的数据集: {raw_datasets}")

    return raw_datasets


def load_model_and_tokenizer(model_args):
    """
    加载模型和分词器
    
    Args:
        model_args: 模型参数
    
    Returns:
        模型、分词器和块大小
    """
    # 设置数据类型
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    # 加载分词器
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 设置块大小（序列长度）
    block_size = tokenizer.model_max_length
    if block_size > 2048:
        logger.warning(
            "所选分词器支持的model_max_length长于默认的block_size值2048。"
            "如果您想使用更长的block_size（最多到tokenizer.model_max_length），"
            "您可以通过数据参数调整此值。"
        )

    # 加载模型配置
    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # 设置量化参数
    load_in_4bit = model_args.load_in_4bit
    load_in_8bit = model_args.load_in_8bit
    if load_in_4bit and load_in_8bit:
        raise ValueError("错误，load_in_4bit和load_in_8bit不能同时设置")
    elif load_in_8bit or load_in_4bit:
        logger.info(f"量化模型, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3与量化不兼容。")
        if load_in_8bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )

    # 在Windows+WSL2环境下，避免使用自动设备映射
    # DeepSpeed Zero-3 不兼容 device_map，需要特殊处理
    if is_deepspeed_zero3_enabled():
        device_map = None
    else:
        if torch.cuda.is_available():
            device_map = {"": 0}  # 手动指定使用第一个GPU
        else:
            device_map = "cpu"

    # 加载因果语言模型
    model_kwargs = {
        "config": config,
        "dtype": dtype,
        "low_cpu_mem_usage": (not is_deepspeed_zero3_enabled()),
        **config_kwargs,
    }
    
    # 只有在非 DeepSpeed Zero-3 模式下才添加 device_map
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    return model, tokenizer, block_size


def preprocess_datasets(raw_datasets, data_args, tokenizer, block_size, training_args):
    """
    预处理数据集

    Args:
        raw_datasets: 原始数据集（字典格式），包含'train'和'validation'等分割
        data_args: 数据参数，包含块大小、预处理工作进程数等配置
        tokenizer: 分词器，用于将文本转换为token IDs
        block_size: 块大小，即每个输入序列的最大长度
        training_args: 训练参数，包含是否训练、是否评估、是否按长度分组等配置

    Returns:
        tuple: (train_dataset, eval_dataset, max_train_samples, max_eval_samples)
            - train_dataset: 预处理后的训练数据集
            - eval_dataset: 预处理后的评估数据集
            - max_train_samples: 最大训练样本数
            - max_eval_samples: 最大评估样本数
    """
    # 获取数据集的列名，用于后续移除原始文本列，只保留tokenized后的数据
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # 判断是否为主进程（用于分布式训练，避免重复处理）
    is_main_process = training_args.local_rank in [-1, 0]
    lm_datasets = {}  # 存储预处理后的语言模型数据集

    # 在主进程优先的情况下进行数据集分词和分组
    # 这确保了在分布式训练中，只有主进程执行缓存操作，其他进程等待
    with training_args.main_process_first(desc="数据集分词和分组"):
        if not data_args.streaming:
            # ===== 非流模式处理（适用于数据集可以完全加载到内存的情况） =====
            if training_args.group_by_length:
                # 按长度分组模式：先将文本分词但不填充，再按长度分组
                # 这种方式可以提高训练效率，减少padding浪费
                for split in ["train", "validation"]:
                    if split in raw_datasets:
                        # 第一步：分词但不进行padding
                        tokenized_dataset = raw_datasets[split].map(
                            lambda examples: tokenize_wo_pad_function(tokenizer, examples),
                            batched=True,  # 批量处理提高效率
                            num_proc=data_args.preprocessing_num_workers,  # 多进程并行处理
                            remove_columns=column_names,  # 移除原始列，只保留tokenized数据
                            load_from_cache_file=not data_args.overwrite_cache,  # 是否加载缓存
                            desc=f"在{split}数据集上运行分词器" if is_main_process else None,
                        )
                        # 第二步：将分词后的数据按block_size分组
                        lm_datasets[split] = tokenized_dataset.map(
                            lambda examples: group_text_function(examples, block_size),
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc=f"将{split}数据集分组成{block_size}大小的块",
                        )
            else:
                # 直接分词并填充模式：一步完成分词和padding到block_size
                for split in ["train", "validation"]:
                    if split in raw_datasets:
                        lm_datasets[split] = raw_datasets[split].map(
                            lambda examples: tokenize_function(tokenizer, examples, block_size),
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc=f"在{split}数据集上运行分词器" if is_main_process else None,
                        )
        else:
            # ===== 流模式处理（适用于大型数据集，无法完全加载到内存的情况） =====
            # 流模式不支持多进程处理，因为数据是逐批加载的
            if training_args.group_by_length:
                # 按长度分组模式：先分词不填充，再分组
                for split in ["train", "validation"]:
                    if split in raw_datasets:
                        tokenized_dataset = raw_datasets[split].map(
                            lambda examples: tokenize_wo_pad_function(tokenizer, examples),
                            batched=True,
                            remove_columns=column_names,
                        )
                        lm_datasets[split] = tokenized_dataset.map(
                            lambda examples: group_text_function(examples, block_size),
                            batched=True,
                        )
            else:
                # 直接分词并填充模式
                for split in ["train", "validation"]:
                    if split in raw_datasets:
                        lm_datasets[split] = raw_datasets[split].map(
                            lambda examples: tokenize_function(tokenizer, examples, block_size),
                            batched=True,
                            remove_columns=column_names,
                        )

    # ===== 设置训练数据集 =====
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        # 检查训练数据集是否存在
        if "train" not in lm_datasets:
            raise ValueError("--do_train需要训练数据集")

        train_dataset = lm_datasets['train']
        max_train_samples = len(train_dataset)

        # 如果用户指定了最大训练样本数，进行截断（用于快速测试或调试）
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # 在主进程中打印调试信息
        if is_main_process:
            logger.debug(f"训练样本数量: {len(train_dataset)}")
            logger.debug("分词后的训练示例:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    # ===== 设置评估数据集 =====
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        # 检查评估数据集是否存在
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval需要验证数据集")

        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)

        # 如果用户指定了最大评估样本数，进行截断
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        # 在主进程中打印调试信息
        if is_main_process:
            logger.debug(f"评估样本数量: {len(eval_dataset)}")
            logger.debug("分词后的评估示例:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    return train_dataset, eval_dataset, max_train_samples, max_eval_samples


def setup_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    """
    设置训练器
    
    Args:
        model: 模型
        tokenizer: 分词器
        training_args: 训练参数
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
    
    Returns:
        训练器和是否分布式训练标志
    """
    # 检查分布式训练设置
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # 设置梯度检查点
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()

    # 设置多GPU并行
    if not ddp and torch.cuda.device_count() > 1:
        # 防止Trainer在有多于1个GPU时尝试自己的DataParallelism
        model.is_parallelizable = True
        model.model_parallel = True

    # 创建自定义训练器
    trainer = SaveModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    return trainer, ddp


def train_model(trainer, max_train_samples, training_args):
    """
    训练模型
    
    Args:
        trainer: 训练器对象，负责模型训练的执行和管理
        max_train_samples: 最大训练样本数量，用于限制训练数据的规模
        training_args: 训练参数配置对象，包含学习率、批次大小、训练轮数等配置
    
    Returns:
        dict: 训练结果字典，包含训练过程中的各项指标(如损失、学习率等)和训练样本数量统计
    """
    # 记录训练开始信息
    logger.info("*** 开始增量预训练 ***")
    
    # 获取并记录训练数据加载器中的示例样本，用于数据格式调试
    sample_data = next(iter(trainer.get_train_dataloader()))
    logger.debug(f"训练数据加载器示例: {sample_data}")
    
    # 初始化检查点路径，支持断点续训功能
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"从检查点恢复训练: {checkpoint}")
    
    # 执行模型训练，传入检查点路径以支持断点续训
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 获取训练结果指标
    metrics = train_result.metrics
    
    # 添加训练样本数量到指标中，便于后续分析
    metrics["train_samples"] = max_train_samples
    
    # 将训练指标记录到日志中，便于监控训练过程
    trainer.log_metrics("train", metrics)
    
    # 将训练指标保存到文件中，便于后续分析和可视化
    trainer.save_metrics("train", metrics)
    
    # 保存训练器状态，包括模型权重、优化器状态、学习率调度器状态等
    trainer.save_state()

    # 返回完整的训练结果
    return train_result


def save_trained_model(model, tokenizer, training_args, trainer):
    """
    保存训练后的模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        training_args: 训练参数
        trainer: 训练器
    """
    # 训练后配置
    model.config.use_cache = True  # 训练后启用缓存
    tokenizer.padding_side = "left"  # 恢复填充方向
    tokenizer.init_kwargs["padding_side"] = "left"

    # 检查当前进程是否为主进程（rank 0），避免多进程训练时重复保存
    if trainer.is_world_process_zero():
        # 记录模型保存路径信息
        logger.info(f"保存模型检查点到 {training_args.output_dir}")
        # 根据是否启用DeepSpeed ZeRO优化选择不同的保存策略
        if is_deepspeed_zero3_enabled():
            # 使用DeepSpeed ZeRO-3优化方式保存模型，适用于大规模分布式训练
            save_model_zero3(model, tokenizer, training_args, trainer)
        else:
            # 使用常规方式保存模型
            save_model(model, tokenizer, training_args)


def evaluate_model(trainer, max_eval_samples):
    """
    评估模型
    
    Args:
        trainer: 训练器
        max_eval_samples: 最大评估样本数
    
    Returns:
        评估指标
    """
    logger.info("*** 开始评估 ***")

    # 执行模型评估：在验证集上计算模型性能指标
    # trainer.evaluate()会返回包含各种评估指标的字典，如eval_loss等
    metrics = trainer.evaluate()

    # 添加评估样本数量到指标中，便于后续分析
    metrics["eval_samples"] = max_eval_samples

    # 计算困惑度(Perplexity)：语言模型的核心评估指标
    # 困惑度 = exp(交叉熵损失)，值越小表示模型预测能力越好
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        # 当loss过大时，exp会溢出，此时将困惑度设为无穷大
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    # 记录评估指标到日志系统（如TensorBoard、WandB等）
    trainer.log_metrics("eval", metrics)

    # 保存评估指标到本地文件（通常为output_dir/eval_results.json）
    trainer.save_metrics("eval", metrics)

    # 仅在主进程（rank 0）上打印调试信息，避免多进程重复输出
    if trainer.is_world_process_zero():
        logger.debug(f"评估指标: {metrics}")

    return metrics


def main():
    """主函数：执行增量预训练的完整流程"""

    # 1.参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2.主进程日志输出
    # 移除显式分布式初始化并简化进程检查
    # Trainer将处理分布式训练设置
    is_main_process = training_args.local_rank in [-1, 0]

    # 仅在主进程上记录日志
    if is_main_process:
        logger.info(f"模型参数: {model_args}")
        logger.info(f"数据参数: {data_args}")
        logger.info(f"训练参数: {training_args}")
        logger.info(
            f"进程排名: {training_args.local_rank}, 设备: {training_args.device}, GPU数量: {training_args.n_gpu}"
            + f" 分布式训练: {bool(training_args.local_rank != -1)}, 16位训练: {training_args.fp16}"
        )

    # 初始化模型前设置随机种子
    set_seed(training_args.seed)

    # 3.加载数据集
    raw_datasets = load_datasets(data_args, model_args, is_main_process)

    # 4.加载模型和分词器
    model, tokenizer, block_size = load_model_and_tokenizer(model_args)

    # 调整数据块大小
    if data_args.block_size is not None:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"传递的block_size ({data_args.block_size}) 大于模型的最大长度"
                f"({tokenizer.model_max_length})。使用block_size={tokenizer.model_max_length}。"
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # 5.预处理数据集
    train_dataset, eval_dataset, max_train_samples, max_eval_samples = preprocess_datasets(
        raw_datasets, data_args, tokenizer, block_size, training_args
    )

    # 6.设置训练器
    trainer, ddp = setup_trainer(model, tokenizer, training_args, train_dataset, eval_dataset)

    # 7.训练阶段
    if training_args.do_train:
        train_result = train_model(trainer, max_train_samples, training_args)
        save_trained_model(model, tokenizer, training_args, trainer)
        if trainer.is_world_process_zero():
            logger.debug(f"训练指标: {train_result.metrics}")

    # 8.评估阶段
    if training_args.do_eval:
        evaluate_model(trainer, max_eval_samples)


if __name__ == "__main__":
    main()

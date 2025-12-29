
import os
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional, List

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import DPOTrainer, DPOConfig

from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ModelArguments:
    """
    模型相关参数，包括模型路径、量化配置、设备映射等
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "模型权重初始化的checkpoint路径或名称。"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "用于初始化tokenizer的路径或名称。"}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "是否以8bit模式加载模型以节省显存。"})
    load_in_4bit: bool = field(default=False, metadata={"help": "是否以4bit模式加载模型以进一步节省显存。"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "存放从 HuggingFace 下载的预训练模型的缓存目录。"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用 fast tokenizer（基于tokenizers库）以提高tokenization速度。"},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "覆盖默认的 torch dtype，用于加载模型权重。传入 `auto` 时会自动根据模型权重类型选择。"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "模型映射到的设备，可选 'auto' 自动选择。"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码，当从远程checkpoint加载模型时生效。"},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("必须指定有效的 model_name_or_path 才能运行训练。")


@dataclass
class DatasetArguments:
    """
    数据集相关参数，包括数据源、长度限制、预处理等
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "使用 HuggingFace datasets 库加载数据集的名称。"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集的配置名称。"}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "训练数据的jsonl文件目录。"})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "验证数据的jsonl文件目录。"})
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "Prompt模板名称，如vicuna。"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "每个设备的训练batch大小。"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "每个设备的验证batch大小。"})
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "输入文本最大长度。"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "输出文本最大长度。"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "输出文本最小长度。"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "为了调试或加快训练，限制训练样本数量。"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "为了调试或加快训练，限制验证样本数量。"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖已有缓存的数据集。"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "如果训练集中没有验证集，则按此比例划分验证集。"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "数据预处理使用的进程数量。"},
    )


@dataclass
class TrainingArguments:
    """
    训练相关参数，包括优化器配置、LoRA设置、训练策略等
    """
    use_peft: bool = field(default=True, metadata={"help": "是否使用PEFT（参数高效微调）。"})
    qlora: bool = field(default=False, metadata={"help": "是否使用QLoRA量化微调。"})
    target_modules: Optional[str] = field(default=None, metadata={"help": "LoRA微调目标模块名称。"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "LoRA矩阵的秩。"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA的dropout概率。"})
    lora_alpha: Optional[float] = field(default=16.0, metadata={"help": "LoRA的缩放系数alpha。"})
    peft_path: Optional[str] = field(default=None, metadata={"help": "PEFT模型路径，可加载已有微调模型。"})
    do_train: bool = field(default=False, metadata={"help": "是否执行训练过程。"})
    do_eval: bool = field(default=False, metadata={"help": "是否在验证集上执行评估。"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "学习率。"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "学习率调度类型，如cosine。"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "预热步数。"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "权重衰减系数。"})
    adam_beta1: Optional[float] = field(default=0.9, metadata={"help": "Adam优化器的beta1参数。"})
    adam_beta2: Optional[float] = field(default=0.95, metadata={"help": "Adam优化器的beta2参数。"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "优化器类型。"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "是否使用FP16训练。"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "是否使用BF16训练。"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "是否启用梯度检查点以节省显存。"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "梯度累积步数，相当于增大batch size。"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "每隔多少步保存一次模型。"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "每隔多少步进行一次评估。"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "每隔多少步记录一次日志。"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "模型输出保存目录。"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "训练总步数。"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "评估策略，如按步数或按epoch。"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "如果使用datasets.Dataset，是否移除未使用的列。"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "日志上报平台，如wandb或tensorboard。"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "DeepSpeed配置文件路径。"})
    local_rank: int = field(default=-1, metadata={"help": "本地进程排名，用于分布式训练。"})


@dataclass
class ScriptArguments:
    """
    脚本主参数类，组合所有参数类别
    
    参数分为几个主要类别：
    1. 模型相关参数（Model arguments）
    2. 数据集相关参数（Dataset arguments）  
    3. 训练相关参数（Training arguments）
    """
    model_args: ModelArguments = field(default_factory=ModelArguments)
    dataset_args: DatasetArguments = field(default_factory=DatasetArguments)
    training_args: TrainingArguments = field(default_factory=TrainingArguments)

    # 为了保持向后兼容性，提供属性访问
    @property
    def model_name_or_path(self):
        return self.model_args.model_name_or_path
    
    @property
    def tokenizer_name_or_path(self):
        return self.model_args.tokenizer_name_or_path
    
    @property
    def load_in_8bit(self):
        return self.model_args.load_in_8bit
    
    @property
    def load_in_4bit(self):
        return self.model_args.load_in_4bit
    
    @property
    def cache_dir(self):
        return self.model_args.cache_dir
    
    @property
    def use_fast_tokenizer(self):
        return self.model_args.use_fast_tokenizer
    
    @property
    def dtype(self):
        return self.model_args.dtype
    
    @property
    def device_map(self):
        return self.model_args.device_map
    
    @property
    def trust_remote_code(self):
        return self.model_args.trust_remote_code
    
    @property
    def dataset_name(self):
        return self.dataset_args.dataset_name
    
    @property
    def dataset_config_name(self):
        return self.dataset_args.dataset_config_name
    
    @property
    def train_file_dir(self):
        return self.dataset_args.train_file_dir
    
    @property
    def validation_file_dir(self):
        return self.dataset_args.validation_file_dir
    
    @property
    def template_name(self):
        return self.dataset_args.template_name
    
    @property
    def per_device_train_batch_size(self):
        return self.dataset_args.per_device_train_batch_size
    
    @property
    def per_device_eval_batch_size(self):
        return self.dataset_args.per_device_eval_batch_size
    
    @property
    def max_source_length(self):
        return self.dataset_args.max_source_length
    
    @property
    def max_target_length(self):
        return self.dataset_args.max_target_length
    
    @property
    def min_target_length(self):
        return self.dataset_args.min_target_length
    
    @property
    def max_train_samples(self):
        return self.dataset_args.max_train_samples
    
    @property
    def max_eval_samples(self):
        return self.dataset_args.max_eval_samples
    
    @property
    def overwrite_cache(self):
        return self.dataset_args.overwrite_cache
    
    @property
    def validation_split_percentage(self):
        return self.dataset_args.validation_split_percentage
    
    @property
    def preprocessing_num_workers(self):
        return self.dataset_args.preprocessing_num_workers
    
    @property
    def use_peft(self):
        return self.training_args.use_peft
    
    @property
    def qlora(self):
        return self.training_args.qlora
    
    @property
    def target_modules(self):
        return self.training_args.target_modules
    
    @property
    def lora_rank(self):
        return self.training_args.lora_rank
    
    @property
    def lora_dropout(self):
        return self.training_args.lora_dropout
    
    @property
    def lora_alpha(self):
        return self.training_args.lora_alpha
    
    @property
    def peft_path(self):
        return self.training_args.peft_path
    
    @property
    def do_train(self):
        return self.training_args.do_train
    
    @property
    def do_eval(self):
        return self.training_args.do_eval
    
    @property
    def learning_rate(self):
        return self.training_args.learning_rate
    
    @property
    def lr_scheduler_type(self):
        return self.training_args.lr_scheduler_type
    
    @property
    def warmup_steps(self):
        return self.training_args.warmup_steps
    
    @property
    def weight_decay(self):
        return self.training_args.weight_decay
    
    @property
    def adam_beta1(self):
        return self.training_args.adam_beta1
    
    @property
    def adam_beta2(self):
        return self.training_args.adam_beta2
    
    @property
    def optim(self):
        return self.training_args.optim
    
    @property
    def fp16(self):
        return self.training_args.fp16
    
    @property
    def bf16(self):
        return self.training_args.bf16
    
    @property
    def gradient_checkpointing(self):
        return self.training_args.gradient_checkpointing
    
    @property
    def gradient_accumulation_steps(self):
        return self.training_args.gradient_accumulation_steps
    
    @property
    def save_steps(self):
        return self.training_args.save_steps
    
    @property
    def eval_steps(self):
        return self.training_args.eval_steps
    
    @property
    def logging_steps(self):
        return self.training_args.logging_steps
    
    @property
    def output_dir(self):
        return self.training_args.output_dir
    
    @property
    def max_steps(self):
        return self.training_args.max_steps
    
    @property
    def eval_strategy(self):
        return self.training_args.eval_strategy
    
    @property
    def remove_unused_columns(self):
        return self.training_args.remove_unused_columns
    
    @property
    def report_to(self):
        return self.training_args.report_to
    
    @property
    def deepspeed(self):
        return self.training_args.deepspeed
    
    @property
    def local_rank(self):
        return self.training_args.local_rank


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


# =========================================================
# 参数解析
# =========================================================

def parse_args():
    """
    解析命令行参数

    使用 HfArgumentParser 将分离的参数类
    映射为命令行参数，组合成ScriptArguments对象。

    Returns:
        args (ScriptArguments): 解析后的训练参数
    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()
    
    # 组合所有参数到主参数类
    args = ScriptArguments(
        model_args=model_args,
        dataset_args=dataset_args,
        training_args=training_args
    )
    logger.info(f"Parse args: {args}")
    return args


# =========================================================
# Tokenizer & Prompt Template
# =========================================================

def load_tokenizer_and_template(args):
    """
    加载 tokenizer 与对话 prompt 模板，并补齐 DPO/SFT 所需的特殊 token

    主要处理：
    1. eos_token：DPO 训练必须存在
    2. bos_token：LLaMA / Qwen 等模型可能需要
    3. pad_token：DataCollator / DPOTrainer 需要

    Args:
        args: 训练参数

    Returns:
        tokenizer (PreTrainedTokenizer)
        prompt_template: 对话模板对象
    """
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # 加载对话模板（FastChat / 自定义）
    prompt_template = get_conv_template(args.template_name)

    # ------------------------------
    # eos_token：DPO / SFT 强依赖
    # ------------------------------
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}")

    # ------------------------------
    # bos_token：LLaMA 系模型常见需求
    # ------------------------------
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # ------------------------------
    # pad_token：batch padding 必须
    # ------------------------------
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    return tokenizer, prompt_template

# =========================================================
# 工具函数
# =========================================================

def parse_comma_list(value: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔字符串为 list"""
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def merge_dataset_dicts(datasets: List[DatasetDict]) -> DatasetDict:
    """
    将多个 DatasetDict 按 split（数据集划分）合并为一个 DatasetDict。

    该函数遍历所有输入的 DatasetDict，收集其中所有的 split 名称（如 'train', 'validation', 'test'），
    然后对每个 split，将所有 DatasetDict 中该 split 对应的数据集使用 concatenate_datasets 进行合并。

    Args:
        datasets (List[DatasetDict]): 要合并的 DatasetDict 列表。
            每个 DatasetDict 可能包含不同的 split，如 {'train': dataset1, 'validation': dataset2}

    Returns:
        DatasetDict: 合并后的 DatasetDict，包含所有输入 DatasetDict 的所有 split。
            每个 split 的数据是对应输入 DatasetDict 中该 split 数据的连接。

    Example:
        >>> from datasets import Dataset, DatasetDict
        >>>
        >>> # 创建第一个 DatasetDict
        >>> ds1_data = {"text": ["hello", "world"]}
        >>> ds1 = Dataset.from_dict(ds1_data)
        >>> dict1 = DatasetDict({"train": ds1, "validation": ds1})
        >>>
        >>> # 创建第二个 DatasetDict
        >>> ds2_data = {"text": ["foo", "bar"]}
        >>> ds2 = Dataset.from_dict(ds2_data)
        >>> dict2 = DatasetDict({"train": ds2, "test": ds2})
        >>>
        >>> # 合并两个 DatasetDict
        >>> result = merge_dataset_dicts([dict1, dict2])
        >>>
        >>> # 结果包含所有 split: train, validation, test
        >>> # train split 包含 4 个样本: ["hello", "world", "foo", "bar"]
        >>> # validation split 包含 2 个样本: ["hello", "world"]
        >>> # test split 包含 2 个样本: ["foo", "bar"]
        >>> print(list(result.keys()))
        ['train', 'validation', 'test']
        >>> print(len(result['train']))
        4

    Input Example:
        [
            DatasetDict({
                'train': Dataset(samples: 1000),
                'validation': Dataset(samples: 200)
            }),
            DatasetDict({
                'train': Dataset(samples: 800),
                'test': Dataset(samples: 150)
            }),
            DatasetDict({
                'validation': Dataset(samples: 300),
                'test': Dataset(samples: 250)
            })
        ]

    Output Example:
        DatasetDict({
            'train': Dataset(samples: 1800),        # 1000 + 800
            'validation': Dataset(samples: 500),     # 200 + 300
            'test': Dataset(samples: 400)           # 150 + 250
        })
    """
    # 创建空的 DatasetDict 用于存放合并结果
    merged = DatasetDict()

    # 获取所有 DatasetDict 中的所有 split 名称的并集
    # 例如：[{'train', 'validation'}, {'train', 'test'}] -> {'train', 'validation', 'test'}
    all_splits = set().union(*[ds.keys() for ds in datasets])

    # 遍历每个 split 名称
    for split in all_splits:
        # 收集所有包含该 split 的 DatasetDict 中对应的数据集
        split_datasets = [
            ds[split] for ds in datasets if split in ds
        ]

        # 如果存在该 split 的数据集，则将它们连接起来
        if split_datasets:
            merged[split] = concatenate_datasets(split_datasets)

    return merged


# =========================================================
# HF Hub 数据加载
# =========================================================

def load_from_hf_hub(args) -> Optional[DatasetDict]:
    """
    从 HuggingFace Hub 加载一个或多个数据集
    若未配置或加载失败，返回 None
    """
    dataset_names = parse_comma_list(args.dataset_name)
    
    # 如果没有指定数据集名称，直接返回 None
    if not dataset_names:
        return None
        
    # 处理 dataset_config_name 可能为 None 的情况
    if args.dataset_config_name is None:
        dataset_configs = [None] * len(dataset_names)
    else:
        dataset_configs = [
            None if (c := config.strip()) in ("", "None", "none") else c
            for config in args.dataset_config_name.split(',')
        ]

    if len(dataset_names) != len(dataset_configs):
        raise ValueError(
            "dataset_name 与 dataset_config_name 数量不一致"
        )

    loaded = []

    for name, config in zip(dataset_names, dataset_configs):
        logger.info(f"📥 Loading HF dataset: {name}, config={config}")
        ds = load_dataset(
            name,
            config,
            cache_dir=args.cache_dir,
        )
        loaded.append(ds)

    return merge_dataset_dicts(loaded)


# =========================================================
# 本地文件数据加载
# =========================================================

def load_from_local_files(args) -> Optional[DatasetDict]:
    """
    从本地 JSON / JSONL 文件加载数据集（支持递归扫描）
    若未找到文件，返回 None
    """
    data_files = {}

    if args.train_file_dir and os.path.exists(args.train_file_dir):
        train_files = glob(
            f"{args.train_file_dir}/**/*.json*", recursive=True
        )
        if train_files:
            data_files["train"] = train_files

    if args.validation_file_dir and os.path.exists(args.validation_file_dir):
        val_files = glob(
            f"{args.validation_file_dir}/**/*.json*", recursive=True
        )
        if val_files:
            data_files["validation"] = val_files

    if not data_files:
        return None

    logger.info(f"📂 Loading local files: {data_files}")

    return load_dataset(
        "json",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )


# =========================================================
# 原始数据集统一入口
# =========================================================

def load_raw_datasets(args) -> DatasetDict:
    """
    加载原始数据集，支持：
    1. HuggingFace Hub
    2. 本地 JSON / JSONL 文件

    特性：
    - 两类数据源可同时使用
    - 任一类有数据即可
    - 自动切 validation
    """
    datasets = []

    # ------------------------------
    # HF Hub
    # ------------------------------
    hf_datasets = load_from_hf_hub(args)
    if hf_datasets is not None:
        datasets.append(hf_datasets)

    # ------------------------------
    # Local files
    # ------------------------------
    local_datasets = load_from_local_files(args)
    if local_datasets is not None:
        datasets.append(local_datasets)

    # ------------------------------
    # 校验
    # ------------------------------
    if not datasets:
        raise ValueError(
            "未加载到任何数据集（HF Hub 与本地文件均为空）"
        )

    raw_datasets = merge_dataset_dicts(datasets)

    # ------------------------------
    # 自动切 validation
    # ------------------------------
    if "validation" not in raw_datasets:
        split_pct = args.validation_split_percentage
        train_len = len(raw_datasets["train"])

        if train_len == 0:
            raise ValueError("训练集为空，无法切分 validation")

        split_idx = int(train_len * split_pct / 100)

        logger.info(
            f"✂️ Auto split validation: {split_pct}% "
            f"({split_idx}/{train_len})"
        )

        raw_datasets["validation"] = raw_datasets["train"].select(
            range(split_idx)
        )
        raw_datasets["train"] = raw_datasets["train"].select(
            range(split_idx, train_len)
        )

    logger.info(f"✅ Raw datasets loaded: {raw_datasets}")
    return raw_datasets




# =========================================================
# DPO 数据集构建（核心）
# =========================================================

def build_dpo_datasets(args, raw_datasets, prompt_template):
    """
    将原始偏好数据构造成 DPOTrainer 所需格式：

    输出字段：
    - prompt
    - chosen
    - rejected

    并进行：
    - prompt 拼接
    - 长度过滤
    - train / eval 分别处理

    Args:
        args: 训练参数
        raw_datasets: 原始数据
        prompt_template: 对话模板

    Returns:
        train_dataset, eval_dataset
    """
    max_length = args.max_source_length + args.max_target_length

    def build_prompt_and_responses(examples) -> Dict[str, str]:
        """
        将 system + history + question 拼接成最终 prompt
        """
        prompts = []
        for system, history, question in zip(
            examples["system"],
            examples["history"],
            examples["question"],
        ):
            system_prompt = system or ""
            history = history or []
            history_with_question = history + [[question, ""]]

            prompts.append(
                prompt_template.get_prompt(
                    messages=history_with_question,
                    system_prompt=system_prompt,
                )
            )

        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    def process_split(split_name, max_samples):
        """
        处理单个数据集 split（train / validation）
        """
        ds = raw_datasets[split_name]

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        ds = ds.map(
            build_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=ds.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Processing {split_name} dataset",
        )

        # 长度过滤（避免 OOM）
        ds = ds.filter(
            lambda x: 0 < len(x["prompt"] + x["chosen"]) <= max_length
            and 0 < len(x["prompt"] + x["rejected"]) <= max_length
        )
        return ds

    train_dataset = (
        process_split("train", args.max_train_samples)
        if args.do_train
        else None
    )

    eval_dataset = (
        process_split("validation", args.max_eval_samples)
        if args.do_eval
        else None
    )

    return train_dataset, eval_dataset


# =========================================================
# 模型加载（DDP / QLoRA / 梯度检查点）
# =========================================================

def load_model(args):
    """
    加载模型并处理：
    - DDP device_map
    - DeepSpeed集成
    - QLoRA / 4bit / 8bit
    - FP16 梯度稳定性问题
    - Gradient Checkpointing

    Args:
        args: 训练参数

    Returns:
        model (AutoModelForCausalLM)
    """
    # DeepSpeed集成
    if args.deepspeed is not None:
        # DeepSpeed模式下不设置device_map，让DeepSpeed处理
        device_map = None
        logger.info("使用DeepSpeed分布式训练，device_map设为None")
    else:
        # 单卡或多卡DDP模式
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        else:
            device_map = args.device_map

    dtype = (
        args.dtype
        if args.dtype in ["auto", None]
        else getattr(torch, args.dtype)
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        cache_dir=args.cache_dir,
    )

    # QLoRA 量化配置
    quant_config = None
    if args.qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )

    # ------------------------------
    # 修复 DPO 中常见的 FP16 梯度异常
    # ------------------------------
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # ------------------------------
    # Gradient Checkpointing
    # ------------------------------
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    return model


# =========================================================
# DPOTrainer 构建
# =========================================================

def build_dpo_trainer(args, model, tokenizer, train_dataset, eval_dataset):
    """
    构建 DPOTrainer，包括：
    - DPOConfig
    - 可选 LoRA / PEFT
    - reference model 处理
    - DeepSpeed集成

    Args:
        args: 训练参数
        model: 主模型
        tokenizer: tokenizer
        train_dataset / eval_dataset

    Returns:
        trainer (DPOTrainer)
    """
    # DeepSpeed集成
    deepspeed_config = None
    if args.deepspeed is not None:
        import json
        with open(args.deepspeed, 'r') as f:
            deepspeed_config = json.load(f)
        logger.info(f"加载DeepSpeed配置: {args.deepspeed}")

    training_args = DPOConfig(
        # =========================
        # 序列长度相关
        # =========================
        # prompt（用户输入 / 指令部分）的最大长度
        # 超过长度会被截断，直接影响：
        # - 上下文保留完整度
        # - 显存占用
        # - DPO 中 preference 对齐效果
        max_prompt_length=args.max_source_length,

        # 单条样本的总最大长度 = prompt + response
        # 对 DPO 来说，chosen / rejected 都会被 pad / truncate 到这个长度
        # 设置过小 → 信息丢失
        # 设置过大 → 显存 & 计算量爆炸
        max_length=args.max_source_length + args.max_target_length,


        # =========================
        # Batch 相关
        # =========================
        # 单卡训练 batch size
        # 实际 effective batch size =
        #   per_device_train_batch_size × gradient_accumulation_steps × GPU 数
        per_device_train_batch_size=args.per_device_train_batch_size,

        # 单卡评估 batch size
        # eval 不反传，一般可以比 train 大一点
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        # 梯度累积步数
        # 用于在显存受限时模拟大 batch
        # 对 DPO 来说，batch 稳定性对 loss 很关键
        gradient_accumulation_steps=args.gradient_accumulation_steps,


        # =========================
        # 优化器 & 学习率
        # =========================
        # 基础学习率
        # DPO 通常比 SFT 小（例如 1e-6 ~ 5e-5）
        # 太大容易 preference 崩
        learning_rate=args.learning_rate,

        # 学习率 warmup 步数
        # 防止一开始 loss 爆炸，DPO 尤其推荐开
        warmup_steps=args.warmup_steps,

        # 使用的优化器类型
        # 常见：adamw_torch / adamw_bnb_8bit（QLoRA）
        optim=args.optim,

        # 权重衰减系数
        # 必须与DeepSpeed配置保持一致
        weight_decay=args.weight_decay,

        # Adam优化器的beta参数
        # 必须与DeepSpeed配置保持一致 [0.9, 0.95]
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,


        # =========================
        # 日志 & checkpoint
        # =========================
        # 每隔多少 step 打一次日志（loss / lr 等）
        logging_steps=args.logging_steps,

        # 每隔多少 step 保存一次 checkpoint
        # DPO 训练建议不要太频繁（磁盘 + IO 压力大）
        save_steps=args.save_steps,


        # =========================
        # 评估相关
        # =========================
        # 评估策略：
        # - "steps"：按步数评估
        # - "epoch"：每个 epoch 评估一次
        # - "no"：不评估
        eval_strategy=args.eval_strategy,

        # 当 eval_strategy="steps" 时生效
        # 指定多少 step 进行一次评估
        eval_steps=args.eval_steps,


        # =========================
        # 精度相关
        # =========================
        # 是否使用 bfloat16
        # Ampere+ GPU 推荐，数值稳定性优于 fp16
        bf16=args.bf16,

        # 是否使用 fp16
        # 与 bf16 二选一，老 GPU 或不支持 bf16 时使用
        fp16=args.fp16,


        # =========================
        # 输出 & 运行信息
        # =========================
        # 模型、checkpoint、trainer 状态的保存目录
        output_dir=args.output_dir,

        # 本次训练 run 的名称
        # 用于日志系统 / 实验追踪（即使 report_to=None 也有用）
        run_name="dpo_v1",

        # 是否移除 dataset 中模型 forward 用不到的字段
        # 对自定义数据结构很重要：
        # - True：省内存，但字段名必须完全匹配
        # - False：更安全，推荐复杂 DPO 数据用 False
        remove_unused_columns=args.remove_unused_columns,

        # 禁用 wandb / swanlab / tensorboard 等自动上报
        # 避免在内网或无权限环境中出现超时或阻塞
        report_to=None,

        # =========================
        # 分布式训练相关
        # =========================
        # 本地进程排名，用于分布式训练
        local_rank=args.local_rank,

        # DeepSpeed配置文件路径
        deepspeed=args.deepspeed,

        # 是否在分布式训练中仅保存主节点模型
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )

    peft_config = None
    if args.use_peft:
        # 处理 target_modules
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            inference_mode=False,
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=None if args.use_peft else deepcopy(model),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print_trainable_parameters(trainer.model)
    return trainer


# =========================================================
# 主入口
# =========================================================

def main():
    """
    DPO 训练主入口（重构后）

    执行流程：
    1. 参数解析
    2. tokenizer & prompt 模板加载
    3. 原始数据集加载
    4. DPO 数据集构建
    5. 模型加载
    6. DPOTrainer 构建
    7. 训练 / 评估 / 保存
    """
    args = parse_args()

    tokenizer, prompt_template = load_tokenizer_and_template(args)
    raw_datasets = load_raw_datasets(args)

    train_dataset, eval_dataset = build_dpo_datasets(
        args, raw_datasets, prompt_template
    )

    model = load_model(args)

    trainer = build_dpo_trainer(
        args,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()

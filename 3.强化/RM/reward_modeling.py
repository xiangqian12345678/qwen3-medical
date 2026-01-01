# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Any, List, Union, Optional, Dict

import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME

from template import get_conv_template


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    用于定义与模型训练和评估相关的数据输入参数。
    该类主要用于管理数据集名称、配置、路径以及预处理等参数。
    """
    # 用于指定 HuggingFace datasets 库中的数据集名称列表，例如 "glue,squad"等。
    # 如果提供了此字段，将直接从 HuggingFace Hub 下载对应数据集。
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "指定要使用的数据集名称（通过 datasets 库加载）。"}
    )

    # 一些数据集有多个配置，例如不同子任务或不同版本的数据。
    # 通过此字段选择具体配置，例如 "sst2,v1.1,none,None" 等。
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "指定数据集的配置名称（通过 datasets 库加载）。"}
    )

    # 如果使用本地数据而不是 HuggingFace 数据集，可以通过此路径加载训练文件。
    train_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "训练数据的文件夹路径（jsonl 格式）"}
    )

    # 指定本地验证数据路径。如果未提供，将会使用训练集的一部分作为验证集。
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据的文件夹路径（jsonl 格式）"}
    )

    # 模型输入序列的最大 token 数，超过此长度将被截断。
    max_source_length: Optional[int] = field(
        default=2048,
        metadata={"help": "输入文本（prompt）的最大长度"}
    )

    # 模型输出序列的最大 token 数，超过此长度将被截断。
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "输出文本（target）的最大长度"}
    )

    # 可用于调试或快速实验，限制训练集样本数量，避免全量训练。
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于调试或快速训练，可限制训练样本数量。"
                "如果设置，将只使用前 N 条训练数据。"
            )
        },
    )

    # 同上，但作用于验证/评估集。
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于调试或快速验证，可限制评估样本数量。"
                "如果设置，将只使用前 N 条验证数据。"
            )
        },
    )

    # 如果为 True，将重新处理数据集并覆盖缓存，适用于数据更新或预处理修改。
    # HuggingFace datasets 库在加载和处理数据集时会缓存处理后的数据到本地（通常在 ~/.cache/huggingface/datasets）。
    # overwrite_cache=True 表示忽略缓存，重新处理数据集（包括读取原始数据、tokenization、格式化等）。
    # overwrite_cache=False（默认值）表示优先使用缓存，加快加载速度。
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "是否覆盖已缓存的训练和验证数据集"}
    )

    # 如果没有单独的验证集，将使用训练集的一部分进行验证，默认 1%。
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "当数据集没有独立验证集时，"
                "从训练集划分一定比例作为验证集，默认1%。"
            )
        },
    )

    # 数据预处理（如 tokenization）时并行处理的进程数量，提升效率。
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "数据预处理时使用的进程数"}
    )

    # # 指定数据集缓存目录路径，用于存储下载和处理后的数据集。
    # # 如果不指定，将使用默认的缓存路径（通常是 ~/.cache/huggingface/datasets）。
    # cache_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "数据集缓存目录路径"}
    # )


@dataclass
class ScriptArguments:
    use_peft: bool = field(
        default=True,
        metadata={
            "help": "是否使用 PEFT（Parameter-Efficient Fine-Tuning）方法进行微调。如果为 True，将使用如 LoRA 的轻量微调策略。"}
    )
    target_modules: Optional[str] = field(
        default="all",
        metadata={
            "help": "指定模型中哪些模块应用 LoRA 等 PEFT 方法。默认 'all' 表示对所有模块生效，也可以指定具体模块名列表。"}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA 的秩（rank）参数，控制低秩分解的维度。数值越大，微调能力越强，但显存占用也越高。"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "LoRA 的 Dropout 概率，用于在微调时防止过拟合。值越大正则化效果越强。"}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "LoRA 的缩放系数（alpha），用于调整低秩更新的幅度。通常与 lora_rank 配合使用。"}
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "指定在保存模型时只保存哪些模块，避免保存整个大模型。可用于 PEFT 模型只保存 LoRA 权重。"}
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "如果已有预训练的 PEFT 模型，可指定其路径进行加载微调。"}
    )
    template_name: Optional[str] = field(
        default="vicuna",
        metadata={"help": "使用的 prompt 模板名称，例如 'vicuna' 或 'alpaca'。用于生成训练或推理时的输入格式。"}
    )


def compute_metrics(eval_preds):
    """
    计算模型在评估数据上的回归指标，包括均方误差(MSE)和平均绝对误差(MAE)。

    参数:
        eval_preds: tuple
            包含预测值和真实标签的元组 (preds, labels)
            - preds: 模型的预测输出，可能是 torch.Tensor 或 numpy.ndarray
            - labels: 对应的真实标签，可能是 torch.Tensor 或 numpy.ndarray

    返回:
        dict: 包含两个指标的字典
            - "mse": 均方误差 (Mean Squared Error)
            - "mae": 平均绝对误差 (Mean Absolute Error)
    """

    preds, labels = eval_preds  # 解包预测值和真实标签

    # 如果预测值是 torch.Tensor，则转为 numpy 数组，便于计算指标
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    # 如果真实标签是 torch.Tensor，也转为 numpy 数组
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 计算均方误差（MSE），衡量预测值与真实值的平方差平均值
    mse = mean_squared_error(labels, preds)

    # 计算平均绝对误差（MAE），衡量预测值与真实值的绝对差平均值
    mae = mean_absolute_error(labels, preds)

    # 返回指标字典
    return {"mse": mse, "mae": mae}


@dataclass
class RewardDataCollatorWithPadding:
    """We need to define a special data collator that batches the data in our chosen vs rejected format"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        # 对已有的 token 序列进行 padding（填充）和对齐，并生成 tensor
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # 对已有的 token 序列进行 padding（填充）和对齐，并生成 tensor
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        # 计算损失：InstructGPT中的pairwise logloss
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prepare inputs for chosen and rejected separately
        device = model.device

        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }
        outputs_chosen = model(**inputs_chosen)
        rewards_chosen = outputs_chosen.logits.detach()

        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }
        outputs_rejected = model(**inputs_rejected)
        rewards_rejected = outputs_rejected.logits.detach()

        # Keep the compute_loss method
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


class CastOutputToFloat(torch.nn.Sequential):
    """Cast the output of the model to float"""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


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
            if 'score' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def load_datasets_from_hub(args, model_args):
    """Load datasets from huggingface hub"""
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    return raw_datasets


def load_datasets_from_files(args, model_args):
    """Load datasets from local files"""
    data_files = {}
    if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
        train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
            f'{args.train_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"train files: {', '.join(train_data_files)}")
        data_files["train"] = train_data_files

    if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
        eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
            f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"eval files: {', '.join(eval_data_files)}")
        data_files["validation"] = eval_data_files

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    return raw_datasets


def process_hub_datasets(args, model_args):
    """处理来自HuggingFace Hub的多个数据集（逗号分隔）"""
    if args.dataset_name is None:
        return None

    dataset_names = [name.strip() for name in args.dataset_name.split(",")]
    dataset_configs = [
        None if (c := config.strip()) in ("", "None", "none") else c
        for config in args.dataset_config_name.split(',')
    ]

    if len(dataset_configs) < len(dataset_names):
        # 如果配置数量少于数据集数量，缺省为 None
        dataset_configs += [None] * (len(dataset_names) - len(dataset_configs))

    combined_datasets = {"train": [], "validation": []}

    try:
        for name, config in zip(dataset_names, dataset_configs):
            raw_datasets = load_dataset(
                name,
                config,
                cache_dir=model_args.cache_dir,
            )

            # 如果没有验证集，从训练集中划分
            if "validation" not in raw_datasets.keys():
                val_split = f"train[:{args.validation_split_percentage}%]"
                train_split = f"train[{args.validation_split_percentage}%:]"
                raw_datasets["validation"] = load_dataset(
                    name, config, split=val_split, cache_dir=model_args.cache_dir
                )
                raw_datasets["train"] = load_dataset(
                    name, config, split=train_split, cache_dir=model_args.cache_dir
                )

            logger.info(f"Loaded dataset '{name}' with splits: {list(raw_datasets.keys())}")
            for split_name in ["train", "validation"]:
                combined_datasets[split_name].extend(raw_datasets[split_name])

        logger.info(
            f"Combined datasets: train={len(combined_datasets['train'])} samples, "
            f"validation={len(combined_datasets['validation'])} samples"
        )
        return combined_datasets

    except Exception as e:
        logger.error(f"Failed to load datasets from hub '{args.dataset_name}': {e}")
        raise ValueError(f"Cannot load datasets from hub: {e}")


def process_file_datasets(args, model_args):
    """处理来自本地文件的数据集"""
    if args.train_file_dir is None and args.validation_file_dir is None:
        return None

    try:
        data_files = {}

        # 处理训练文件
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            if not train_data_files:
                raise ValueError(f"No JSON or JSONL files found in {args.train_file_dir}")

            logger.info(f"Found train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files

        # 处理验证文件
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            if not eval_data_files:
                raise ValueError(f"No JSON or JSONL files found in {args.validation_file_dir}")

            logger.info(f"Found eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files

        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

        # 如果没有验证集，从训练集中划分
        if "validation" not in raw_datasets.keys() and "train" in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

        logger.info(f"Successfully loaded datasets from files: {list(raw_datasets.keys())}")
        for split_name, split_data in raw_datasets.items():
            logger.info(f"  File {split_name}: {len(split_data)} samples")

        return raw_datasets
    except Exception as e:
        logger.error(f"Failed to load datasets from files: {e}")
        raise ValueError(f"Cannot load datasets from files: {e}")


def merge_datasets(hub_datasets, file_datasets):
    """合并来自不同来源的数据集"""
    if hub_datasets is None and file_datasets is None:
        raise ValueError("No valid data source found. Both hub and file datasets are None.")

    if hub_datasets is None:
        return file_datasets

    if file_datasets is None:
        return hub_datasets

    # 合并两个数据源
    merged_datasets = hub_datasets.copy()

    for split_name, split_data in file_datasets.items():
        if split_name in merged_datasets:
            # 连接相同split的数据集
            merged_datasets[split_name] = concatenate_datasets([merged_datasets[split_name], split_data])
            logger.info(f"Merged {split_name} datasets. Hub: {len(hub_datasets[split_name])}, "
                        f"Files: {len(split_data)}, Total: {len(merged_datasets[split_name])} samples")
        else:
            # 添加新的split
            merged_datasets[split_name] = split_data
            logger.info(f"Added {split_name} split from files. Size: {len(split_data)} samples")

    return merged_datasets


def load_raw_datasets(args, model_args):
    """加载并合并来自不同来源的原始数据集"""
    # 分别处理两种数据源
    hub_datasets = process_hub_datasets(args, model_args)
    file_datasets = process_file_datasets(args, model_args)

    # 合并数据集
    raw_datasets = merge_datasets(hub_datasets, file_datasets)

    logger.info(f"Final merged datasets: {list(raw_datasets.keys())}")
    for split_name, split_data in raw_datasets.items():
        logger.info(f"  Final {split_name}: {len(split_data)} samples")

    return raw_datasets


def setup_logging_and_seed(training_args):
    """设置日志和随机种子"""
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    set_seed(training_args.seed)


def load_model_and_tokenizer(model_args, script_args):
    """加载模型和分词器"""
    # Load model
    if not model_args.model_name_or_path:
        raise ValueError(f"Error, model_name_or_path is None, RM must be loaded from a pre-trained model")

    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    # DeepSpeed 模式下检查是否传递了 deepspeed 参数
    import sys
    use_deepspeed = "--deepspeed" in sys.argv

    if use_deepspeed:
        # DeepSpeed 模式下不使用 device_map，由 DeepSpeed 自动管理设备分配
        device_map_arg = None
    else:
        # 非 DeepSpeed 模式下使用 device_map
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        device_map_arg = model_args.device_map

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir
    )

    # 构建模型加载参数
    model_kwargs = {
        "config": config,
        "dtype": dtype,
        "load_in_4bit": model_args.load_in_4bit,
        "load_in_8bit": model_args.load_in_8bit,
        "trust_remote_code": model_args.trust_remote_code,
    }
    # 仅在非 DeepSpeed 模式下添加 device_map 参数
    if device_map_arg is not None:
        model_kwargs["device_map"] = device_map_arg

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    prompt_template = get_conv_template(script_args.template_name)

    # Setup special tokens
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    logger.debug(f"Tokenizer: {tokenizer}")
    return model, tokenizer, prompt_template


def setup_peft_model(model, model_args, script_args):
    """设置PEFT模型"""
    logger.info("Fine-tuning method: LoRA(PEFT)")

    if script_args.peft_path is not None:
        logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
    else:
        logger.info("Init new peft model")
        if model_args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)

        modules_to_save = script_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')

        logger.info(f"Peft target_modules: {target_modules}")
        logger.info(f"Peft lora_rank: {script_args.lora_rank}")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model


def setup_full_model(model):
    """设置全参数训练模型"""
    logger.info("Fine-tuning method: Full parameters training")
    print_trainable_parameters(model)
    return model


def create_preprocess_function(tokenizer, prompt_template):
    """创建数据预处理函数"""

    def preprocess_reward_function(examples):
        """
        Turn the dataset into pairs of Question + Answer, where input_ids_chosen is the preferred question + answer
            and text_rejected is the other.
        """
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for system, history, question, chosen, rejected in zip(
                examples["system"],
                examples["history"],
                examples["question"],
                examples["response_chosen"],
                examples["response_rejected"]
        ):
            system_prompt = system or ""
            chosen_messages = history + [[question, chosen]] if history else [[question, chosen]]
            chosen_prompt = prompt_template.get_prompt(messages=chosen_messages, system_prompt=system_prompt)
            rejected_messages = history + [[question, rejected]] if history else [[question, rejected]]
            rejected_prompt = prompt_template.get_prompt(messages=rejected_messages, system_prompt=system_prompt)

            tokenized_chosen = tokenizer(chosen_prompt)
            tokenized_rejected = tokenizer(rejected_prompt)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    return preprocess_reward_function


def prepare_train_dataset(raw_datasets, data_args, training_args, tokenizer, preprocess_function, full_max_length):
    """
    准备训练数据集

    主要功能：
    1. 检查是否执行训练（do_train标志）
    2. 从raw_datasets中提取训练集
    3. 可选：限制训练样本数量（用于调试或快速训练）
    4. 使用tokenize函数对数据进行批处理分词
    5. 过滤掉超长或无效的样本
    6. 返回处理后的训练集和样本数量

    Args:
        raw_datasets (DatasetDict): 原始数据集，包含 'train' 和 'validation' split
        data_args (DatasetArguments): 数据相关参数（max_train_samples, preprocessing_num_workers等）
        training_args (TrainingArguments): 训练相关参数（do_train, local_rank等）
        tokenizer: 分词器对象，用于解码token查看效果
        preprocess_function (callable): 预处理函数，将原始文本转换为token ids
        full_max_length (int): 最大允许长度（prompt+response），用于过滤样本

    Returns:
        tuple: (train_dataset, max_train_samples)
            - train_dataset: 处理后的训练集（已tokenization和过滤），或None
            - max_train_samples: 实际使用的训练样本数量，或0
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

    # ========== 检查是否执行训练 ==========
    if not training_args.do_train:
        # 如果配置了不执行训练，直接返回None和0
        return None, 0

    # ========== 校验训练集是否存在 ==========
    if "train" not in raw_datasets:
        # raw_datasets中必须包含'train'键，否则抛出异常
        # 这是因为DPO训练必须有训练数据
        raise ValueError("--do_train requires a train dataset")

    # ========== 获取训练集 ==========
    train_dataset = raw_datasets['train']

    # ========== 初始化最大样本数 ==========
    max_train_samples = len(train_dataset)  # 默认使用全部训练样本

    # ========== 限制训练样本数量（可选） ==========
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        # 如果配置了max_train_samples参数，则限制使用样本数
        # 例如：原始1000条，max_train_samples=100，则只使用前100条
        # 用途：调试、快速验证、显存不足时减少数据量
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        # 使用select方法选取前max_train_samples条样本
        train_dataset = train_dataset.select(range(max_train_samples))

    # ========== 打印原始样本数据（调试用）==========
    logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
    # 输出示例：{
    #     'system': '',
    #     'history': [],
    #     'question': '20个关于新鲜果汁菜单的口号...',
    #     'response_chosen': '这里是一个名为\"Dishes\"的餐厅的...',
    #     'response_rejected': '1. \"与菜肴一起品尝新鲜！\"...'
    # }

    # ========== 批量分词处理 ==========
    with training_args.main_process_first(desc="Train dataset tokenization"):
        # main_process_first：在分布式训练中，仅主进程执行tokenization并缓存
        # 其他进程直接加载缓存，避免重复工作
        # desc：描述信息，用于日志输出

        tokenized_dataset = train_dataset.shuffle().map(
            preprocess_function,  # 预处理函数，将文本转为token ids
            batched=True,  # 批处理模式，提高效率
            num_proc=data_args.preprocessing_num_workers,  # 多进程数量，加速处理
            remove_columns=train_dataset.column_names,  # 移除原始列（system, history等），只保留token ids
            load_from_cache_file=not data_args.overwrite_cache,  # 是否加载缓存（不覆盖时加载）
            desc="Running tokenizer on dataset",  # 进度条描述
        )
        # shuffle(): 打乱数据顺序，保证训练随机性
        # map(): 对每个样本应用preprocess_function
        # tokenized_dataset结构：{
        #     'input_ids_chosen': [[151644, 8948, ...], ...],
        #     'input_ids_rejected': [[151644, 8948, ...], ...],
        #     'attention_mask_chosen': [[1, 1, ...], ...],
        #     'attention_mask_rejected': [[1, 1, ...], ...]
        # }

    # ========== 长度过滤 ==========
    train_dataset = tokenized_dataset.filter(
        lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length
                  and 0 < len(x['input_ids_chosen']) <= full_max_length
    )
    # 过滤条件：
    # 1. 0 < len <= full_max_length：长度必须在1到最大长度之间
    #    - 0表示过滤掉空样本
    #    - full_max_length防止超长样本导致OOM（显存溢出）
    # 2. chosen和rejected都必须满足长度限制
    #    - DPO训练需要同时偏好对都有效
    # 示例：full_max_length=1536，则只保留token数量在(0, 1536]范围内的样本

    logger.debug(f"Num train_samples: {len(train_dataset)}")
    # 输出过滤后的样本数量，例如：Num train_samples: 85（从100条过滤到85条）

    # ========== 打印tokenization后的样本（调试用）==========
    logger.debug("Tokenized training example:")
    logger.debug(tokenizer.decode(train_dataset[0]['input_ids_chosen']))
    # 将token ids解码回文本，查看tokenization效果
    # 输出示例：'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n20个关于新鲜果汁菜单的口号...<|im_end|>\n<|im_start|>assistant\n这里是一个名为"Dishes"的餐厅的...'

    # ========== 返回处理结果 ==========
    return train_dataset, max_train_samples
    # train_dataset: 已tokenization、已过滤的训练集
    # max_train_samples: 实际使用的最大样本数（用于日志统计）


def prepare_eval_dataset(raw_datasets, data_args, training_args, tokenizer, preprocess_function, full_max_length):
    """准备评估数据集"""
    if not training_args.do_eval:
        return None, 0

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")

    eval_dataset = raw_datasets["validation"]
    max_eval_samples = len(eval_dataset)

    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")

    with training_args.main_process_first(desc="Eval dataset tokenization"):
        tokenized_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                x['input_ids_chosen']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids_chosen']))

    return eval_dataset, max_eval_samples


def setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, full_max_length):
    """设置训练器"""
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    model.enable_input_require_grads()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=full_max_length, padding="max_length"
        ),
    )
    return trainer


def run_training(trainer, training_args, max_train_samples):
    """运行训练"""
    if not training_args.do_train:
        return

    logger.info("*** Train ***")
    logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.model.config.use_cache = True

    if trainer.is_world_process_zero():
        logger.debug(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(trainer.model, trainer.tokenizer, training_args)


def run_evaluation(trainer, training_args, max_eval_samples):
    """运行评估"""
    if not training_args.do_eval:
        return

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = max_eval_samples

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if trainer.is_world_process_zero():
        logger.debug(f"Eval metrics: {metrics}")


def main():
    # 初始化命令行参数解析器，支持四类参数：模型参数、数据参数、训练参数、脚本参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments))

    # 从命令行解析参数，并转换为对应的 dataclass 实例
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    # 打印解析得到的参数，方便调试和记录
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")

    # 设置日志配置和随机种子，保证训练可复现
    setup_logging_and_seed(training_args)

    # 加载模型、分词器和提示模板（prompt template）
    model, tokenizer, prompt_template = load_model_and_tokenizer(model_args, script_args)

    # 根据脚本参数决定是否使用 PEFT（如 LoRA）进行参数高效微调
    if script_args.use_peft:
        model = setup_peft_model(model, model_args, script_args)
    else:
        # 若不使用 PEFT，加载完整模型进行训练
        model = setup_full_model(model)

    # 加载原始数据集（未经过 tokenization）
    raw_datasets = load_raw_datasets(data_args, model_args)

    # 计算序列的最大长度：输入 + 输出，方便后续 padding 或截断
    full_max_length = data_args.max_source_length + data_args.max_target_length

    # 创建数据预处理函数，包括 tokenization 和构建模型输入
    preprocess_function = create_preprocess_function(tokenizer, prompt_template)

    # 准备训练集，返回 dataset 和可能的样本上限（用于调试或限制训练数据量）
    train_dataset, max_train_samples = prepare_train_dataset(
        raw_datasets, data_args, training_args, tokenizer, preprocess_function, full_max_length
    )

    # 准备验证/评估集，同样返回 dataset 和样本上限
    eval_dataset, max_eval_samples = prepare_eval_dataset(
        raw_datasets, data_args, training_args, tokenizer, preprocess_function, full_max_length
    )

    # 初始化 Trainer，用于统一管理训练、评估、保存模型等操作
    trainer = setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, full_max_length)

    # 执行训练流程
    run_training(trainer, training_args, max_train_samples)

    # 执行评估流程
    run_evaluation(trainer, training_args, max_eval_samples)


if __name__ == "__main__":
    main()

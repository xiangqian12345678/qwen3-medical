import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from latex2sympy2_extended import NormalizationConfig
from loguru import logger
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    用于训练脚本的参数定义，主要包括模型、数据集以及训练配置。
    """

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "用于初始化权重的分词器（Tokenizer），可以是预训练模型名称或本地路径。"}
    )

    # 数据集相关参数 例如： "openai/gsm8k"
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "使用 HuggingFace datasets 库加载数据集的名称，支持逗号分隔的多个数据集。"}
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称，支持逗号分隔的多个配置。"}
    )

    train_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "本地训练文件所在目录，用于加载本地数据集。"}
    )

    train_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "训练样本数量，-1 表示使用全部样本。"}
    )

    subset_name: Optional[str] = field(
        default="main",
        metadata={"help": "数据集子集名称，例如 'default'、'main'，默认为 'default'。"}
    )

    dataset_splits: Optional[str] = field(
        default="train",
        metadata={"help": "数据集分割名称，如 'train'、'validation' 或 'test'。"}
    )

    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={"help": "当没有验证集时，从训练集中划分验证集的百分比，默认为10%。"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=10,
        metadata={"help": "预处理数据时使用的工作线程数量，通常设置为 CPU 核心数。"}
    )

    # QLoRA 参数
    qlora: bool = field(
        default=False,
        metadata={"help": "是否使用 QLoRA 技术进行低秩微调。"}
    )


def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    # 如果输入文本是 None，返回空字符串，避免后续操作报错
    if text is None:
        return ""

    # 去掉文本首尾的空白字符，并将中间连续的空白字符替换为单个空格
    # 同时将所有字符转换为小写
    text = re.sub(r'\s+', ' ', text.strip().lower())

    # 返回处理后的标准化文本
    return text


def extract_answer(text):
    """Extract content between <answer> tags."""

    # 如果输入文本是 None，返回空字符串，避免后续操作报错
    if text is None:
        return ""

    # 使用正则匹配 <answer> 和 </answer> 标签之间的内容
    # re.DOTALL 使得 '.' 可以匹配换行符
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    # 如果匹配成功，返回标签内的内容，并去掉首尾空白字符
    if match:
        return match.group(1).strip()

    # 如果没有 <answer> 标签，返回去掉首尾空白的原文本
    return text.strip()


def accuracy_reward(completions, answer, **kwargs):
    """
    计算每条模型生成结果的准确性奖励（reward）。

    主要流程：
    1. 提取每条生成内容。
    2. 根据是否包含“####”标记（例如 GSM8K 数据集）选择不同解析方式。
    3. 将生成内容与标准答案解析为可比的形式。
    4. 调用 verify 函数验证生成内容是否与标准答案一致，得到奖励。
    5. 返回所有生成结果的奖励列表。

    Args:
        completions (List[List[Dict]]): 模型生成的内容，每条 completion 是一个列表，
            列表中的第一个元素包含字典 {"content": "生成文本"}。
        answer (List[str]): 对应每条 completion 的标准答案。
        **kwargs: 可选参数，当前未使用。

    Returns:
        List[float]: 每条 completion 的奖励分数，通常为 0.0（错误）或 1.0（正确）。
    """

    # 提取每条生成内容的文本
    contents = [completion[0]["content"] for completion in completions]

    rewards = []  # 用于存储每条生成的奖励分数

    for content, sol in zip(contents, answer):
        if '####' in sol:
            # 针对 GSM8K 数据集的特殊处理
            # 将答案中 '####' 后面的部分提取出来进行解析
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            # 将模型生成的文本中提取的答案解析
            answer_parsed = parse(extract_answer(content))
        else:
            # 常规情况：尝试使用 LaTeX 解析标准答案
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],  # 使用 LaTeX 配置解析
            )
            # 对生成内容进行 LaTeX 解析
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,  # 是否规范化单位（这里不规范化）
                            malformed_operators=False,  # 是否允许解析错误的操作符（这里不允许）
                            basic_latex=True,  # 是否解析基础 LaTeX
                            equations=True,  # 是否解析方程
                            boxed="all",  # 解析所有被框的答案
                            units=True,  # 是否解析单位
                        ),
                        boxed_match_priority=0,  # 优先匹配 boxed 内容
                        try_extract_without_anchor=False,  # 是否尝试不依赖 anchor 提取（这里不尝试）
                    )
                ],
                extraction_mode="first_match",
            )

        # 验证生成答案是否与标准答案匹配
        try:
            reward = float(verify(answer_parsed, gold_parsed))  # verify 返回 True/False 或可转为 float 的值
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0  # 如果解析或验证出错，奖励置为 0

        # 输出调试信息
        logger.debug(
            f"predict_answer: {content}, \nground_truth: {sol}, \n"
            f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n"
        )

        rewards.append(reward)  # 保存当前生成的奖励分数

    logger.debug(f'accuracy rewards: {rewards}')  # 输出所有奖励分数
    return rewards


def format_reward(completions, **kwargs):
    """
    计算每条模型生成结果的格式奖励（reward）。

    主要流程：
    1. 检查生成内容是否符合指定格式（正则匹配）。
    2. 如果符合格式，则奖励为 1.0，否则为 0.0。
    3. 返回所有生成结果的奖励列表。

    Args:
        completions (List[List[Dict]]): 模型生成的内容，每条 completion 是一个列表，
            列表中的第一个元素包含字典 {"content": "生成文本"}。
        **kwargs: 可选参数，当前未使用。

    Returns:
        List[float]: 每条 completion 的奖励分数，1.0 表示格式正确，0.0 表示格式不正确。
    """

    # 定义格式匹配的正则表达式
    # 要求生成文本以 <think>...</think><answer>...</answer> 结尾
    pattern = r"<think>.*?</think><answer>.*?</answer>$"

    # 提取每条生成内容的文本
    completion_contents = [completion[0]["content"] for completion in completions]

    # 对每条文本进行正则匹配，检查是否符合指定格式
    matches = [re.match(pattern, content) for content in completion_contents]

    # 根据匹配结果生成奖励：匹配成功 1.0，不成功 0.0
    rewards = [1.0 if match else 0.0 for match in matches]

    # 输出调试信息
    logger.debug(f'format rewards: {rewards}')

    return rewards


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


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


from datasets import load_dataset, concatenate_datasets


def load_datasets_from_hub(script_args):
    """
    Load one or multiple datasets from HuggingFace hub.
    Supports comma-separated dataset_name and dataset_config_name.

    Returns a dict with "train" and "validation" datasets.
    """
    if script_args.dataset_name is None:
        return {}

    # 处理逗号分隔的多个数据集名称
    dataset_names = [name.strip() for name in script_args.dataset_name.split(",") if name.strip()]

    # 处理逗号分隔的多个数据集配置名称
    dataset_configs = None
    if hasattr(script_args, 'dataset_config_name') and script_args.dataset_config_name:
        dataset_configs = [
            None if (c := config.strip()) in ("", "None", "none") else c
            for config in script_args.dataset_config_name.split(',')
        ]

    all_datasets = []

    logger.info(f"Loading datasets from hub: {dataset_names}")
    if dataset_configs:
        logger.info(f"With configs: {dataset_configs}")

    for i, dataset_name in enumerate(dataset_names):
        try:
            # 获取对应的配置名称，如果配置名称数量不足，使用 None
            dataset_config = dataset_configs[i] if dataset_configs and i < len(dataset_configs) else None

            logger.info(f"Loading dataset {i + 1}/{len(dataset_names)}: {dataset_name} (config: {dataset_config})")

            # 加载数据集
            if dataset_config:
                ds = load_dataset(dataset_name, dataset_config, cache_dir=getattr(script_args, 'cache_dir', None))
            else:
                ds = load_dataset(dataset_name, cache_dir=getattr(script_args, 'cache_dir', None))

            all_datasets.append(ds)
            logger.info(f"Successfully loaded: {list(ds.keys())}")

        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}' (config: {dataset_config}): {e}")
            # 继续处理其他数据集，不要因为一个失败就全部停止
            continue

    if not all_datasets:
        raise ValueError("No datasets could be loaded successfully")

    # 合并所有数据集
    merged_datasets = {}

    # 收集所有可能的 split 名称
    all_splits = set()
    for ds in all_datasets:
        all_splits.update(ds.keys())

    # 对每个 split 进行合并
    for split in all_splits:
        split_datasets = []
        for ds in all_datasets:
            if split in ds:
                split_datasets.append(ds[split])

        if split_datasets:
            merged_datasets[split] = concatenate_datasets(split_datasets)
            logger.info(f"Merged {split}: {len(merged_datasets[split])} samples from {len(split_datasets)} datasets")

    # 如果没有 validation 分割，从训练集中划分
    if "validation" not in merged_datasets and "train" in merged_datasets:
        validation_split_percentage = getattr(script_args, 'validation_split_percentage', 10)
        logger.info(f"No validation split found, creating one from train data ({validation_split_percentage}%)")

        train_data = merged_datasets["train"]
        split_result = train_data.train_test_split(test_size=validation_split_percentage / 100.0, seed=42)
        merged_datasets["train"] = split_result["train"]
        merged_datasets["validation"] = split_result["test"]

    logger.info(f"Final merged datasets: {list(merged_datasets.keys())}")
    for split_name, split_data in merged_datasets.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")

    return merged_datasets


def load_datasets_from_files(script_args):
    """Load datasets from local files"""
    dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return {"train": train_dataset, "test": test_dataset}


def load_raw_datasets(script_args):
    """
    Load raw datasets from files and/or HuggingFace Hub, and merge them if both exist.

    Args:
        script_args: Arguments containing dataset_name, train_file_dir, etc.

    Returns:
        raw_datasets (DatasetDict): A dict containing 'train', 'validation', etc.
    """
    raw_datasets = {}

    # 1. Load datasets from files if specified
    file_datasets = {}
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        try:
            file_datasets = load_datasets_from_files(script_args)
            logger.info(f"Loaded datasets from files: {list(file_datasets.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load datasets from files: {e}")

    # 2. Load datasets from hub if specified
    hub_datasets = {}
    if script_args.dataset_name is not None:
        try:
            hub_datasets = load_datasets_from_hub(script_args)
            logger.info(f"Loaded datasets from hub: {list(hub_datasets.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load datasets from hub: {e}")

    # 3. Merge datasets (file_datasets + hub_datasets)
    if not file_datasets and not hub_datasets:
        raise ValueError("No data source specified. Please provide either dataset_name or train_file_dir.")

    # 合并每个 split，如果某一类数据没有该 split，则直接使用另一类
    all_splits = set(file_datasets.keys()) | set(hub_datasets.keys())
    for split in all_splits:
        if split in file_datasets and split in hub_datasets:
            # 使用 concatenate_datasets 合并
            raw_datasets[split] = concatenate_datasets([file_datasets[split], hub_datasets[split]])
        elif split in file_datasets:
            raw_datasets[split] = file_datasets[split]
        else:
            raw_datasets[split] = hub_datasets[split]

    # 确保有 validation 分割，如果没有则从训练集中创建
    if "validation" not in raw_datasets and "train" in raw_datasets:
        validation_split_percentage = getattr(script_args, 'validation_split_percentage', 10)
        logger.info(
            f"No validation split found in merged datasets, creating one from train data ({validation_split_percentage}%)")

        train_data = raw_datasets["train"]
        split_result = train_data.train_test_split(test_size=validation_split_percentage / 100.0, seed=42)
        raw_datasets["train"] = split_result["train"]
        raw_datasets["validation"] = split_result["test"]

    logger.info(f"Final raw datasets: {list(raw_datasets.keys())}")
    for split_name, split_data in raw_datasets.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")

    return raw_datasets


def setup_logging(training_args, model_args, script_args):
    """Setup distributed training initialization and logging."""
    is_main_process = training_args.local_rank in [-1, 0]

    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    return is_main_process


def load_tokenizer(model_args, script_args):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def prepare_datasets(script_args, training_args, is_main_process):
    """Load, process and split datasets."""
    # Load datasets
    raw_datasets = load_raw_datasets(script_args)
    dataset = raw_datasets["train"]

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset
    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': x['answer']
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset


def setup_quantization_config(model_args, script_args, dtype, is_main_process):
    """Setup quantization configuration."""
    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    quantization_config = None
    if (script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit)) or \
            model_args.load_in_4bit or model_args.load_in_8bit:
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    return quantization_config


def setup_device_map(training_args, model_kwargs):
    """Setup device mapping for distributed and multi-GPU training."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    num_gpus = torch.cuda.device_count()

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        model_kwargs["device_map"] = device_map
        # Ensure gradient_accumulation_steps is at least 1 after division
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // world_size, 1)
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    return num_gpus


def load_model(model_args, model_kwargs, is_main_process, num_gpus):
    """
    加载因果语言模型 (Causal LM) 并记录模型设备信息。

    参数:
        model_args: 包含模型加载相关参数的对象（如 model_name_or_path）。
        model_kwargs: 用于 `from_pretrained` 的其他关键字参数。
        is_main_process: bool，是否是主进程（用于分布式训练中仅主进程打印日志）。
        num_gpus: int，当前使用的 GPU 数量。

    返回:
        model: 加载好的模型实例。
    """

    # 如果是主进程，打印初始化信息
    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # 从预训练模型路径加载模型，并传入其他 kwargs
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # 如果模型对象有 hf_device_map 属性（如使用 HuggingFace 的 accelerate 或 device_map 功能）
    if is_main_process and hasattr(model, 'hf_device_map'):
        # 打印模型各部分的设备映射信息
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")

    # 如果没有 hf_device_map 且使用了多 GPU，则手动遍历模型参数打印其设备
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            # 确保参数有 device 属性
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break  # 这里只打印第一个参数的设备，可能是示例或调试用

    return model


def setup_peft_model(model, model_args, training_args, is_main_process):
    """
    配置 LoRA（低秩适应，Low-Rank Adaptation）微调，如果启用了 PEFT（Parameter-Efficient Fine-Tuning）。

    参数：
        model: 待微调的模型（transformers 模型）。
        model_args: 模型相关参数，包含 PEFT/LoRA 配置。
        training_args: 训练参数，例如 gradient_checkpointing。
        is_main_process: 是否为主进程（分布式训练时用于控制日志输出）。

    返回：
        配置好 LoRA 的模型（如果启用 PEFT），否则返回原模型。
    """

    # 检查是否启用了 PEFT/LoRA 微调
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")

        # 如果启用了 gradient checkpointing，则给出警告并关闭
        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False

        # 获取需要应用 LoRA 的模型模块
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None

        # 如果指定 'all'，则自动查找所有线性层进行 LoRA
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)

        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")

        # 配置 LoRA 参数
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
            target_modules=target_modules,  # LoRA 作用的模块
            inference_mode=False,  # 推理模式关闭，表示训练模式
            r=model_args.lora_r,  # LoRA 的秩
            lora_alpha=model_args.lora_alpha,  # LoRA 缩放系数
            lora_dropout=model_args.lora_dropout,  # LoRA dropout 概率
        )

        # 获取 LoRA 模型，将原模型包裹成 PEFT 模型
        model = get_peft_model(model, peft_config)

        # 对量化模型进行 FP16 修复：将可训练参数转为 float32 避免 ValueError
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        # 打印可训练参数信息
        model.print_trainable_parameters()
    else:
        # 如果没有启用 PEFT，则执行全量参数微调
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    # 返回配置后的模型
    return model


def setup_gradient_checkpointing(model, training_args):
    """Setup gradient checkpointing."""
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")


def create_trainer(model, tokenizer, training_args, train_dataset, test_dataset):
    """Initialize GRPO trainer with distributed training support."""
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            accuracy_reward,
            format_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")
    return trainer


def run_training(trainer, training_args, train_dataset, is_main_process):
    """Execute training and handle checkpoint resumption."""
    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    return train_result


def save_model_and_artifacts(trainer, tokenizer, training_args, script_args, is_main_process):
    """保存训练好的模型、分词器及相关训练产物。"""

    # 确保在保存时模型启用缓存，以便推理更高效
    trainer.model.config.use_cache = True

    # 只有主进程执行保存操作（分布式训练时避免重复保存）
    if is_main_process:
        # 保存模型权重及配置到指定输出目录
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    # 在分布式训练中，等待所有进程完成保存
    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        # 保存 tokenizer 到输出目录
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # 创建并保存模型卡（model card），包含数据集信息和标签
        kwargs = {
            "dataset_name": script_args.dataset_name,  # 使用的训练数据集名称
            "tags": ["r1", "grpo"],  # 可自定义标签，用于描述模型特性
        }
        trainer.create_model_card(**kwargs)

        # 再次确保模型配置启用缓存
        trainer.model.config.use_cache = True
        # 保存模型配置到输出目录（包含如 hidden_size、num_layers 等信息）
        trainer.model.config.save_pretrained(training_args.output_dir)

        logger.info("*** Training complete! ***")  # 日志提示训练及保存完成


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    """
    GRPO 主训练函数（模块化设计）

    参数:
        model_args (ModelConfig): 模型配置，包括模型路径、数据类型、注意力实现等。
        script_args (ScriptArguments): 脚本参数，包括数据集路径、tokenizer 配置等。
        training_args (GRPOConfig): GRPO 训练相关配置，包括学习率、batch size、分布式设置等。

    功能:
        - 初始化日志与分布式训练
        - 加载 tokenizer
        - 准备训练/测试数据集
        - 初始化模型（可选量化、PEFT）
        - 配置梯度检查点
        - 创建 trainer 并执行训练
        - 保存模型与相关训练产物
    """

    # is_main_process: 是否为主进程，用于分布式训练中控制日志打印和模型保存
    is_main_process = setup_logging(training_args, model_args, script_args)

    # tokenizer 用于将文本转为模型输入的 token id
    tokenizer = load_tokenizer(model_args, script_args)

    # 根据 script_args 和 training_args 加载训练集和测试集，并支持分布式数据划分
    train_dataset, test_dataset = prepare_datasets(script_args, training_args, is_main_process)

    # 模型初始化
    dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    # 设置模型数据类型，例如 float32、float16 等

    # 配置量化（可选）
    quantization_config = setup_quantization_config(model_args, script_args, dtype, is_main_process)
    # 支持模型量化以减少显存占用和加速推理

    # 模型参数配置
    model_kwargs = dict(
        # 指定要加载的模型版本/分支，比如 "main"、"v1.0" 等
        revision=model_args.model_revision,

        # 是否信任远程仓库上的自定义代码，如果模型包含自定义模块（如自定义 attention 层），需要设为 True
        trust_remote_code=model_args.trust_remote_code,

        # 指定注意力机制的实现方式，例如 "flash_attention" 或 "standard_attention"，可能影响速度和显存
        attn_implementation=model_args.attn_implementation,

        # 指定模型参数的数据类型，例如 torch.float32 / torch.float16 / torch.bfloat16
        dtype=dtype,

        # 是否启用低 CPU 内存占用模式，通常在显存受限或非 ZeRO3 分布式训练时启用
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),

        # 量化配置，如果想用 4bit/8bit 量化加载模型，可通过这个参数设置
        quantization_config=quantization_config,
    )

    # 设置设备映射并加载模型
    num_gpus = setup_device_map(training_args, model_kwargs)
    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    # 根据设备映射加载模型，支持多 GPU 或分布式训练
    model = load_model(model_args, model_kwargs, is_main_process, num_gpus)

    # 如果启用了 PEFT（如 LoRA），会对模型进行轻量微调的适配
    model = setup_peft_model(model, model_args, training_args, is_main_process)

    # 开启梯度检查点可以降低显存占用，但可能增加计算开销
    setup_gradient_checkpointing(model, training_args)

    # 封装训练循环，包括优化器、调度器、评估逻辑等
    trainer = create_trainer(model, tokenizer, training_args, train_dataset, test_dataset)

    # 开始训练过程，支持分布式训练和训练中评估
    run_training(trainer, training_args, train_dataset, is_main_process)

    # 保存训练好的模型、tokenizer、配置文件等，以便推理或继续训练
    save_model_and_artifacts(trainer, tokenizer, training_args, script_args, is_main_process)


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

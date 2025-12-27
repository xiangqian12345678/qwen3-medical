import os
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import torch
import torch.utils.data
from datasets import load_dataset, DatasetDict, concatenate_datasets
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import ORPOConfig, ORPOTrainer

from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ModelArguments:
    """Model related arguments"""
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
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


@dataclass
class DatasetArguments:
    """Dataset related arguments"""
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class PeftArguments:
    """PEFT/LoRA related arguments"""
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    target_modules: Optional[str] = field(default="all", metadata={"help": "The target modules for peft"})
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments:
    """Training related arguments"""
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "Train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Eval batch size per device"})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "The number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "The weight decay"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer type"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Whether to use fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "The number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "X steps to evaluate the model"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "X steps to log the model"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "The output directory"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})


@dataclass
class ORPOSpecificArguments:
    """ORPO specific arguments"""
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    orpo_beta: float = field(
        default=0.1,
        metadata={"help": "The beta (lambda) parameter in ORPO loss representing the weight of the SFT loss."},
    )


@dataclass
class ScriptArguments(ModelArguments, DatasetArguments, PeftArguments, TrainingArguments, ORPOSpecificArguments):
    """
    Combined arguments for ORPO training
    """

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


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


def parse_arguments():
    """Parse command line arguments and return combined args"""
    parser = HfArgumentParser(
        (ModelArguments, DatasetArguments, PeftArguments, TrainingArguments, ORPOSpecificArguments)
    )
    model_args, dataset_args, peft_args, training_args, orpo_args = parser.parse_args_into_dataclasses()

    args = ScriptArguments(
        **model_args.__dict__,
        **dataset_args.__dict__,
        **peft_args.__dict__,
        **training_args.__dict__,
        **orpo_args.__dict__,
    )
    return args


def setup_distributed_training():
    """Setup distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank in [-1, 0]
    return local_rank, is_main_process


def load_tokenizer(args, is_main_process):
    """Load and configure tokenizer"""
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = get_conv_template(args.template_name)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        if is_main_process:
            logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        if is_main_process:
            logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        if is_main_process:
            logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    if is_main_process:
        logger.debug(f"Tokenizer: {tokenizer}")

    return tokenizer, prompt_template


def load_datasets_from_hub(args):
    """Load multiple datasets from HuggingFace Hub and merge them"""
    # 将逗号分隔的 dataset_name 和 config_name 转为列表
    dataset_names = [name.strip() for name in args.dataset_name.split(",")]
    config_names = [cfg.strip() if cfg else None for cfg in args.dataset_config_name.split(",")]

    # 如果 config_names 数量不足 dataset_names，补 None
    if len(config_names) < len(dataset_names):
        config_names.extend([None] * (len(dataset_names) - len(config_names)))

    merged_datasets = {}

    for name, config in zip(dataset_names, config_names):
        raw_datasets = load_dataset(name, config, cache_dir=args.cache_dir)

        for split in raw_datasets.keys():
            if split not in merged_datasets:
                merged_datasets[split] = raw_datasets[split]
            else:
                merged_datasets[split] = concatenate_datasets([merged_datasets[split], raw_datasets[split]])

    # 如果没有 validation split，则按比例从 train 切分
    if "validation" not in merged_datasets:
        validation_size = int(len(merged_datasets["train"]) * args.validation_split_percentage / 100)
        merged_datasets["validation"] = merged_datasets["train"].select(range(validation_size))
        merged_datasets["train"] = merged_datasets["train"].select(
            range(validation_size, len(merged_datasets["train"])))

    return DatasetDict(merged_datasets)


def load_datasets_from_files(args, is_main_process):
    """Load datasets from local files"""
    data_files = {}
    if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
        train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
            f'{args.train_file_dir}/**/*.jsonl', recursive=True)
        if is_main_process:
            logger.info(f"train files: {', '.join(train_data_files)}")
        data_files["train"] = train_data_files

    if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
        eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
            f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
        if is_main_process:
            logger.info(f"eval files: {', '.join(eval_data_files)}")
        data_files["validation"] = eval_data_files

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
        )
    return raw_datasets


def load_raw_datasets(args, is_main_process):
    """Load raw datasets from hub or files, and merge them if both exist"""
    raw_datasets = None

    # Load datasets from hub if specified
    if args.dataset_name is not None:
        try:
            hub_datasets = load_datasets_from_hub(args)
            if is_main_process:
                logger.info(f"Loaded datasets from hub: {list(hub_datasets.keys())}")
            raw_datasets = hub_datasets
        except Exception as e:
            if is_main_process:
                logger.warning(f"Failed to load datasets from hub: {e}")
            if args.train_file_dir is None and args.validation_file_dir is None:
                raise ValueError("No valid data source found. Both hub and file loading failed.")

    # Load datasets from files if specified
    if args.train_file_dir is not None or args.validation_file_dir is not None:
        try:
            file_datasets = load_datasets_from_files(args, is_main_process)
            if is_main_process:
                logger.info(f"Loaded datasets from files: {list(file_datasets.keys())}")

            # Merge datasets if both hub and file datasets exist
            if raw_datasets is not None:
                for split_name, split_data in file_datasets.items():
                    if split_name in raw_datasets:
                        # Concatenate datasets for the same split
                        from datasets import concatenate_datasets
                        raw_datasets[split_name] = concatenate_datasets([raw_datasets[split_name], split_data])
                        if is_main_process:
                            logger.info(f"Merged {split_name} datasets. Total size: {len(raw_datasets[split_name])}")
                    else:
                        # Add new split
                        raw_datasets[split_name] = split_data
                        if is_main_process:
                            logger.info(f"Added {split_name} split from files. Size: {len(split_data)}")
            else:
                raw_datasets = file_datasets
        except Exception as e:
            if is_main_process:
                logger.warning(f"Failed to load datasets from files: {e}")
            if raw_datasets is None:
                raise ValueError("No valid data source found. Both hub and file loading failed.")

    if raw_datasets is None:
        raise ValueError(
            "No data source specified. Please provide either dataset_name or train_file_dir/validation_file_dir.")

    if is_main_process:
        logger.info(f"Final raw datasets: {list(raw_datasets.keys())}")
        for split_name, split_data in raw_datasets.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")

    return raw_datasets


def create_dataset_preprocessor(prompt_template, full_max_length):
    """Create dataset preprocessing function"""

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        prompts = []
        for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
            system_prompt = system or ""
            history_with_question = history + [[question, '']] if history else [[question, '']]
            prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    def filter_by_length(example):
        return (0 < len(example['prompt'] + example['chosen']) <= full_max_length
                and 0 < len(example['prompt'] + example['rejected']) <= full_max_length)

    return return_prompt_and_responses, filter_by_length


def preprocess_datasets(raw_datasets, args, prompt_template, full_max_length, is_main_process):
    """Preprocess training and evaluation datasets"""
    return_prompt_and_responses, filter_by_length = create_dataset_preprocessor(prompt_template, full_max_length)

    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        tokenized_dataset = train_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )
        train_dataset = tokenized_dataset.filter(filter_by_length)

        if is_main_process:
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("First train example:")
            first_example = train_dataset[0]
            logger.debug(f"prompt:\n{first_example['prompt']}")
            logger.debug(f"chosen:\n{first_example['chosen']}")
            logger.debug(f"rejected:\n{first_example['rejected']}")

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        if is_main_process:
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")

        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(filter_by_length)

        if is_main_process:
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("First eval example:")
            first_example = eval_dataset[0]
            logger.debug(f"prompt:\n{first_example['prompt']}")
            logger.debug(f"chosen:\n{first_example['chosen']}")
            logger.debug(f"rejected:\n{first_example['rejected']}")

    return train_dataset, max_train_samples, eval_dataset, max_eval_samples


def load_model(args, local_rank, is_main_process):
    """
    加载并配置 Causal LM 模型，支持：
    - 单卡 / 多卡 DDP
    - FP16 / BF16 / FP32
    - 4bit / 8bit 量化（QLoRA）
    - Gradient Checkpointing
    - DeepSpeed ZeRO-3（部分场景）

    Args:
        args: 训练参数（ScriptArguments / TrainingArguments）
        local_rank: 当前进程的 GPU rank（DDP 下使用）
        is_main_process: 是否为主进程（用于日志打印）

    Returns:
        model: 加载完成并配置好的模型
        ddp: 是否启用 DDP（bool）
    """

    # ---------------------------------------------------------
    # 1. 处理 torch_dtype
    # ---------------------------------------------------------
    # args.dtype 可以是:
    # - "auto" / None：交给 transformers 自动处理
    # - "float16" / "bfloat16" / "float32"
    torch_dtype = (
        args.dtype
        if args.dtype in ["auto", None]
        else getattr(torch, args.dtype)
    )

    # ---------------------------------------------------------
    # 2. 判断是否为 DDP 训练
    # ---------------------------------------------------------
    # WORLD_SIZE > 1 表示多进程（多卡）训练
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    if ddp:
        # DDP Distributed Data Parallel多数据并行模式下：
        # - 每个进程只负责一张卡
        # - device_map 必须明确指定到 local_rank
        args.device_map = {"": local_rank}

        # 梯度累积步数需要按 world_size 均分
        # （保证全局 batch size 不变）
        args.gradient_accumulation_steps = (
                args.gradient_accumulation_steps // world_size
        )
    else:
        # 单卡 / 推理模式下使用 transformers 自动 device_map
        args.device_map = "auto"

    if is_main_process:
        logger.info(f"Device map: {args.device_map}")

    # ---------------------------------------------------------
    # 3. QLoRA 与 DeepSpeed ZeRO-3 兼容性提示
    # ---------------------------------------------------------
    # 目前 ZeRO-3 与 QLoRA（bitsandbytes）存在已知不兼容问题
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # ---------------------------------------------------------
    # 4. 加载模型配置（不加载权重）
    # ---------------------------------------------------------
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )

    # ---------------------------------------------------------
    # 5. 量化配置日志
    # ---------------------------------------------------------
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(
            f"Quantizing model, "
            f"load_in_4bit: {args.load_in_4bit}, "
            f"load_in_8bit: {args.load_in_8bit}"
        )

    # ---------------------------------------------------------
    # 6. 加载模型权重
    # ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,

        # ZeRO-3 下不能启用 low_cpu_mem_usage
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),

        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,

        # QLoRA / 4bit / 8bit 量化配置
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,  # 二次量化，进一步省显存
            bnb_4bit_quant_type="nf4",  # NF4：QLoRA 标配
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )

    # ---------------------------------------------------------
    # 7. 强制将可训练参数转成 FP32
    # ---------------------------------------------------------
    # 这是 QLoRA / LoRA 常见做法：
    # - base model 可能是 4bit / 8bit
    # - LoRA / 可训练参数保持 FP32，数值更稳定
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # ---------------------------------------------------------
    # 8. Gradient Checkpointing 设置
    # ---------------------------------------------------------
    if args.gradient_checkpointing:
        # 启用梯度检查点以节省显存
        model.gradient_checkpointing_enable()

        # 开启 gradient checkpointing 时必须关闭 cache
        # 否则会导致 forward 过程错误
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    return model, ddp


def create_trainer(args, model, tokenizer, train_dataset, eval_dataset, full_max_length, ddp, is_main_process):
    """Create ORPO trainer with configuration"""
    training_args = ORPOConfig(
        # 模型生成的最大长度（包括 prompt + 生成内容）
        max_length=full_max_length,

        # prompt 的最大长度，超过会截断
        max_prompt_length=args.max_source_length,

        # 每个设备（GPU/TPU）训练时的 batch size
        per_device_train_batch_size=args.per_device_train_batch_size,

        # 每个设备验证时的 batch size
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        # 最大训练步数，如果设置了 num_train_epochs 通常可以忽略
        max_steps=args.max_steps,

        # 日志记录间隔（训练多少步记录一次）
        logging_steps=args.logging_steps,

        # 保存模型检查点的间隔步数
        save_steps=args.save_steps,

        # 梯度累积步数，用于模拟更大 batch size
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # 是否启用梯度检查点（节省显存，训练更大模型）
        gradient_checkpointing=args.gradient_checkpointing,

        # 学习率
        learning_rate=args.learning_rate,

        # 验证策略，可选 'steps' 或 'epoch'
        eval_strategy=args.eval_strategy,

        # 验证间隔步数（eval_strategy='steps' 时生效）
        eval_steps=args.eval_steps,

        # 输出目录（保存模型、日志等）
        output_dir=args.output_dir,

        # 报告工具，可选 ['wandb', 'tensorboard', 'none'] 等
        report_to=args.report_to,

        # 学习率调度器类型，如 'linear', 'cosine' 等
        lr_scheduler_type=args.lr_scheduler_type,

        # 预热步数（learning rate warmup）
        warmup_steps=args.warmup_steps,

        # 优化器类型，如 'adamw', 'adamw_torch' 等
        optim=args.optim,

        # 是否使用 bfloat16 训练（节省显存，同时保持数值稳定性）
        bf16=args.bf16,

        # 是否使用 float16 训练（节省显存，但可能出现数值不稳定）
        fp16=args.fp16,

        # 是否删除 dataset 中未使用的列（提高训练效率）
        remove_unused_columns=args.remove_unused_columns,

        # 训练任务名称（用于日志追踪）
        run_name=f"orpo_v1",

        # ORPO 特有参数，控制奖励加权 beta
        beta=args.orpo_beta,

        # DDP 分布式训练时是否查找未使用参数
        ddp_find_unused_parameters=False if ddp else None,
    )

    peft_config = None
    if args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")

    trainer = ORPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if args.use_peft else None,
    )

    if is_main_process:
        print_trainable_parameters(trainer.model)

    return trainer


def run_training(trainer, max_train_samples, is_main_process):
    """Run training process"""
    if is_main_process:
        logger.info("*** Train ***")

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if trainer.is_world_process_zero():
        logger.debug(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {trainer.args.output_dir}")
        trainer.save_model(trainer.args.output_dir)
        trainer.tokenizer.save_pretrained(trainer.args.output_dir)
        trainer.model.save_pretrained(trainer.args.output_dir)


def run_evaluation(trainer, max_eval_samples):
    """Run evaluation process"""
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = max_eval_samples
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if trainer.is_world_process_zero():
        logger.debug(f"Eval metrics: {metrics}")


def main():
    # 解析命令行参数或配置文件参数
    args = parse_arguments()

    # 设置分布式训练环境，返回当前进程的 rank 以及是否为主进程
    local_rank, is_main_process = setup_distributed_training()

    # 仅在主进程打印日志，避免多进程重复输出
    if is_main_process:
        logger.info(f"Parse args: {args}")

    # 加载 tokenizer 和 prompt 模板
    # tokenizer 用于将文本转为模型可识别的 token
    # prompt_template 用于生成训练 / 推理时的输入格式
    tokenizer, prompt_template = load_tokenizer(args, is_main_process)

    # 加载原始数据集（可以是本地文件或 HuggingFace Hub 数据集）
    raw_datasets = load_raw_datasets(args, is_main_process)

    # 设置最大输入长度
    max_source_length = args.max_source_length  # prompt 的最大长度
    max_target_length = args.max_target_length  # 模型生成目标的最大长度
    full_max_length = max_source_length + max_target_length  # 输入 + 输出的总长度

    # 对数据集进行预处理：
    # 1. 使用 tokenizer 和 prompt_template 对文本进行编码
    # 2. 截断或填充到 full_max_length
    # 3. 返回训练集和验证集，以及每个集的最大样本数
    train_dataset, max_train_samples, eval_dataset, max_eval_samples = preprocess_datasets(
        raw_datasets, args, prompt_template, full_max_length, is_main_process
    )

    # 加载模型，并返回是否为 DDP 分布式模式
    model, ddp = load_model(args, local_rank, is_main_process)

    # 创建训练器（Trainer），封装训练、评估、优化器、学习率调度等功能
    trainer = create_trainer(args, model, tokenizer, train_dataset, eval_dataset, full_max_length, ddp, is_main_process)

    # 执行训练（如果 args.do_train=True）
    # 包含梯度累积、日志打印、模型保存等逻辑
    if args.do_train:
        run_training(trainer, max_train_samples, is_main_process)

    # 执行评估（如果 args.do_eval=True）
    # 计算验证集 loss、准确率或其他指标
    if args.do_eval:
        run_evaluation(trainer, max_eval_samples)


if __name__ == "__main__":
    main()

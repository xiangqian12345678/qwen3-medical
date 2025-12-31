import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
from loguru import logger
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
    用于训练脚本的参数定义，支持 GRPO 标准格式数据（一个问题，多个有序答案）。
    """

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "用于初始化权重的分词器（Tokenizer），可以是预训练模型名称或本地路径。"}
    )

    # 数据集相关参数
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "使用 HuggingFace datasets 库加载数据集的名称。"}
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称。"}
    )

    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "本地训练文件路径（JSON/JSONL格式），每个样本包含 prompt 和多个 responses。"}
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

    # GRPO 标准数据相关参数
    use_standard_rewards: bool = field(
        default=True,
        metadata={"help": "是否使用数据集中提供的 reward 值（标准 GRPO 格式），如果为 False 则通过 reward 函数计算。"}
    )

    max_responses_per_prompt: Optional[int] = field(
        default=8,
        metadata={"help": "每个 prompt 使用的最大 response 数量，按 reward 排序后保留前 N 个用于 GRPO 对比学习。"}
    )

    # QLoRA 参数
    qlora: bool = field(
        default=False,
        metadata={"help": "是否使用 QLoRA 技术进行低秩微调。"}
    )


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The assistant should provide accurate, well-reasoned responses based on medical knowledge."
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
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


from datasets import load_dataset, Dataset, DatasetDict


def load_standard_grpo_dataset(script_args):
    """
    加载 GRPO 标准格式数据集。

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

    处理流程:
    1. 加载原始数据
    2. 将每个样本展开为多个 prompt-response 对
    3. 按 reward 排序，保留前 max_responses_per_prompt 个回答（用于 GRPO 对比学习）
    4. 构建 GRPO 需要的格式
    """
    if script_args.train_file_path and os.path.exists(script_args.train_file_path):
        # 从本地文件加载
        file_extension = os.path.splitext(script_args.train_file_path)[1].lower()
        
        if file_extension == '.jsonl':
            # 读取 JSONL 文件
            data = []
            with open(script_args.train_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif file_extension == '.json':
            # 读取 JSON 文件
            with open(script_args.train_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded {len(data)} samples from {script_args.train_file_path}")
        
    elif script_args.dataset_name:
        # 从 HuggingFace Hub 加载
        logger.info(f"Loading dataset from hub: {script_args.dataset_name}")
        dataset = load_dataset(
            script_args.dataset_name,
            script_args.dataset_config_name,
            split=script_args.dataset_splits,
            cache_dir=getattr(script_args, 'cache_dir', None)
        )
        data = dataset.to_list()
        logger.info(f"Loaded {len(data)} samples from hub")
    else:
        raise ValueError("必须指定 train_file_path 或 dataset_name")

    # 处理数据，展开为 GRPO 格式
    processed_data = []

    for idx, sample in enumerate(data):
        prompt = sample.get('prompt', '')
        responses = sample.get('responses', [])

        if not prompt or not responses:
            logger.warning(f"Sample {idx} missing prompt or responses, skipping")
            continue

        # 按 reward 降序排序
        sorted_responses = sorted(responses, key=lambda x: x.get('reward', 0), reverse=True)

        # 限制每个 prompt 的 response 数量，保留多个回答用于 GRPO 对比
        top_responses = sorted_responses[:script_args.max_responses_per_prompt]

        # 为每个 response 创建一个样本（GRPO 需要多个回答用于对比学习）
        for resp in top_responses:
            processed_data.append({
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                'response': resp.get('text', ''),
                'reward': float(resp.get('reward', 0.0)),
                'group_id': sample.get('group_id', f'group_{idx}'),
                'response_id': resp.get('response_id', '')
            })
    
    logger.info(f"Processed {len(processed_data)} prompt-response pairs from {len(data)} original samples")
    
    # 创建 Dataset
    dataset = Dataset.from_list(processed_data)

    return dataset


def prepare_standard_datasets(script_args, training_args, is_main_process):
    """
    准备 GRPO 标准格式的训练数据集。

    返回格式:
        train_dataset: Dataset, 包含 'prompt', 'response', 'reward' 等字段
        eval_dataset: Dataset, 验证集
    """
    # 加载数据集
    dataset = load_standard_grpo_dataset(script_args)
    
    # 限制样本数量
    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(min(script_args.train_samples, len(dataset))))
        logger.info(f"Limited training samples to {len(dataset)}")
    
    # 划分训练集和验证集
    split_result = dataset.train_test_split(
        test_size=script_args.validation_split_percentage / 100.0,
        seed=42
    )
    
    train_dataset = split_result['train']
    eval_dataset = split_result['test']
    
    if is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        logger.info(f"Sample data: {train_dataset[0]}")
    
    return train_dataset, eval_dataset


def standard_reward_function(prompts, completions, dataset_rewards=None, **kwargs):
    """
    GRPO 标准格式奖励函数，直接使用数据集中提供的 reward 值。

    注意：这个函数主要用于兼容 GRPOTrainer 的接口，
    实际的 reward 值会在 prepare_standard_datasets 中预先计算。
    """
    if dataset_rewards is not None and len(dataset_rewards) > 0:
        return dataset_rewards
    
    # 如果没有预先计算的 reward，返回默认值
    return [0.0] * len(completions)


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


def setup_quantization_config(model_args, script_args, dtype, is_main_process):
    """Setup quantization configuration."""
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

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

    if is_deepspeed_zero3_enabled():
        is_main_process = (int(os.environ.get("RANK", "0")) == 0)
        if is_main_process:
            logger.info("DeepSpeed Zero-3 detected, skipping device_map setup")
        return num_gpus

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        model_kwargs["device_map"] = device_map
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // world_size, 1)
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    return num_gpus


def load_model(model_args, model_kwargs, is_main_process, num_gpus):
    """Load causal language model and log model device information."""
    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    return model


def setup_peft_model(model, model_args, training_args, is_main_process):
    """Configure LoRA fine-tuning if PEFT is enabled."""
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")

        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False

        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None

        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)

        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )

        model = get_peft_model(model, peft_config)

        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

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


def create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    """Initialize GRPO trainer for standard format data."""
    # 对于标准 GRPO 格式，我们需要自定义一个适配器来处理预计算的 rewards
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[standard_reward_function],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized for standard format ***")
    return trainer


def run_training(trainer, training_args, train_dataset, is_main_process):
    """Execute training and handle checkpoint resumption."""
    last_checkpoint = get_checkpoint(training_args)
    last_checkpoint = None  # 禁用检查点恢复以避免DeepSpeed ZeRO-3兼容性问题

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

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
    trainer.model.config.use_cache = True

    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo", "standard-format"],
        }
        trainer.create_model_card(**kwargs)

        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        logger.info("*** Training complete! ***")


def grpo_standard_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    """
    GRPO 标准格式主训练函数。

    参数:
        model_args: 模型配置
        script_args: 脚本参数（包含标准格式数据配置）
        training_args: GRPO 训练配置

    标准格式数据特点:
        - 一个问题对应多个有序答案
        - 每个答案都有预计算的 reward 值
        - 直接使用这些 reward 进行训练，无需在线计算
    """
    is_main_process = setup_logging(training_args, model_args, script_args)
    
    tokenizer = load_tokenizer(model_args, script_args)
    
    # 加载标准格式数据集
    train_dataset, eval_dataset = prepare_standard_datasets(script_args, training_args, is_main_process)

    # 模型初始化
    dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    quantization_config = setup_quantization_config(model_args, script_args, dtype, is_main_process)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = setup_device_map(training_args, model_kwargs)
    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    model = load_model(model_args, model_kwargs, is_main_process, num_gpus)
    model = setup_peft_model(model, model_args, training_args, is_main_process)
    setup_gradient_checkpointing(model, training_args)

    # 创建 trainer
    trainer = create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset)

    # 训练
    run_training(trainer, training_args, train_dataset, is_main_process)

    # 保存模型
    save_model_and_artifacts(trainer, tokenizer, training_args, script_args, is_main_process)


def main():
    """
    GRPO 标准格式训练入口。

    数据格式说明:
        {
          "prompt": "问题描述",
          "group_id": "分组ID",
          "responses": [
            {"response_id": "r1", "text": "回答1", "reward": 0.92},
            {"response_id": "r2", "text": "回答2", "reward": 0.35}
          ]
        }

    参数说明:
        - train_file_path: 本地 JSON/JSONL 数据文件路径
        - dataset_name: HuggingFace 数据集名称
        - max_responses_per_prompt: 每个问题使用的最多回答数（按 reward 排序）
        - use_standard_rewards: 是否使用预计算的 reward 值
    """
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    grpo_standard_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()

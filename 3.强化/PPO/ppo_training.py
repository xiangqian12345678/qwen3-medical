import os
from glob import glob

from datasets import load_dataset, concatenate_datasets
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    ModelConfig,
    get_peft_config,
)
from trl.trainer.ppo_config import PPOConfig as OriginalPPOConfig

from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass, field
from typing import Optional

# Monkey patch PPOConfig to fix world_size issue
_original_ppo_config_init = OriginalPPOConfig.__init__


def _patched_ppo_config_init(self, *args, **kwargs):
    # Ensure world_size has a default value
    if 'world_size' not in kwargs or kwargs['world_size'] is None:
        kwargs['world_size'] = int(os.environ.get("WORLD_SIZE", "1"))
    _original_ppo_config_init(self, *args, **kwargs)


OriginalPPOConfig.__init__ = _patched_ppo_config_init


@dataclass
class PPOArguments:
    """
    PPO（Proximal Policy Optimization）训练阶段的参数配置。

    该配置主要用于控制：
    1. PPO 训练所使用的数据来源（HuggingFace 数据集 or 本地 json/jsonl 文件）
    2. 训练 / 评估数据集的切分方式
    3. Prompt 构造所使用的模板
    4. Prompt 输入长度限制

    注意：
    - 这里的参数主要描述「数据与 Prompt 相关配置」
    - 不包含 PPO 算法本身的超参数（如 kl_coef、clip_range 等）
    """

    # =========================
    # 数据集（HuggingFace Hub）
    # =========================

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Tokenizer 的路径或名称。"
                "如果为 None，则使用 sft_model_path 中的 tokenizer。"
            )
        },
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "HuggingFace Hub 上的数据集名称。"
                "例如：'Anthropic/hh-rlhf'、'lvwerra/stack-exchange-paired'。"
                "当该参数不为 None 时，优先使用 HuggingFace 数据集加载数据。"
            )
        },
    )

    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "HuggingFace 数据集的配置名称（subset / config）。"
                "某些数据集必须指定该参数，否则无法正确加载。"
                "如果数据集没有子配置，可保持为 None。"
            )
        },
    )

    dataset_train_split: str = field(
        default="train",
        metadata={
            "help": (
                "用于 PPO 训练的数据集 split 名称。"
                "该 split 通常只提供 prompt（query），"
                "PPO 会基于模型生成的 response 计算 reward。"
            )
        },
    )

    dataset_test_split: str = field(
        default="test",
        metadata={
            "help": (
                "用于 PPO 训练过程中的评估数据集 split 名称。"
                "通常用于监控 reward、KL divergence 等指标，"
                "而非传统的 supervised evaluation。"
            )
        },
    )

    # =========================
    # 本地数据文件（json / jsonl）
    # =========================

    train_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "本地训练数据文件所在目录（json 或 jsonl 格式）。"
                "当 dataset_name 为 None 时使用该参数。"
                "数据通常包含 prompt 字段，"
                "PPO 训练阶段不要求必须提供 labels。"
            )
        },
    )

    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "本地评估数据文件所在目录（json 或 jsonl 格式）。"
                "用于 PPO 训练过程中的评估与监控，"
                "建议即使数据量较小也提供，以防 reward 崩溃或 KL 发散。"
            )
        },
    )

    # =========================
    # Prompt 模板配置
    # =========================

    template_name: Optional[str] = field(
        default="vicuna",
        metadata={
            "help": (
                "Prompt 模板名称，用于将原始 prompt 构造成模型输入。"
                "例如：'vicuna'、'chatml'、'llama2'、'qwen' 等。"
                "在 RLHF 流程中（SFT / Reward Model / PPO），"
                "模板必须保持严格一致，否则会导致 reward 分布不一致。"
            )
        },
    )

    # =========================
    # 输入长度限制
    # =========================

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Prompt（source / query）的最大 token 长度限制。"
                "该长度不包含模型生成的 response 部分。"
                "长度设置过小可能丢失关键信息，"
                "设置过大则会增加显存占用并影响 PPO rollout 效率。"
            )
        },
    )


def load_datasets_from_hub(args):
    """
    从HuggingFace Hub加载数据集。

    Args:
        args: 包含数据集配置的参数

    Returns:
        dict: 包含train和validation数据集的字典
    """
    dataset_names = [name.strip() for name in args.dataset_name.split(",")]
    dataset_configs = [cfg.strip() if cfg else None for cfg in args.dataset_config.split(",")] \
        if args.dataset_config else [None] * len(dataset_names)

    all_train_datasets = []
    all_eval_datasets = []

    for name, cfg in zip(dataset_names, dataset_configs):
        dataset = load_dataset(name, cfg, split=args.dataset_train_split)

        # Split into train and eval
        eval_samples = min(100, len(dataset) // 10)
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

        all_train_datasets.append(train_dataset)
        all_eval_datasets.append(eval_dataset)

    # 合并所有数据集
    final_train_dataset = concatenate_datasets(all_train_datasets)
    final_eval_dataset = concatenate_datasets(all_eval_datasets)

    return {"train": final_train_dataset, "validation": final_eval_dataset}


def load_datasets_from_files(args):
    """
    从本地文件加载数据集。

    Args:
        args: 包含数据文件目录的参数

    Returns:
        dict: 包含train和validation数据集的字典
    """
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

    dataset = load_dataset(
        'json',
        data_files=data_files,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))

    return {"train": train_dataset, "validation": eval_dataset}

    return {"train": train_dataset, "validation": eval_dataset}


def load_raw_datasets(args):
    """
    从HuggingFace Hub或本地文件加载原始数据集。

    Args:
        args: 包含数据源配置的参数

    Returns:
        dict: 包含train和validation数据集的字典

    Raises:
        ValueError: 当没有有效的数据源时抛出异常
    """
    raw_datasets = None

    # Load datasets from hub if specified
    if args.dataset_name is not None:
        try:
            hub_datasets = load_datasets_from_hub(args)
            logger.info(f"Loaded datasets from hub: {list(hub_datasets.keys())}")
            raw_datasets = hub_datasets
        except Exception as e:
            logger.warning(f"Failed to load datasets from hub: {e}")
            if args.train_file_dir is None and args.validation_file_dir is None:
                raise ValueError("No valid data source found. Both hub and file loading failed.")

    # Load datasets from files if specified
    if args.train_file_dir is not None or args.validation_file_dir is not None:
        try:
            file_datasets = load_datasets_from_files(args)
            logger.info(f"Loaded datasets from files: {list(file_datasets.keys())}")

            # Merge datasets if both hub and file datasets exist
            if raw_datasets is not None:
                for split_name, split_data in file_datasets.items():
                    if split_name in raw_datasets:
                        # Concatenate datasets for the same split
                        raw_datasets[split_name] = concatenate_datasets([raw_datasets[split_name], split_data])
                        logger.info(f"Merged {split_name} datasets. Total size: {len(raw_datasets[split_name])}")
                    else:
                        # Add new split
                        raw_datasets[split_name] = split_data
                        logger.info(f"Added {split_name} split from files. Size: {len(split_data)}")
            else:
                raw_datasets = file_datasets
        except Exception as e:
            logger.warning(f"Failed to load datasets from files: {e}")
            if raw_datasets is None:
                raise ValueError("No valid data source found. Both hub and file loading failed.")

    if raw_datasets is None:
        raise ValueError(
            "No data source specified. Please provide either dataset_name or train_file_dir/validation_file_dir.")

    logger.info(f"Final raw datasets: {list(raw_datasets.keys())}")
    for split_name, split_data in raw_datasets.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")

    return raw_datasets


def main():
    # 解析命令行参数和训练配置，确定是否为主进程
    args, training_args, model_args, is_main_process = parse_and_setup()

    # 加载并配置分词器，设置特殊token（eos_token, bos_token, pad_token）
    tokenizer = load_and_prepare_tokenizer(args, training_args, model_args, is_main_process)

    # 加载所需的所有模型：策略模型、参考策略模型、奖励模型、价值模型，以及PEFT配置
    policy, ref_policy, reward_model, value_model, peft_config = load_models(training_args, model_args)
    reward_model.eval()

    # 加载并预处理训练和评估数据集，包括分词和过滤
    train_dataset, eval_dataset = load_and_prepare_datasets(args, training_args, tokenizer, is_main_process)

    # 构建PPO训练器，整合所有必要的组件
    trainer = build_trainer(
        training_args,
        tokenizer,
        policy,
        ref_policy,
        reward_model,
        value_model,
        train_dataset,
        eval_dataset,
        peft_config,
    )

    # 执行训练过程并进行推理生成
    run_training_and_inference(trainer, training_args, is_main_process)


# =========================
# Setup & Args
# =========================

def parse_and_setup():
    """
    解析命令行参数并进行初始化设置。

    Returns:
        tuple: (args, training_args, model_args, is_main_process)
            - args: PPO训练参数
            - training_args: PPO训练配置
            - model_args: 模型配置
            - is_main_process: 是否为主进程标志
    """
    # 在解析参数之前，确保 WORLD_SIZE 环境变量已设置
    # torchrun 会设置 LOCAL_RANK、RANK、WORLD_SIZE 等环境变量
    if "WORLD_SIZE" not in os.environ:
        if "NNODES" in os.environ and "NPROC_PER_NODE" in os.environ:
            os.environ["WORLD_SIZE"] = str(int(os.environ["NNODES"]) * int(os.environ["NPROC_PER_NODE"]))
        elif "NPROC_PER_NODE" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["NPROC_PER_NODE"]
        else:
            os.environ["WORLD_SIZE"] = "1"

    parser = HfArgumentParser(
        (PPOArguments, PPOConfig, ModelConfig)
    )
    args, training_args, model_args = parser.parse_args_into_dataclasses()

    # LOCAL_RANK 是分布式训练框架自动分配的环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    if is_main_process:
        logger.info(f"Parse args: {args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Model args: {model_args}")

    return args, training_args, model_args, is_main_process


# =========================
# Tokenizer
# =========================

def load_and_prepare_tokenizer(args, training_args, model_args, is_main_process):
    """
    加载并规范化 tokenizer，确保 eos / bos / pad 等特殊 token 都已正确设置。

    Args:
        args: PPOArguments参数
        training_args: 训练参数
        model_args: 模型参数
        is_main_process: 是否为主进程

    Returns:
        AutoTokenizer: 配置好的tokenizer
    """
    # 从指定的 tokenizer 路径加载，如果未指定则使用 SFT 模型路径
    tokenizer_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else training_args.sft_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 处理 EOS token - 用于标识序列结束
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        if is_main_process:
            logger.info(f"Add eos_token: {tokenizer.eos_token}")

    # 处理 BOS token - 用于标识序列开始
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        if is_main_process:
            logger.info(f"Add bos_token: {tokenizer.bos_token}")

    # 处理 PAD token - 用于 batch padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        if is_main_process:
            logger.info(f"Add pad_token: {tokenizer.pad_token}")

    return tokenizer


# =========================
# Models
# =========================

def load_models(training_args, model_args):
    """
    加载PPO训练所需的所有模型。

    Args:
        training_args: 训练参数，包含模型路径
        model_args: 模型配置参数

    Returns:
        tuple: (policy, ref_policy, reward_model, value_model, peft_config)
    """
    import torch

    # 使用 torch_dtype 映射
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_args.dtype, torch.float32)

    # 加载价值模型
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        dtype=torch_dtype,
    )

    # 加载奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        dtype=torch_dtype,
    )

    # 加载策略模型
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        dtype=torch_dtype,
    )

    # 获取PEFT配置
    peft_config = get_peft_config(model_args)

    # 如果不使用PEFT，则需要加载参考策略模型
    ref_policy = None
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            dtype=torch_dtype,
        )

    return policy, ref_policy, reward_model, value_model, peft_config


# =========================
# Dataset
# prompt_template样例：
# Conversation(
#     name="qwen",
#     system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
#     messages=[],
#     roles=("user", "assistant"),
#     prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
#     sep="\n",
#     stop_str="<|im_end|>",
# )
# =========================
def load_and_prepare_datasets(args, training_args, tokenizer, is_main_process):
    """
    加载并准备训练和评估数据集。

    完整流程：
    1. 加载原始数据集（train/validation）
    2. 获取对话模板（如qwen模板）
    3. 构建预处理函数（将对话转换为token序列）
    4. 对数据集进行分词和过滤

    Args:
        args: PPOArguments参数，包含数据路径、模板名称等配置
        training_args: 训练参数，包含数据并行处理等配置
        tokenizer: 分词器，用于将文本转换为token
        is_main_process: 是否为主进程（用于分布式训练日志控制）

    Returns:
        tuple: (train_dataset, eval_dataset)
            - train_dataset: 处理后的训练数据集，包含input_ids等字段
            - eval_dataset: 处理后的评估数据集，包含input_ids等字段

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
    """
    # 步骤1：加载原始数据集（train和validation split）
    raw_datasets = load_raw_datasets(args)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    # 步骤2：打印数据集信息（仅主进程打印，避免分布式训练重复打印）
    if is_main_process:
        logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    # 步骤3：获取对话模板（根据args.template_name，如"qwen"）
    # 模板定义了如何将对话格式化为模型输入
    prompt_template = get_conv_template(args.template_name)

    # 步骤4：构建预处理函数
    # 该函数会将JSON格式的对话数据转换为模型可理解的token序列
    preprocess_fn = build_preprocess_function(
        tokenizer, prompt_template, args.max_source_length
    )

    # 步骤5：对训练数据集进行分词和过滤
    # - 使用map函数批量应用预处理函数
    # - 过滤掉无效数据（如input_ids为空的样本）
    train_dataset = tokenize_and_filter(
        train_dataset,
        preprocess_fn,
        training_args,
        is_main_process,
        name="train",
    )

    # 步骤6：对评估数据集进行分词和过滤（同上）
    eval_dataset = tokenize_and_filter(
        eval_dataset,
        preprocess_fn,
        training_args,
        is_main_process,
        name="eval",
    )

    return train_dataset, eval_dataset


def tokenize_and_filter(dataset, preprocess_fn, training_args, is_main_process, name):
    """
    对数据集进行分词和过滤。

    Args:
        dataset: 原始数据集
        preprocess_fn: 预处理函数
        training_args: 训练参数
        is_main_process: 是否为主进程
        name: 数据集名称（用于日志）

    Returns:
        Dataset: 分词并过滤后的数据集
    """
    if not is_main_process:
        return dataset

    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=training_args.dataset_num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing {name} dataset",
    )

    tokenized = tokenized.filter(
        lambda x: len(x["input_ids"]) > 0
    )

    logger.debug(f"{name} samples top3: {tokenized[:3]}")
    return tokenized


def build_preprocess_function(tokenizer, prompt_template, max_source_length):
    """
    构建数据预处理函数。

    Args:
        tokenizer: 分词器
        prompt_template: prompt模板
        max_source_length: 最大源长度（预留参数，用于未来扩展）

    Returns:
        function: 预处理函数
    """
    # max_source_length 参数预留用于未来扩展
    _ = max_source_length
    roles = ["human", "gpt"]

    def preprocess_function(examples):
        """将对话数据转换为模型输入格式"""
        new_examples = {"input_ids": []}
        system_prompts = examples.get("system_prompt", "")

        for i, source in enumerate(examples["conversations"]):
            if len(source) < 2:
                continue

            if source[0].get("from") != roles[0]:
                source = source[1:]

            messages = []
            for j, sentence in enumerate(source):
                if sentence.get("from") == roles[j % 2]:
                    messages.append(sentence["value"])

            if len(messages) < 2 or len(messages) % 2 != 0:
                continue

            history = [
                [messages[k], messages[k + 1]]
                for k in range(0, len(messages), 2)
            ]

            system_prompt = (
                system_prompts[i] if system_prompts else None
            )

            # 使用prompt模板构建对话
            dialog = prompt_template.get_dialog(
                history, system_prompt=system_prompt
            )

            for idx in range(len(dialog) // 2):
                tokenized = tokenizer(
                    dialog[2 * idx], padding=False
                )
                new_examples["input_ids"].append(
                    tokenized["input_ids"]
                )

        return new_examples

    return preprocess_function


# =========================
# Trainer
# =========================

def build_trainer(
        training_args,
        tokenizer,
        policy,
        ref_policy,
        reward_model,
        value_model,
        train_dataset,
        eval_dataset,
        peft_config,
):
    """
    构建PPO训练器，整合所有必要的组件。

    Args:
        training_args: 训练参数配置
        tokenizer: 分词器
        policy: 策略模型
        ref_policy: 参考策略模型
        reward_model: 奖励模型
        value_model: 价值模型
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        peft_config: PEFT配置

    Returns:
        PPOTrainer: 配置好的PPO训练器
    """
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    return trainer


# =========================
# Run
# =========================

def run_training_and_inference(trainer, training_args, is_main_process):
    """
    执行训练过程并进行推理生成。

    Args:
        trainer: PPO训练器
        training_args: 训练参数
        is_main_process: 是否为主进程
    """
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
        trainer.train()

        if is_main_process:
            trainer.save_model(training_args.output_dir)

    trainer.generate_completions()


if __name__ == "__main__":
    main()

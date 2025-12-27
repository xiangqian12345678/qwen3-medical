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

from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass, field
from typing import Optional


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
    """Load multiple datasets from HuggingFace Hub"""
    dataset_names = [name.strip() for name in args.dataset_name.split(",")]
    dataset_configs = [cfg.strip() if cfg else None for cfg in args.dataset_config.split(",")] \
        if args.dataset_config else [None] * len(dataset_names)

    all_train_datasets = []
    all_eval_datasets = []

    for name, cfg in zip(dataset_names, dataset_configs):
        dataset = load_dataset(name, cfg, split=args.dataset_train_split)

        # Split into train and eval
        eval_samples = min(100, len(dataset) // 10)  # 可以根据实际数据量调整
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

        all_train_datasets.append(train_dataset)
        all_eval_datasets.append(eval_dataset)

    # 合并所有数据集
    final_train_dataset = concatenate_datasets(all_train_datasets)
    final_eval_dataset = concatenate_datasets(all_eval_datasets)

    return {"train": final_train_dataset, "validation": final_eval_dataset}


def load_datasets_from_files(args):
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

    dataset = load_dataset(
        'json',
        data_files=data_files,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))

    return {"train": train_dataset, "validation": eval_dataset}


def load_raw_datasets(args):
    """Load raw datasets from hub or files, and merge them if both exist"""
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
    tokenizer = load_and_prepare_tokenizer(training_args, model_args, is_main_process)

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
    parser = HfArgumentParser(
        (PPOArguments, PPOConfig, ModelConfig)
    )
    args, training_args, model_args = parser.parse_args_into_dataclasses()

    # LOCAL_RANK 是分布式训练框架（如 PyTorch DDP、DeepSpeed、Accelerate 等）自动分配的环境变量
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

def load_and_prepare_tokenizer(training_args, model_args, is_main_process):
    """
    加载并规范化 tokenizer，确保 eos / bos / pad 等特殊 token 都已正确设置。

    在实际训练（尤其是 SFT / PPO / GRPO 等 RLHF 场景）中：
    - 不同模型的 tokenizer 对特殊 token 的定义并不统一
    - 有些模型缺失 eos_token / bos_token / pad_token
    - 如果不统一补齐，容易在 batch padding、生成、loss 计算时出错

    该函数的目标：
    1. 从指定路径加载 tokenizer
    2. 检查并补齐 eos_token、bos_token、pad_token
    3. 只在主进程打印日志（兼容分布式训练）
    """

    # 从 SFT 模型路径加载 tokenizer
    # trust_remote_code=True 允许加载自定义 tokenizer（如 Qwen / InternLM 等）
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # =========================
    # 处理 EOS token
    # =========================
    # eos_token 用于标识一句话/序列的结束
    # 如果缺失，在生成任务和 RL 训练中会导致：
    # - 生成无法正常停止
    # - reward / loss 计算异常
    if tokenizer.eos_token_id is None:
        # 优先复用已有的 eos_token，其次使用 sep_token 兜底
        tokenizer.eos_token = tokenizer.eos_token or tokenizer.sep_token

        # 将 eos_token 注册为特殊 token
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})

        if is_main_process:
            logger.info(f"Add eos_token: {tokenizer.eos_token}")

    # =========================
    # 处理 BOS token
    # =========================
    # bos_token 用于标识序列开始
    # 一些 Causal LM（如 LLaMA 系）可能没有显式 bos_token
    if tokenizer.bos_token_id is None:
        # 这里直接复用 eos_token 作为 bos_token
        # 这是一个常见的工程兜底方案，避免 embedding 越界或空值
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id

        if is_main_process:
            logger.info(f"Add bos_token: {tokenizer.bos_token}")

    # =========================
    # 处理 PAD token
    # =========================
    # pad_token 用于 batch padding
    # 在以下场景必不可少：
    # - DataCollatorWithPadding
    # - PPO / GRPO 中的对齐 padding
    # - attention_mask 正确构建
    if tokenizer.pad_token_id is None:
        # 优先使用 unk_token，其次使用 eos_token 兜底
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

        if is_main_process:
            logger.info(f"Add pad_token: {tokenizer.pad_token}")

    # 返回已经规范化的 tokenizer
    return tokenizer


# =========================
# Models
# =========================

def load_models(training_args, model_args):
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    peft_config = get_peft_config(model_args)

    ref_policy = None
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
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
    raw_datasets = load_raw_datasets(args)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    if is_main_process:
        logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    prompt_template = get_conv_template(args.template_name)
    preprocess_fn = build_preprocess_function(
        tokenizer, prompt_template, args.max_source_length
    )

    train_dataset = tokenize_and_filter(
        train_dataset,
        preprocess_fn,
        training_args,
        is_main_process,
        name="train",
    )

    eval_dataset = tokenize_and_filter(
        eval_dataset,
        preprocess_fn,
        training_args,
        is_main_process,
        name="eval",
    )

    return train_dataset, eval_dataset


def tokenize_and_filter(
        dataset, preprocess_fn, training_args, is_main_process, name
):
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


# =========================
# Preprocess
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
# example：
#   {"conversations":[{"from":"human","value":"轻度白内障的临床表现有些什么？"},{"from":"gpt","value":"轻度白内障伴玻璃体混浊"}]}
# =========================

def build_preprocess_function(tokenizer, prompt_template, max_source_length):
    roles = ["human", "gpt"]

    def preprocess_function(examples):
        """
        预处理函数：将对话数据转换为模型输入格式
        
        该函数将多轮对话数据按照指定的prompt模板进行格式化，然后进行分词处理，
        最终生成适合PPO训练的input_ids序列。
        
        Args:
            examples (dict): 包含以下键的字典：
                - "conversations": 对话列表，每个对话包含多个消息对象
                - "system_prompt" (可选): 系统提示词列表
        
        Returns:
            dict: 包含 "input_ids" 键的字典，值为分词后的token序列列表
        
        输入样例:
            examples = {
                "conversations": [
                    [
                        {"from": "human", "value": "轻度白内障的临床表现有些什么？"},
                        {"from": "gpt", "value": "轻度白内障伴玻璃体混浊"}
                    ],
                    [
                        {"from": "human", "value": "如何预防近视？"},
                        {"from": "gpt", "value": "预防近视需要注意用眼卫生..."}
                    ]
                ],
                "system_prompt": ["你是一个专业的医疗助手", ""]
            }
        
        输出样例:
            {
                "input_ids": [
                    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9238, 42805, 7423, 350, 458, 19398, 13, 151645],  # 第一个对话的prompt
                    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 5348, 6263, 12046, 7921, 13, 151645]   # 第二个对话的prompt
                ]
            }
        """
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

            '''
            prompt_template样例：
                Conversation(
                    name="qwen",
                    system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                    messages=[],
                    roles=("user", "assistant"),
                    prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
                    sep="\n",
                    stop_str="<|im_end|>",
                )
            dialog样例：
                [
                    "<|im_start|>system\n你是一个有帮助的助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
                    "你好！很高兴为您服务",
                    "\n<|im_start|>user\n今天天气怎么样<|im_end|>\n<|im_start|>assistant\n",
                    "今天天气很好"
                ]
            '''
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
    return PPOTrainer(
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


# =========================
# Run
# =========================

def run_training_and_inference(
        trainer, training_args, is_main_process
):
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
        trainer.train()

        if is_main_process:
            trainer.save_model(training_args.output_dir)

    trainer.generate_completions()


if __name__ == "__main__":
    main()

import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Literal, Optional, Tuple

import torch
import torch.utils.data
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import LabelSmoother
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False
from template import get_conv_template


@dataclass
class ModelArguments:
    """与要加载 / 微调的模型、配置和 tokenizer 相关的参数"""

    # 预训练模型名称或本地路径
    # 例如: "meta-llama/Llama-2-7b-hf" 或 "./checkpoints/llama"
    model_name_or_path: Optional[str] = field(default=None)

    # 是否以 8bit 量化方式加载模型（bitsandbytes）
    # 优点：显存占用显著降低
    # 缺点：推理 / 训练精度略有损失
    load_in_8bit: bool = field(default=False)

    # 是否以 4bit 量化方式加载模型（QLoRA 场景常用）
    # 通常与 LoRA 一起使用，极大降低显存占用
    load_in_4bit: bool = field(default=False)

    # tokenizer 名称或路径
    # 默认与 model_name_or_path 相同
    tokenizer_name_or_path: Optional[str] = field(default=None)

    # HuggingFace 缓存目录
    # 用于存放下载的模型权重、配置、tokenizer 等
    cache_dir: Optional[str] = field(default=None)

    # 模型版本（git revision / branch / tag / commit）
    # 常见取值："main"、"v1.0"、具体 commit hash
    model_revision: Optional[str] = field(default="main")

    # HuggingFace Hub 访问 token
    # 用于加载私有模型或避免频繁 rate limit
    hf_hub_token: Optional[str] = field(default=None)

    # 是否使用 fast tokenizer（Rust 实现）
    # 优点：速度快
    # 缺点：个别模型（尤其是自定义 tokenizer）可能不兼容
    use_fast_tokenizer: bool = field(default=False)

    # 模型权重的计算精度
    # 常见取值："float16"、"bfloat16"、"float32"
    # 通常与 mixed precision 训练相关
    dtype: Optional[str] = field(default="float16")

    # 设备映射策略
    # "auto"：由 accelerate 自动分配到多卡 / CPU / GPU
    # 也可手动指定，如 {"": 0}
    device_map: Optional[str] = field(default="auto")

    # 是否信任远程仓库中的自定义代码
    # 很多国产模型 / 定制模型必须设为 True
    trust_remote_code: bool = field(default=True)

    # RoPE（旋转位置编码）缩放策略
    # linear：线性缩放
    # dynamic：动态 NTK 缩放
    # 常用于扩展上下文长度（如 4k -> 32k）
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)

    # 是否启用 FlashAttention-2
    # 显著提升长序列训练 / 推理速度，并降低显存
    # 需要：
    # 1. 支持的 GPU（如 A100 / H100 / 部分 RTX）
    # 2. 对应版本的 torch / flash-attn
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )


@dataclass
class DataArguments:
    # 使用 datasets 库加载的「数据集名称」
    # 例如：wikitext, c4, openwebtext
    # 支持多个数据集，用英文逗号分隔：
    # --dataset_name wikitext,c4
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library). "
                    "Support multiple datasets separated by commas."
        }
    )

    # 数据集的「配置名称」
    # 常见于同一个 dataset 下的不同子版本
    # 例如：wikitext-2-raw-v1, wikitext-103-v1
    # 与 dataset_name 一一对应，也支持逗号分隔
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library). "
                    "Support multiple configs separated by commas."
        }
    )

    # 本地训练数据目录
    # 当不使用 HuggingFace datasets，而是自有数据时使用
    # 通常目录下是 json/jsonl/txt 等文件
    train_file_dir: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )

    # 本地验证数据目录
    # 如果不提供，常见做法是从训练集里切分一部分作为验证集
    validation_file_dir: str = field(
        default=None,
        metadata={"help": "Path to the validation data."}
    )

    # 最大训练样本数（调试 / 小规模实验非常有用）
    # 例如：--max_train_samples 10000
    # None 表示使用全部训练数据
    max_train_samples: Optional[int] = field(default=None)

    # 最大验证样本数
    # 用于快速验证或降低评测成本
    max_eval_samples: Optional[int] = field(default=None)

    # 是否覆盖 datasets 的本地缓存
    # 当你修改了数据处理逻辑（tokenize / map）但缓存没变时，需要设为 True
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # 当未显式提供 validation_file_dir 时，
    # 从训练集中切分出多少百分比作为验证集
    # 默认 1 表示 1%
    validation_split_percentage: Optional[int] = field(default=1)

    # 数据预处理（tokenize / map）使用的进程数
    # 一般设为 CPU 核数，能显著加快数据处理
    # 例如：--preprocessing_num_workers 8
    preprocessing_num_workers: Optional[int] = field(default=None)

    # 是否在计算 loss 时忽略 padding token
    # 对于 causal LM / seq2seq 训练几乎必须为 True
    # 否则 pad token 会干扰 loss
    ignore_pad_token_for_loss: bool = field(default=True)


@dataclass
class ScriptArguments:
    # 是否启用 PEFT（Parameter-Efficient Fine-Tuning）
    # True：使用 LoRA / QLoRA 等轻量微调方式
    # False：对模型全参数进行微调（显存、算力消耗巨大）
    use_peft: bool = field(default=True)

    # 是否在训练时对 prompt（输入部分）计算 loss
    # False：只对 assistant 的回答部分计算 loss（推荐，符合对话模型训练范式）
    # True：输入 + 输出都会参与 loss（更像语言模型续写训练）
    train_on_inputs: bool = field(default=False)

    # LoRA 注入的目标模块
    # "all"：自动查找所有 Linear 层并注入（常见默认）
    # 也可以是字符串形式的模块名列表，如 "q_proj,k_proj,v_proj,o_proj"
    target_modules: Optional[str] = field(default="all")

    # LoRA 的秩（rank）
    # 决定 LoRA 的参数量与表达能力
    # 常见取值：4 / 8 / 16
    # rank 越大，效果上限越高，但显存和计算量也会增加
    lora_rank: Optional[int] = field(default=8)

    # LoRA 的 dropout 概率
    # 用于防止过拟合，通常在小数据集场景下很有用
    # 常见取值：0.0 ~ 0.1
    lora_dropout: Optional[float] = field(default=0.05)

    # LoRA 的缩放系数（alpha）
    # 实际生效权重缩放为：alpha / rank
    # alpha 越大，LoRA 更新对原模型的影响越强
    lora_alpha: Optional[float] = field(default=32.0)

    # 除 LoRA 外，还需要"完整保存"的模块
    # 常用于 embedding / lm_head / router 等不适合 LoRA 的模块
    # 例如："embed_tokens,lm_head"
    modules_to_save: Optional[str] = field(default=None)

    # 已训练好的 PEFT 权重路径
    # 用于：
    # 1）继续训练（resume）
    # 2）加载已有 LoRA 权重进行推理
    peft_path: Optional[str] = field(default=None)

    # 是否使用 QLoRA（4bit 量化 + LoRA）
    # True：显存占用极低，适合单卡训练大模型（7B/13B）
    # False：普通 LoRA（fp16 / bf16）
    qlora: bool = field(default=False)

    # 模型最大支持的序列长度（token 数）
    # 会影响：
    # - tokenizer 截断
    # - position embedding
    # - 显存占用
    model_max_length: int = field(default=2048)

    # 对话模板名称
    # 决定 prompt 的格式（system / user / assistant 的组织方式）
    # 常见：vicuna、chatml、alpaca、llama2 等
    template_name: Optional[str] = field(default="vicuna")

    # 是否启用张量并行（Tensor Parallelism）
    # True：将单层参数按维度切分到多张 GPU 上（适合超大模型）
    # False：单卡或普通 DDP 训练
    use_tensor_parallel: bool = field(
    default = False,
    metadata = {"help": "Whether to use tensor parallelism for large models"}

)

def find_all_linear_names(model, int4=False, int8=False):
    """查找模型中所有的线性层名称"""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def save_model(model, tokenizer, output_dir):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
    if all_param > 0:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    else:
        print("No parameters found in the model (possibly using DeepSpeed ZeRO optimization)")

from datasets import load_dataset, concatenate_datasets, Dataset
import os
from glob import glob

def load_hf_datasets(data_args, model_args):
    """
    加载 HuggingFace Hub 数据集（支持多个数据集及配置，并合并训练/验证集）
    """
    hf_train_datasets = []
    hf_validation_datasets = []

    if not data_args.dataset_name:
        return None, None

    dataset_names = [name.strip() for name in data_args.dataset_name.split(',') if name.strip()]

    config_names = []
    if data_args.dataset_config_name:
        config_names = [c.strip() for c in data_args.dataset_config_name.split(',') if c.strip()]

    # 对齐数据集与配置数量
    if not config_names:
        config_names = [None] * len(dataset_names)
    elif len(config_names) < len(dataset_names):
        config_names.extend([config_names[-1]] * (len(dataset_names) - len(config_names)))
    elif len(config_names) > len(dataset_names):
        config_names = config_names[:len(dataset_names)]

    for i, dataset_name in enumerate(dataset_names):
        config_name = config_names[i]
        try:
            logger.info(f"Loading HF dataset {dataset_name} (config={config_name})")
            dataset = load_dataset(dataset_name, config_name, cache_dir=model_args.cache_dir)

            if "train" in dataset:
                hf_train_datasets.append(dataset["train"])
            if "validation" in dataset:
                hf_validation_datasets.append(dataset["validation"])
            elif "test" in dataset:
                hf_validation_datasets.append(dataset["test"])

        except Exception as e:
            logger.warning(f"Failed to load HF dataset {dataset_name}: {e}")

    # 合并训练集和验证集
    hf_train_dataset = concatenate_datasets(hf_train_datasets) if hf_train_datasets else None
    hf_validation_dataset = concatenate_datasets(hf_validation_datasets) if hf_validation_datasets else None

    return hf_train_dataset, hf_validation_dataset


def load_local_datasets(data_args, model_args):
    """
    加载本地 JSON/JSONL 数据集（递归目录搜索），返回 DatasetDict
    """
    data_files = {}

    # 训练文件
    if data_args.train_file_dir and os.path.exists(data_args.train_file_dir):
        train_files = glob(f"{data_args.train_file_dir}/**/*.json", recursive=True) + \
                      glob(f"{data_args.train_file_dir}/**/*.jsonl", recursive=True)
        if train_files:
            data_files["train"] = train_files

    # 验证文件
    if data_args.validation_file_dir and os.path.exists(data_args.validation_file_dir):
        val_files = glob(f"{data_args.validation_file_dir}/**/*.json", recursive=True) + \
                    glob(f"{data_args.validation_file_dir}/**/*.jsonl", recursive=True)
        if val_files:
            data_files["validation"] = val_files

    if not data_files:
        return {}

    try:
        local_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        return local_datasets
    except Exception as e:
        logger.warning(f"Failed to load local datasets: {e}")
        return {}


def load_datasets(data_args, model_args):
    """
    主函数：分别加载 HF 数据集和本地数据集，然后融合，必要时从训练集切分验证集
    """
    # 分别加载
    hf_train_dataset, hf_validation_dataset = load_hf_datasets(data_args, model_args)
    local_datasets = load_local_datasets(data_args, model_args)

    if not hf_train_dataset and not hf_validation_dataset and not local_datasets:
        raise ValueError("No valid datasets found from either HF Hub or local files.")

    # 合并训练集
    train_datasets = []
    if hf_train_dataset: train_datasets.append(hf_train_dataset)
    if "train" in local_datasets: train_datasets.append(local_datasets["train"])

    merged_datasets = {}
    if train_datasets:
        merged_datasets["train"] = train_datasets[0] if len(train_datasets) == 1 else concatenate_datasets(train_datasets)

    # 合并验证集
    val_datasets = []
    if hf_validation_dataset: val_datasets.append(hf_validation_dataset)
    if "validation" in local_datasets: val_datasets.append(local_datasets["validation"])

    if val_datasets:
        merged_datasets["validation"] = val_datasets[0] if len(val_datasets) == 1 else concatenate_datasets(val_datasets)

    # 如果没有验证集，从训练集切分
    if "validation" not in merged_datasets:
        shuffled_train = merged_datasets["train"].shuffle(seed=42)
        split = shuffled_train.train_test_split(test_size=data_args.validation_split_percentage / 100, seed=42)
        merged_datasets["train"] = split["train"]
        merged_datasets["validation"] = split["test"]

    logger.info(f"Final datasets: train={len(merged_datasets['train'])}, validation={len(merged_datasets['validation'])}")
    return merged_datasets



def create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """
    构建一个用于 HuggingFace datasets.map 的预处理函数

    参数说明：
    - tokenizer: tokenizer，用于把文本转成 token ids
    - prompt_template: 对话模板，负责把多轮对话拼成模型可用的 prompt
    - script_args: 训练参数（如 max_length、是否对 input 计算 loss）
    - IGNORE_INDEX: label 中被忽略位置的填充值（通常是 -100）
    """
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """
        对一个 batch 的样本进行预处理
        examples 通常是 datasets 传入的一个 dict，key -> list
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []

        # 只支持 human -> gpt 的对话顺序
        roles = ["human", "gpt"]

        def get_dialog(examples):
            """
            从原始 examples 中解析出标准化的对话格式
            最终 yield 的是：prompt_template 处理后的 dialog（list[str]）
            """
            # system_prompt 可能是一个 list（batch 级别）
            system_prompts = examples.get("system_prompt", "")

            for i, source in enumerate(examples['conversations']):
                system_prompt = ""

                # 至少要有一问一答
                if len(source) < 2:
                    continue

                data_role = source[0].get("from", "")

                # 如果第一条是 system 消息，先取出来
                if data_role == "system":
                    system_prompt = source[0]["value"]
                    source = source[1:]
                    data_role = source[0].get("from", "")

                # 如果第一条不是 human，跳过一条
                # 保证从 human 开始
                if data_role not in roles or data_role != roles[0]:
                    source = source[1:]

                if len(source) < 2:
                    continue

                messages = []

                # 按 human / gpt 交替收集消息
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break

                    # roles[j % 2] 确保角色顺序正确
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])

                # 必须是偶数条（human, gpt 成对）
                if len(messages) % 2 != 0:
                    continue

                # 转成 [[human, gpt], [human, gpt], ...] 的形式
                history_messages = [
                    [messages[k], messages[k + 1]]
                    for k in range(0, len(messages), 2)
                ]

                # 如果当前对话没 system_prompt，用 batch 级的
                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""

                # 通过模板生成最终 dialog
                yield prompt_template.get_dialog(
                    history_messages,
                    system_prompt=system_prompt
                )

        # 对每一个标准化后的 dialog 进行 token 化
        for dialog in get_dialog(examples):
            input_ids = []
            labels = []

            # dialog 结构通常是：
            # [prompt_0, answer_0, prompt_1, answer_1, ...]
            for i in range(len(dialog) // 2):
                # prompt 部分（human）
                source_ids = tokenizer.encode(
                    text=dialog[2 * i],
                    add_special_tokens=(i == 0)  # 只在第一轮加 BOS 等
                )

                # answer 部分（gpt）
                target_ids = tokenizer.encode(
                    text=dialog[2 * i + 1],
                    add_special_tokens=False
                )

                # 按 source / target 比例动态分配 max_length
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                # 截断 source
                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]

                # 截断 target，预留 eos
                if len(target_ids) > max_target_len - 1:
                    target_ids = target_ids[:max_target_len - 1]

                # 避免 source 以 eos 开头
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]

                # 避免 target 以 eos 结尾（后面会手动加）
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]

                # 如果再加一轮就超长，直接停止
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                # 拼接 input_ids
                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]

                # 构建 labels
                if script_args.train_on_inputs:
                    # 对 prompt + answer 都计算 loss
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    # prompt 部分用 IGNORE_INDEX mask 掉
                    labels += (
                            [IGNORE_INDEX] * len(source_ids)
                            + target_ids
                            + [tokenizer.eos_token_id]
                    )

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    return preprocess_function


def filter_empty_labels(example, IGNORE_INDEX):
    """Remove empty labels dataset."""
    return not all(label == IGNORE_INDEX for label in example["labels"])


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

        if free < 2.0:  # 如果可用内存少于2GB
            logger.warning(f"⚠️ GPU {i} 可用内存不足 ({free:.1f}GB)，建议:")
            logger.warning("  1. 使用 --load_in_4bit 启用4bit量化")
            logger.warning("  2. 减小 --per_device_train_batch_size")
            logger.warning("  3. 增加 --gradient_accumulation_steps")
            logger.warning("  4. 减小 --model_max_length")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def get_unwrapped_model(model):
    """获取未包装的原始模型，无论它是否被DDP包装"""
    if hasattr(model, "module"):
        return model.module
    return model


def parse_arguments():
    """解析命令行参数和配置文件"""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    return model_args, data_args, training_args, script_args


def setup_accelerator():
    """初始化Accelerator并设置日志"""
    logger.info(f"🚀 使用Accelerate库进行多GPU训练")
    logger.info("🚀 开始初始化Accelerator...")
    accelerator = Accelerator()
    logger.info("✅ Accelerator初始化完成")

    try:
        logger.info(f"设备: {accelerator.device}")
        logger.info(f"检测到 {accelerator.num_processes} 个进程")
        logger.info(f"当前进程: {accelerator.process_index}")
        logger.info(f"分布式类型: {accelerator.distributed_type}")
    except:
        logger.warning("无法获取完整的Accelerator信息，但这不影响训练")

    return accelerator


def setup_tokenizer(model_args, script_args):
    """
    配置和加载tokenizer
    
    Args:
        model_args: 模型相关参数，包含模型路径、缓存目录等配置
        script_args: 脚本参数，包含模板名称等配置
        
    Returns:
        tuple: (tokenizer, prompt_template) - 配置好的tokenizer和对话模板
    """
    # 构建tokenizer的初始化参数
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,  # 缓存目录，用于存储下载的模型文件
        "use_fast": model_args.use_fast_tokenizer,  # 是否使用fast tokenizer（Rust实现，速度更快）
        "trust_remote_code": model_args.trust_remote_code,  # 是否信任远程代码（国产模型通常需要）
    }

    # 确定tokenizer的路径：优先使用指定的tokenizer路径，否则使用模型路径
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path

    # 从预训练模型加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 获取对话模板，用于定义对话格式和停止符
    prompt_template = get_conv_template(script_args.template_name)

    # 配置特殊token：eos_token（结束符）
    # 如果tokenizer没有设置结束符，使用对话模板的停止字符串作为结束符
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        # tokenizer.eos_token_id 会被自动设置为新增 token 的 ID
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}")

    '''
    # 假设 eos_token = "<|im_end|>"，其 token_id = 151643
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = 151643
    
    # 将 bos_token 设置为相同的值
    tokenizer.bos_token = "<|im_end|>"  # 与 eos_token 相同
    tokenizer.bos_token_id = 151643     # 显式同步 ID
    
    将 bos_token 设置为与 eos_token 相同主要是基于以下几个实用性的考虑：

    1. 简化模型训练
        对于对话式语言模型，开始和结束边界的重要性相对较低：
            模型主要通过上下文理解对话的开始和结束
            而不是依赖特殊的开始/结束标记
            这样可以减少模型需要学习的特殊 token 数量
    2. 避免空值导致的错误
        很多 tokenizer（特别是自定义的对话模型 tokenizer）可能没有明确定义 bos_token ：
    3. 实际训练中的考虑
        在 SFT（Supervised Fine-Tuning）训练中：
            输入数据已经有明确的格式（如模板化的对话）
            模型学习的是对话模式，而不是依赖边界 token
            相同的开始和结束 token 不会影响模型性能
    '''
    # 配置特殊token：bos_token（开始符）
    # 如果没有开始符，使用结束符作为开始符（很多模型这样做）
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}")

    # 配置特殊token：pad_token（填充符）
    # 用于批处理时将不同长度的序列填充到相同长度
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            # 优先使用未知符作为填充符
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # 如果没有未知符，使用结束符作为填充符
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}")

    logger.info("✅ Tokenizer配置完成")
    return tokenizer, prompt_template


def estimate_model_size(model_args, config):
    """估算模型大小（GB）"""
    if hasattr(config, 'num_parameters'):
        return config.num_parameters * 2 / 1024 ** 3  # 假设fp16
    else:
        model_name_lower = model_args.model_name_or_path.lower()
        if '70b' in model_name_lower or '72b' in model_name_lower:
            return 140  # 70B模型大约140GB
        elif '32b' in model_name_lower or '34b' in model_name_lower:
            return 64  # 32B模型大约64GB
        elif '13b' in model_name_lower or '14b' in model_name_lower:
            return 26  # 13B模型大约26GB
        elif '7b' in model_name_lower or '8b' in model_name_lower:
            return 14  # 7B模型大约14GB
        elif '3b' in model_name_lower:
            return 6  # 3B模型大约6GB
        else:
            return 10  # 默认估算


def load_and_configure_model(model_args, script_args, accelerator):
    """
    加载并配置大模型的统一入口函数

    核心目标：
    1. 根据用户配置决定是否使用 4bit / 8bit 量化
    2. 根据 GPU 数量、显存规模、模型大小，自动选择：
       - Tensor Parallel（权重切分）
       - 或 DDP（数据并行）
    3. 可选启用 FlashAttention-2
    4. 在 Tensor Parallel 失败时自动回退到 DDP
    """

    logger.info("🔄 开始加载模型...")

    # ============================================================
    # 1. 确定模型计算精度（fp16 / bf16 / fp32）
    # ============================================================
    # dtype 会影响：
    # - 模型参数 dtype
    # - attention / matmul 的计算精度
    # - 显存占用
    dtype = model_args.dtype

    # ============================================================
    # 2. 构建量化配置（BitsAndBytes）
    # ============================================================
    # 默认不使用量化
    quantization_config = None

    # ---- 4bit 量化（QLoRA 推荐配置）----
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用 4bit 权重量化
            bnb_4bit_compute_dtype=dtype,  # 实际计算仍使用 fp16 / bf16
            bnb_4bit_use_double_quant=True,  # 对量化参数再次量化，进一步省显存
            bnb_4bit_quant_type="nf4"  # NF4：QLoRA 论文推荐的量化方式
        )

    # ---- 8bit 量化 ----
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # ============================================================
    # 3. 构建模型配置加载参数（不加载权重）
    # ============================================================
    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,  # 是否信任模型仓库中的自定义代码
        "cache_dir": model_args.cache_dir,  # 本地缓存目录
        "revision": model_args.model_revision,  # 模型版本（commit / tag）
        "hf_hub_token": model_args.hf_hub_token,  # HuggingFace 私有模型 token
    }

    # ---- FlashAttention-2 开关 ----
    # FlashAttention 必须在 config 阶段启用
    if model_args.flash_attn:
        if is_flash_attn_2_available:
            config_kwargs["use_flash_attention_2"] = True
            logger.info("Using FlashAttention-2 for faster training and inference.")
        else:
            logger.warning("FlashAttention-2 is not installed.")

    # ============================================================
    # 4. 加载模型配置（仅 config，不占 GPU 显存）
    # ============================================================
    # 这里不加载权重，主要用于：
    # - 读取模型结构信息
    # - 后续估算模型大小
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        **config_kwargs
    )

    # ============================================================
    # 5. GPU 环境检测与显存信息打印
    # ============================================================
    total_memory = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"检测到 {num_gpus} 个GPU")

        # 遍历所有 GPU，打印显存使用情况
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            free = gpu_memory - allocated

            total_memory += gpu_memory

            logger.info(
                f"GPU {i}: 总内存={gpu_memory:.1f}GB, "
                f"已分配={allocated:.1f}GB, "
                f"缓存={cached:.1f}GB, "
                f"可用={free:.1f}GB"
            )

        logger.info(f"总GPU内存: {total_memory:.1f}GB")

        # 清理 CUDA cache，避免历史碎片影响大模型加载
        torch.cuda.empty_cache()
        logger.info("已清理GPU缓存")

    # ============================================================
    # 6. 估算模型大小（用于并行策略决策）
    # ============================================================
    # 该函数通常基于：
    # - hidden_size
    # - num_layers
    # - vocab_size
    # - dtype / 量化方式
    estimated_model_size_gb = estimate_model_size(model_args, config)
    logger.info(f"估算模型大小: {estimated_model_size_gb:.1f}GB")

    # ============================================================
    # 7. 决定并行策略（DDP vs Tensor Parallel）
    # ============================================================
    num_gpus = torch.cuda.device_count()

    # 是否是多进程（Accelerate / torchrun）
    is_distributed = accelerator.num_processes > 1

    # 默认允许使用 Tensor Parallel
    use_tensor_parallel = True

    if is_distributed:
        # ---- 多 GPU 场景 ----
        if script_args.use_tensor_parallel and estimated_model_size_gb > 20:
            logger.info(
                f"🔧 使用张量并行策略 (模型大小: {estimated_model_size_gb:.1f}GB)"
            )

            # Tensor Parallel 对 PyTorch 版本有要求
            import pkg_resources
            torch_version = pkg_resources.get_distribution("torch").version

            if pkg_resources.parse_version(torch_version) < pkg_resources.parse_version("2.5.0"):
                logger.warning(
                    f"⚠️ 当前PyTorch版本 {torch_version} 不支持张量并行，需要 >= 2.5.0"
                )
                logger.warning("⚠️ 自动切换到DDP模式")
                use_tensor_parallel = False
            else:
                logger.info(f"✅ PyTorch版本 {torch_version} 支持张量并行")

        else:
            # 模型较小 or 用户未启用 Tensor Parallel
            logger.info(
                f"🔧 使用DDP进行多GPU训练 (模型大小: {estimated_model_size_gb:.1f}GB)"
            )
            use_tensor_parallel = False

    else:
        # ---- 单进程（单卡 / 单机） ----
        logger.info("🔧 单进程训练")

    # ============================================================
    # 8. 构建模型加载参数
    # ============================================================
    model_kwargs = {
        "config": config,  # 模型结构配置
        "dtype": dtype,  # 计算精度
        "trust_remote_code": model_args.trust_remote_code,
        "quantization_config": quantization_config,  # 量化配置（可能为 None）
        "low_cpu_mem_usage": True,  # 减少 CPU 内存峰值
    }

    # ============================================================
    # 9. Tensor Parallel 场景下的 device_map / max_memory
    # ============================================================
    if use_tensor_parallel:
        # device_map="auto" 让 HF 自动切分模型权重到多张 GPU
        model_kwargs["device_map"] = "auto"

        if num_gpus > 1:
            max_memory = {}

            # 为每张 GPU 设置最大可用显存（预留 20% 给 runtime / activation）
            for i in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                total_mem = gpu_props.total_memory
                usable_mem = int(total_mem * 0.8)

                max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

            model_kwargs["max_memory"] = max_memory
            logger.info(
                f"🔧 张量并行配置: device_map=auto, max_memory={max_memory}"
            )
    else:
        logger.info("🔧 DDP配置: 不使用device_map")

    # ============================================================
    # 10. 实际加载模型（带 Tensor Parallel 失败兜底）
    # ============================================================
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("✅ 模型加载完成")

    except OSError as e:
        # Tensor Parallel 在某些模型 / PyTorch 组合下会直接报错
        if "tensor parallel is only supported for" in str(e):
            logger.error(f"❌ 张量并行加载失败: {e}")
            logger.info("🔄 尝试使用DDP模式重新加载...")

            # 移除 Tensor Parallel 相关参数
            model_kwargs.pop("device_map", None)
            model_kwargs.pop("max_memory", None)

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs
            )

            logger.info("✅ 使用DDP模式加载模型成功")
            use_tensor_parallel = False
        else:
            # 其他错误直接抛出
            raise

    # ============================================================
    # 11. 打印模型结构 / 分布信息（用于 sanity check）
    # ============================================================
    display_model_info(model)

    # ============================================================
    # 12. 返回模型及并行策略
    # ============================================================
    return model, use_tensor_parallel


def display_model_info(model):
    """显示模型分布和GPU内存信息"""
    logger.info("📊 模型分布情况:")
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        logger.info("🔧 使用HuggingFace设备映射:")
        for module_name, device in model.hf_device_map.items():
            logger.info(f"  {module_name}: {device}")

        device_count = {}
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_count[device_str] = device_count.get(device_str, 0) + 1

        logger.info("📈 设备使用统计:")
        for device, count in device_count.items():
            logger.info(f"  {device}: {count} 个模块")
    else:
        device_params = {}
        total_params = 0
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_params:
                device_params[device] = {'count': 0, 'size': 0}
            device_params[device]['count'] += 1
            device_params[device]['size'] += param.numel()
            total_params += param.numel()

        logger.info("📈 参数设备分布:")
        if total_params > 0:
            for device, info in device_params.items():
                param_size_gb = info['size'] * 4 / 1024 ** 3
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")
        else:
            logger.info("  未检测到模型参数（可能使用了DeepSpeed ZeRO等优化技术）")

    if torch.cuda.is_available():
        logger.info("💾 GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")


def setup_peft(model, model_args, script_args, training_args):
    """
    配置 PEFT / LoRA 训练逻辑
    - 支持：
        1）从已有 LoRA 权重继续训练
        2）新建 LoRA
        3）4bit / 8bit 量化 + LoRA（QLoRA）
        4）不开 PEFT，走全参数微调
    """
    if script_args.use_peft:
        logger.info("🔧 配置LoRA")

        # ===== 1. 是否从已有 LoRA 权重加载 =====
        # 常见场景：二次微调 / 继续训练
        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(
                model,
                script_args.peft_path,
                is_trainable=True  # 关键：确保 LoRA 参数可训练
            )
        else:
            # ===== 2. 新建 LoRA =====

            # 2.1 如果是 4bit / 8bit 量化模型
            # 必须做额外处理：
            # - 冻结 base model
            # - 处理 LayerNorm / embedding 的 dtype
            # - 配合 gradient checkpointing
            if model_args.load_in_8bit or model_args.load_in_4bit:
                model = prepare_model_for_kbit_training(
                    model,
                    training_args.gradient_checkpointing
                )

            # 2.2 解析 LoRA 作用的目标模块
            # e.g. "q_proj,k_proj,v_proj"
            target_modules = (
                script_args.target_modules.split(',')
                if script_args.target_modules
                else None
            )

            # 特殊值：all
            # 自动扫描模型中所有 Linear 层
            # 常见于 QLoRA / 不想手写模块名
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(
                    model,
                    int4=model_args.load_in_4bit,
                    int8=model_args.load_in_8bit
                )

            # 2.3 一些模块不走 LoRA，但需要保存
            # 典型例子：
            # - lm_head
            # - embedding
            # - 特定 adapter
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')

            # 2.4 构建 LoRA 配置
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # 因果语言模型
                target_modules=target_modules,  # LoRA 注入的模块
                inference_mode=False,  # 训练模式
                r=script_args.lora_rank,  # LoRA rank（低秩维度）
                lora_alpha=script_args.lora_alpha,  # 缩放因子
                lora_dropout=script_args.lora_dropout,  # LoRA dropout
                modules_to_save=modules_to_save
            )

            # 2.5 将 LoRA 注入模型
            model = get_peft_model(model, peft_config)

        # ===== 3. dtype 修正 =====
        # 对所有「可训练参数」统一转成 float32
        # 原因：
        # - 避免 LoRA 参数在 fp16 / bf16 下不稳定
        # - QLoRA 中是非常常见的写法
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

        # 打印可训练参数比例（非常重要的 sanity check）
        model.print_trainable_parameters()

    else:
        # ===== 4. 不使用 PEFT：全参数微调 =====
        logger.info("🔧 全参数训练模式")

        # 整个模型转成 float32
        model = model.float()

        # 打印所有可训练参数
        print_trainable_parameters(model)

    return model


def prepare_datasets(raw_datasets, training_args, data_args, tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """准备和预处理数据集"""
    logger.info("🔄 开始预处理数据集...")
    preprocess_function = create_preprocess_function(tokenizer, prompt_template, script_args, IGNORE_INDEX)

    train_dataset = None
    eval_dataset = None

    # 处理训练数据
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)

        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda example: filter_empty_labels(example, IGNORE_INDEX),
            num_proc=data_args.preprocessing_num_workers
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
        replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                           for label in list(train_dataset[0]['labels'])]
        logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    # 处理验证数据
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets['validation']

        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

        eval_size = len(eval_dataset)
        logger.debug(f"Num eval_samples: {eval_size}")
        if eval_size > 500:
            logger.warning(f"Num eval_samples is large: {eval_size}, "
                           f"training slow, consider reduce it by `--max_eval_samples=50`")

        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda example: filter_empty_labels(example, IGNORE_INDEX),
            num_proc=data_args.preprocessing_num_workers
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    logger.info("✅ 数据集预处理完成")
    return train_dataset, eval_dataset


def prepare_training_components(train_dataset, eval_dataset, model, tokenizer, training_args, IGNORE_INDEX):
    """准备训练组件：数据加载器、优化器、学习率调度器"""
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    )

    train_dataloader = None
    eval_dataloader = None

    if training_args.do_train and train_dataset is not None:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

    if training_args.do_eval and eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    optimizer = None
    lr_scheduler = None

    if training_args.do_train:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps * training_args.warmup_ratio),
            num_training_steps=max_train_steps,
        )

    return train_dataloader, eval_dataloader, optimizer, lr_scheduler


def prepare_accelerator_components(
        accelerator,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        training_args,
        model_is_distributed
):
    """
    使用 HuggingFace Accelerate 对训练相关组件进行统一封装和分布式适配

    这个函数的核心目标：
    - 根据模型是否已经是“分布式加载”的状态，选择不同的 prepare 策略
    - 正确处理 model / optimizer / dataloader / scheduler
    - 避免 Accelerate 对“已分布式模型”重复 wrap 导致的问题
    """

    logger.info("🔄 开始准备训练组件...")

    # ============================================================
    # 情况一：模型已经是分布式的（例如：
    # - 使用 device_map="auto"
    # - 使用 FSDP / DeepSpeed 预先包裹
    # - QLoRA + load_in_4bit + auto device map）
    #
    # 这类模型【不能】再交给 accelerator.prepare(model)
    # 否则会出现：
    # - 参数重复 wrap
    # - device 不一致
    # - 训练直接报错
    # ============================================================
    if model_is_distributed:
        logger.info("🔧 检测到模型已分布在多设备，使用兼容模式")

        # ----------------------------
        # 训练模式
        # ----------------------------
        if training_args.do_train:
            # ⚠️ 注意：这里只 prepare optimizer / dataloader / scheduler
            # 不再 prepare model
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                optimizer,
                train_dataloader,
                lr_scheduler
            )

            # 验证集 dataloader 单独 prepare
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        # ----------------------------
        # 仅评估模式（不训练）
        # ----------------------------
        else:
            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        # 根据是否训练，显式设置模型状态
        # （避免某些 wrapper 场景下状态不一致）
        model.train() if training_args.do_train else model.eval()

        logger.info("✅ 分布式模型训练组件准备完成")

    # ============================================================
    # 情况二：标准模式
    # - 模型是普通 nn.Module
    # - 尚未进行任何分布式封装
    #
    # 这种情况让 Accelerate 接管一切是最安全的
    # ============================================================
    else:
        logger.info("🔧 标准模式，让Accelerate处理所有组件")

        # ----------------------------
        # 训练模式
        # ----------------------------
        if training_args.do_train:
            # accelerator.prepare 会做的事情包括：
            # - model -> DDP / FSDP / DeepSpeed
            # - optimizer 参数同步
            # - dataloader 自动加 DistributedSampler
            # - scheduler 适配多进程步数
            model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model,
                optimizer,
                train_dataloader,
                lr_scheduler
            )

            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        # ----------------------------
        # 仅评估模式
        # ----------------------------
        else:
            # 即使只评估，也要 prepare model
            # 否则在多 GPU 下 forward 会有问题
            model = accelerator.prepare(model)

            if eval_dataloader is not None:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        logger.info("✅ 标准训练组件准备完成")

    # ============================================================
    # 启用梯度检查点（Gradient Checkpointing）
    #
    # 作用：
    # - 用计算换显存
    # - 对大模型 / LoRA / QLoRA 非常关键
    #
    # ⚠️ 必须在 prepare 之后调用：
    # - 否则可能拿到的是未 wrap 的 model
    # ============================================================
    setup_gradient_checkpointing(model, training_args)

    logger.info("🎉 Accelerate多GPU训练配置成功！")

    # 返回所有可能被 accelerator 替换/包装后的对象
    return model, train_dataloader, eval_dataloader, optimizer, lr_scheduler


def setup_gradient_checkpointing(model, training_args):
    """
    设置梯度检查点（Gradient Checkpointing）

    作用：
    - 用“算力换显存”，在前向传播时不保存中间激活
    - 反向传播时重新计算前向，从而显著降低显存占用
    - 常用于大模型 / 长上下文 / LoRA 微调场景

    注意：
    - 启用 gradient checkpointing 时，必须关闭 use_cache
    - 否则会导致显存异常或直接报错
    """

    # 只有在：
    # 1）训练参数中开启了 gradient_checkpointing
    # 2）模型本身支持 gradient checkpointing
    # 才真正启用
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):

        # 启用模型的梯度检查点功能
        model.gradient_checkpointing_enable()

        # 如果模型被 DDP / FSDP / Accelerate 包了一层
        if hasattr(model, "module"):
            # 关闭 KV cache（否则与 gradient checkpointing 冲突）
            model.module.config.use_cache = False
            logger.info("Gradient checkpointing enabled for DDP model.")
        else:
            # 单卡 / 非分布式场景
            model.config.use_cache = False
            logger.info("Gradient checkpointing enabled.")

    else:
        # 未启用 gradient checkpointing 的情况
        # 这里显式把 use_cache 打开，保证推理/训练行为正常
        if hasattr(model, "module"):
            model.module.config.use_cache = True
            logger.info("Gradient checkpointing disabled for DDP model.")
        else:
            model.config.use_cache = True
            logger.info("Gradient checkpointing disabled.")

    # 强制开启 input.requires_grad
    #
    # 目的：
    # - 对于 LoRA / PEFT / 部分冻结参数的训练非常关键
    # - 确保 embedding / input 相关张量能正确参与反向传播
    #
    # 否则可能出现：
    # - loss.backward() 不报错但参数不更新
    # - LoRA 权重梯度为 None
    if hasattr(model, "module"):
        model.module.enable_input_require_grads()
    else:
        model.enable_input_require_grads()


def train_model(
        accelerator,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        training_args,
        model_is_distributed
):
    """
    模型训练主循环

    支持两种训练模式：
        1. model_is_distributed=True  : 手写分布式 / 张量并行训练逻辑
        2. model_is_distributed=False : 使用 Accelerate 标准训练范式

    Args:
        accelerator: Accelerate 对象，负责多卡 / 混合精度 / 梯度同步
        model: 待训练模型
        train_dataloader: 训练数据加载器
        eval_dataloader: 验证数据加载器（可为 None）
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        training_args: 训练超参数（epoch、logging_steps 等）
        model_is_distributed: 是否为特殊分布式模型（如张量并行）
    """
    logger.info("*** 开始训练 ***")

    # 设置模型为训练模式
    model.train()

    # 用于累计 logging_steps 内的 loss
    total_loss = 0

    # 已完成的「优化器更新步数」（不是 dataloader step）
    completed_steps = 0

    # =========================
    # 创建训练进度条
    # =========================
    progress_bar = tqdm(
        range(int(training_args.num_train_epochs * len(train_dataloader))),
        disable=not accelerator.is_local_main_process,  # 只在主进程显示
        desc="Training"
    )

    # =========================
    # Epoch 级别循环
    # =========================
    for epoch in range(int(training_args.num_train_epochs)):
        logger.info(
            f"开始第 {epoch + 1}/{int(training_args.num_train_epochs)} 轮训练"
        )

        # =========================
        # Step 级别循环（batch）
        # =========================
        for step, batch in enumerate(train_dataloader):

            # =========================================================
            # 情况一：自定义分布式 / 张量并行训练逻辑
            # =========================================================
            if model_is_distributed:
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss

                # 梯度累积：loss 需要除以累积步数
                if training_args.gradient_accumulation_steps > 1:
                    loss = loss / training_args.gradient_accumulation_steps

                # 反向传播
                loss.backward()

                # 达到梯度累积步数，才真正更新参数
                if (step + 1) % training_args.gradient_accumulation_steps == 0:

                    # 梯度裁剪，防止梯度爆炸
                    if training_args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            training_args.max_grad_norm
                        )

                    # 参数更新
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    completed_steps += 1
                    progress_bar.update(1)

            # =========================================================
            # 情况二：使用 Accelerate 标准训练流程
            # =========================================================
            else:
                # accelerator.accumulate 会自动处理梯度累积 & 同步
                with accelerator.accumulate(model):

                    # 前向传播
                    outputs = model(**batch)
                    loss = outputs.loss

                    # Accelerate 统一 backward（支持 AMP / 多卡）
                    accelerator.backward(loss)

                    # 只有在真正同步梯度的 step 才裁剪梯度
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            training_args.max_grad_norm
                        )

                    # 参数更新
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # sync_gradients=True 表示完成了一次 optimizer step
                if accelerator.sync_gradients:
                    completed_steps += 1
                    progress_bar.update(1)

            # =========================
            # 累计 loss（用于 logging）
            # =========================
            total_loss += loss.detach().float()

            # =========================
            # 判断是否完成了一个“优化器更新 step”
            # =========================
            if model_is_distributed:
                step_completed = (
                        (step + 1) % training_args.gradient_accumulation_steps == 0
                )
            else:
                step_completed = accelerator.sync_gradients

            if step_completed:

                # =========================
                # 日志打印
                # =========================
                if completed_steps % training_args.logging_steps == 0:
                    avg_loss = total_loss / training_args.logging_steps
                    current_lr = (
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler else training_args.learning_rate
                    )
                    logger.info(
                        f"Step {completed_steps}: "
                        f"loss = {avg_loss:.4f}, "
                        f"lr = {current_lr:.2e}"
                    )
                    total_loss = 0

                # =========================
                # 保存 checkpoint
                # =========================
                if (
                        training_args.save_steps > 0 and
                        completed_steps % training_args.save_steps == 0
                ):
                    output_dir = os.path.join(
                        training_args.output_dir,
                        f"checkpoint-{completed_steps}"
                    )

                    if model_is_distributed:
                        # 分布式模型：手动保存
                        os.makedirs(output_dir, exist_ok=True)
                        model.save_pretrained(output_dir)

                        torch.save(
                            {
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": (
                                    lr_scheduler.state_dict()
                                    if lr_scheduler else None
                                ),
                                "completed_steps": completed_steps,
                            },
                            os.path.join(output_dir, "training_state.pt")
                        )
                    else:
                        # Accelerate 推荐方式
                        accelerator.save_state(output_dir)

                    logger.info(f"保存检查点到: {output_dir}")

                # =========================
                # 定期评估（eval）
                # =========================
                if (
                        training_args.do_eval and
                        training_args.eval_steps > 0 and
                        completed_steps % training_args.eval_steps == 0 and
                        eval_dataloader is not None
                ):
                    model.eval()

                    eval_loss = 0
                    eval_steps = 0

                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            eval_outputs = model(**eval_batch)
                            eval_loss += eval_outputs.loss.detach().float()
                            eval_steps += 1

                    avg_eval_loss = eval_loss / eval_steps

                    # 困惑度（Perplexity）
                    try:
                        perplexity = math.exp(avg_eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    logger.info(
                        f"Step {completed_steps}: "
                        f"eval_loss = {avg_eval_loss:.4f}, "
                        f"perplexity = {perplexity:.2f}"
                    )

                    # 切回训练模式
                    model.train()

    progress_bar.close()
    return completed_steps


def save_final_model(
        accelerator,
        model,
        tokenizer,
        training_args,
        model_is_distributed,
        completed_steps
):
    """
    保存最终训练完成的模型和 tokenizer

    该函数主要处理以下几件事：
    1. 只在主进程打印日志，避免分布式下日志重复
    2. 恢复模型在训练过程中被关闭的配置（如 use_cache）
    3. 正确处理 Accelerate / DDP 包装后的模型保存
    4. 确保多进程之间同步，避免保存冲突
    """

    # 只在主进程打印保存日志（DDP / 多卡下非常重要）
    if accelerator.is_main_process:
        logger.info(f"保存模型到: {training_args.output_dir}")

    # ================================
    # 1. 恢复模型的推理相关配置
    # ================================
    # 训练阶段（尤其是梯度检查点）通常会关闭 use_cache
    # 训练结束后需要恢复，否则会影响推理性能
    unwrapped = get_unwrapped_model(model)
    unwrapped.config.use_cache = True

    # 启用输入梯度（通常用于 LoRA / PEFT 或后续继续微调）
    # 有些训练流程在中途会关闭 input grads
    unwrapped.enable_input_require_grads()

    # ================================
    # 2. 根据是否是分布式模型选择保存方式
    # ================================
    if model_is_distributed:
        # ----------------------------
        # 分布式（DDP）模型保存
        # ----------------------------
        # 某些情况下模型已经是“可直接保存”的状态
        # （例如 FullyShardedDataParallel / 特定 DDP 配置）
        logger.info("🔧 保存分布式模型...")

        # 直接保存模型和 tokenizer
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    else:
        # ----------------------------
        # 标准 Accelerate 保存流程
        # ----------------------------
        # 等待所有进程到达这里，防止主进程提前保存
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # unwrap_model 会去掉 Accelerate / DDP 的外层包装
            # 拿到真正的 HuggingFace 模型对象
            unwrapped_model = accelerator.unwrap_model(model)

            # 使用自定义的 save_model 方法保存
            # （通常内部会处理 safetensors / config / 权重）
            save_model(
                unwrapped_model,
                tokenizer,
                training_args.output_dir
            )

            logger.info("✅ 标准模型保存完成")


def final_evaluation(accelerator, model, eval_dataloader):
    """
    在训练结束后对模型进行最终评估（Final Evaluation）

    主要作用：
    1. 在验证集 / 测试集上计算平均 loss
    2. 根据 loss 计算 perplexity（困惑度）
    3. 只在主进程（main process）上输出评估日志，避免分布式重复打印
    """

    # 如果没有提供评估数据集，则直接跳过最终评估
    if eval_dataloader is not None:
        logger.info("*** 最终评估 ***")

        # 切换模型到评估模式
        # - 关闭 dropout
        # - 固定 BatchNorm / LayerNorm 的行为
        model.eval()

        # 累积评估 loss（注意：这是所有 batch 的 loss 求和）
        eval_loss = 0

        # 统计评估步数（batch 数）
        eval_steps = 0

        # 遍历评估数据集
        for eval_batch in eval_dataloader:
            # 评估阶段不需要反向传播
            # torch.no_grad() 可以：
            # - 关闭梯度计算
            # - 减少显存占用
            # - 提升推理速度
            with torch.no_grad():
                # 前向计算
                # eval_batch 通常包含：
                # - input_ids
                # - attention_mask
                # - labels
                eval_outputs = model(**eval_batch)

                # eval_outputs.loss 是一个标量 tensor
                # detach(): 从计算图中分离，防止梯度跟踪
                # float(): 确保是 FP32，避免混合精度下数值问题
                eval_loss += eval_outputs.loss.detach().float()

                # 评估步数 +1
                eval_steps += 1

        # 计算平均 loss
        # 注意：这里是“batch 平均”，不是“token 平均”
        avg_eval_loss = eval_loss / eval_steps

        # 根据平均 loss 计算困惑度（Perplexity）
        # perplexity = exp(loss)
        # 对语言模型来说，ppl 越低表示模型越好
        try:
            perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            # 当 loss 非常大时，exp(loss) 可能溢出
            # 此时将 perplexity 设为无穷大
            perplexity = float("inf")

        # 只在主进程打印日志
        # 在多卡 / 多进程训练中，避免重复输出
        if accelerator.is_main_process:
            logger.info(
                f"最终评估结果: eval_loss = {avg_eval_loss:.4f}, "
                f"perplexity = {perplexity:.2f}"
            )


def main():
    """主函数"""
    # 解析参数
    model_args, data_args, training_args, script_args = parse_arguments()

    # 初始化Accelerator
    accelerator = setup_accelerator()

    # 设置随机种子
    accelerate_set_seed(training_args.seed)

    # 输出参数信息
    logger.info(f"Model args: {model_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")

    # 配置tokenizer
    tokenizer, prompt_template = setup_tokenizer(model_args, script_args)
    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # 检查和优化内存
    check_and_optimize_memory()

    # 加载和配置模型
    model, model_is_distributed = load_and_configure_model(model_args, script_args, accelerator)

    # 配置PEFT
    model = setup_peft(model, model_args, script_args, training_args)

    # 加载数据集
    logger.info("🔄 开始加载数据集...")
    raw_datasets = load_datasets(data_args, model_args)

    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(raw_datasets, training_args, data_args, tokenizer, prompt_template,
                                                   script_args, IGNORE_INDEX)

    # 准备训练组件
    train_dataloader, eval_dataloader, optimizer, lr_scheduler = prepare_training_components(train_dataset,
                                                                                             eval_dataset, model,
                                                                                             tokenizer, training_args,
                                                                                             IGNORE_INDEX)

    # 使用Accelerate准备所有组件
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = prepare_accelerator_components(accelerator,
                                                                                                       model,
                                                                                                       train_dataloader,
                                                                                                       eval_dataloader,
                                                                                                       optimizer,
                                                                                                       lr_scheduler,
                                                                                                       training_args,
                                                                                                       model_is_distributed)

    # 开始训练
    completed_steps = 0
    if training_args.do_train:
        completed_steps = train_model(accelerator, model, train_dataloader, eval_dataloader, optimizer, lr_scheduler,
                                      training_args, model_is_distributed)
        save_final_model(accelerator, model, tokenizer, training_args, model_is_distributed, completed_steps)

    # 最终评估
    if training_args.do_eval:
        final_evaluation(accelerator, model, eval_dataloader)


if __name__ == "__main__":
    main()

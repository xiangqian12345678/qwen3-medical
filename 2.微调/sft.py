import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
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
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled

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
    """
    与模型 / 配置 / tokenizer 相关的参数
    用于指定：加载哪个模型、如何加载、用什么精度、是否启用高级特性等
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8bit mode or not."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4bit mode or not."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. "
                "If `auto` is passed, the dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Device to map model to. "
                "If `auto` is passed, the device will be selected automatically."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to trust remote code when loading a model from a remote checkpoint."
        },
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable shifted sparse attention (S^2-Attn) proposed by LongLoRA."
        }
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={
            "help": "The alpha parameter to control the noise magnitude in NEFTune. value can be 5."
        }
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
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
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
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
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use peft"}
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "Whether to train on inputs"}
    )
    target_modules: Optional[str] = field(
        default="all"
    )
    lora_rank: Optional[int] = field(
        default=8
    )
    lora_dropout: Optional[float] = field(
        default=0.05
    )
    lora_alpha: Optional[float] = field(
        default=32.0
    )
    modules_to_save: Optional[str] = field(
        default=None
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the peft model"}
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use qlora"}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum model context length. "
                "suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"
            )
        }
    )
    template_name: Optional[str] = field(
        default="vicuna",
        metadata={"help": "The prompt template name."}
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError(
                "You must specify a valid model_max_length >= 60 to run training"
            )


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
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


def find_all_linear_names(peft_model, int4=False, int8=False):
    """找出模型中所有可用于 LoRA 注入的 Linear 层名称"""
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


def check_and_optimize_memory():
    """检查并优化 GPU 显存使用情况"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")
    torch.cuda.empty_cache()
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

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def get_dialog_from_examples(examples, prompt_template, roles):
    """
    从训练样本中提取规范化后的对话文本，并生成最终 prompt

    参数说明：
    - examples: 数据集中的一个 batch，通常包含：
        - examples["conversations"]: 多轮对话列表
        - examples["system_prompt"]（可选）: 与样本一一对应的 system prompt
    - prompt_template: prompt 模板对象，负责将对话历史拼装成模型输入格式
    - roles: 角色顺序定义，例如 ["user", "assistant"]

    产出：
    - yield 经过 prompt_template 处理后的完整对话字符串（generator）

    样例：
    - 输入：
        examples = {
            "conversations": [
                [
                    {"from": "system", "value": "你是一个专业的医疗助手，请提供准确的医疗建议。"},
                    {"from": "human", "value": "治疗阳痿吃什么药呢？"},
                    {"from": "gpt", "value": "男子早泄、早泄病症的再次发生，多由恣情纵欲..."}
                ],
                [
                    {"from": "human", "value": "两只脚明显大小不一样，该怎么办？"},
                    {"from": "gpt", "value": "与走路姿势没有关系的，人的器官，没有完全对称的..."}
                ]
            ],
            "system_prompt": ["默认系统提示", "默认系统提示"]
        }

        roles = ["human", "gpt"]
    - 输出：
        # 第一条对话的输出（包含system prompt）
        "系统：你是一个专业的医疗助手，请提供准确的医疗建议。\n用户：治疗阳痿吃什么药呢？\n助手：男子早泄、早泄病症的再次发生，多由恣情纵欲..."

        # 第二条对话的输出（使用batch级别的system prompt）
        "系统：默认系统提示\n用户：两只脚明显大小不一样，该怎么办？\n助手：与走路姿势没有关系的，人的器官，没有完全对称的..."
    """

    # 取 batch 级别的 system_prompt（如果存在）
    system_prompts = examples.get("system_prompt", "")

    # 遍历 batch 中的每条对话样本
    for i, source in enumerate(examples['conversations']):
        system_prompt = ""

        # 至少需要一问一答，长度不足直接跳过
        if len(source) < 2:
            continue

        # 读取第一条消息的角色
        data_role = source[0].get("from", "")

        # 如果第一条是 system 角色，则单独抽取 system_prompt
        if data_role == "system":
            system_prompt = source[0]["value"]
            source = source[1:]  # 去掉 system 消息
            data_role = source[0].get("from", "")

        # 如果首条消息不是 roles[0]（如 user），则跳过第一条
        # 用于修复数据中不规范的对话起始
        if data_role not in roles or data_role != roles[0]:
            source = source[1:]

        # 再次校验，确保至少还有一问一答
        if len(source) < 2:
            continue

        messages = []

        # 遍历剩余对话内容
        for j, sentence in enumerate(source):
            data_role = sentence.get("from", "")

            # 出现未知角色，直接丢弃整条样本
            if data_role not in roles:
                logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                break

            # 校验角色是否符合 user/assistant 交替顺序
            if data_role == roles[j % 2]:
                messages.append(sentence["value"])

        # 消息数必须为偶数（user/assistant 成对）
        if len(messages) % 2 != 0:
            continue

        # 将消息整理成 [[user, assistant], ...] 的历史对话结构
        history_messages = [
            [messages[k], messages[k + 1]]
            for k in range(0, len(messages), 2)
        ]

        # 如果对话内没有 system_prompt，则尝试使用 batch 级 system_prompt
        if not system_prompt:
            system_prompt = system_prompts[i] if system_prompts else ""

        # 使用模板生成最终 prompt（通常是模型的输入文本）
        yield prompt_template.get_dialog(
            history_messages,
            system_prompt=system_prompt
        )


def preprocess_dialogue_data(dialog, tokenizer, max_length, script_args, IGNORE_INDEX):
    """
    预处理对话数据，将对话格式转换为模型训练所需的input_ids和labels
    
    参数:
        dialog: list - 对话数据，格式为[用户1, 助手1, 用户2, 助手2, ...]
        tokenizer: object - 分词器对象，用于文本编码
        max_length: int - 序列最大长度限制
        script_args: object - 脚本参数配置，包含train_on_inputs等训练设置
        IGNORE_INDEX: int - 忽略标记，用于labels中表示不参与loss计算的位置
    
    返回:
        tuple: (input_ids, labels) - 处理后的输入ID序列和对应的标签序列
    
    使用示例:
    ```python
    dialog = [
        "治疗阳痿吃什么药呢？性生活一直很正常的，但是这段时间感觉性欲变低了，有时勃起都感觉很困难。",
        "男子早泄、早泄病症的再次发生，多由恣情纵欲，或青年误犯性交，至命门火衰，精气虚寒；或因湿热下注，宗筋弛而痿的。",
        "需要做什么检查？",
        "建议到医院做相关检查，包括血液检查、激素水平检测等。"
    ]
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    max_length = 512
    script_args = argparse.Namespace()
    script_args.train_on_inputs = False
    IGNORE_INDEX = -100
    
    # 调用函数
    input_ids, labels = preprocess_dialogue_data(dialog, tokenizer, max_length, script_args, IGNORE_INDEX)
    
    # 预期输出：
    # input_ids: [用户1的token序列] + [助手1的token序列] + [eos_token] + [用户2的token序列] + [助手2的token序列] + [eos_token]
    # labels: [-100, -100, -100, ..., 助手1的token序列, eos_token, -100, -100, ..., 助手2的token序列, eos_token]
    ```
    """
    input_ids, labels = [], []  # 初始化输入ID和标签列表

    # 遍历对话，每次处理一组用户-助手对话
    for i in range(len(dialog) // 2):
        # 获取用户输入和助手回复的文本
        user_text = dialog[2 * i]  # 用户输入
        assistant_text = dialog[2 * i + 1]  # 助手回复

        # 使用分词器编码文本
        # 用户输入：第一个对话时添加特殊token（如bos_token），后续不添加
        source_ids = tokenizer.encode(text=user_text, add_special_tokens=(i == 0))
        # 助手回复：不添加特殊token
        target_ids = tokenizer.encode(text=assistant_text, add_special_tokens=False)

        # 计算总长度，用于按比例分配最大长度
        total_len = len(source_ids) + len(target_ids)
        # 根据源文本长度占比计算最大源文本长度
        max_source_len = int(max_length * (len(source_ids) / total_len))
        # 根据目标文本长度占比计算最大目标文本长度
        max_target_len = int(max_length * (len(target_ids) / total_len))

        # 截断过长的序列
        if len(source_ids) > max_source_len:
            source_ids = source_ids[:max_source_len]
        if len(target_ids) > max_target_len - 1:  # -1为后面的eos_token预留空间
            target_ids = target_ids[:max_target_len - 1]

        # 处理特殊token，避免重复
        # 如果源文本以eos_token开头，移除它
        if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
            source_ids = source_ids[1:]
        # 如果目标文本以eos_token结尾，移除它（后面会统一添加）
        if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
            target_ids = target_ids[:-1]

        # 检查是否会超出最大长度限制
        if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
            break

        # 构建输入序列：源文本 + 目标文本 + eos_token
        input_ids += source_ids + target_ids + [tokenizer.eos_token_id]

        # 构建标签序列
        if script_args.train_on_inputs:
            # 如果训练时包含输入，则全部token都参与loss计算
            labels += source_ids + target_ids + [tokenizer.eos_token_id]
        else:
            # 如果不训练输入，则源文本部分设为IGNORE_INDEX，只计算目标部分的loss
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_function(examples, tokenizer, max_length, script_args, IGNORE_INDEX, prompt_template):
    """
    数据预处理函数 - 将原始对话数据转换为模型训练所需的格式

    功能说明：
    - 批量处理对话样本，将其转换为模型输入格式
    - 生成input_ids（token序列）、attention_mask（注意力掩码）、labels（标签）三个关键字段
    - 支持多轮对话处理，自动处理human/gpt角色交替

    参数说明：
    - examples: 原始对话数据样本列表
    - tokenizer: 分词器对象，用于文本tokenization
    - max_length: 序列最大长度限制
    - script_args: 脚本参数配置字典
    - IGNORE_INDEX: 忽略索引值，用于mask不需要计算loss的token
    - prompt_template: 提示词模板，用于格式化对话

    输入样例：
    examples = [
        {
            "conversations": [
                {"from": "human", "value": "你好，请问头疼怎么办？"},
                {"from": "gpt", "value": "头疼可能是由多种原因引起的，建议您先休息，如果持续不缓解请及时就医。"}
            ]
        }
    ]
    tokenizer = AutoTokenizer.from_pretrained("qwen-model")
    max_length = 512
    script_args = {"padding": "max_length", "truncation": True}
    IGNORE_INDEX = -100
    prompt_template = "{role}: {content}\n"

    输出样例：
    {
        "input_ids": [[1, 234, 567, 890, 123, 456, 789, 2]],  # token序列
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],           # 注意力掩码
        "labels": [[-100, -100, 567, 890, 123, 456, 789, 2]]  # 标签（human部分为-100）
    }
    """
    input_ids_list = []
    attention_mask_list = []
    targets_list = []
    roles = ["human", "gpt"]

    for dialog in get_dialog_from_examples(examples, prompt_template, roles):
        input_ids, labels = preprocess_dialogue_data(
            dialog, tokenizer, max_length, script_args, IGNORE_INDEX
        )
        input_ids_list.append(input_ids)
        attention_mask_list.append([1] * len(input_ids))
        targets_list.append(labels)

    return dict(
        input_ids=input_ids_list,
        attention_mask=attention_mask_list,
        labels=targets_list,
    )


def filter_empty_labels(example):
    """
    Remove empty labels dataset.
    # 示例数据
        dataset = [
            {"text": "Hello", "labels": [1, 2, 3]},        # 有效样本
            {"text": "World", "labels": [-100, -100, -100]}, # 无效样本（全padding）
            {"text": "Test", "labels": [5, -100, 7]},       # 有效样本（部分有效）
        ]
    filtered_dataset = filter(filter_empty_labels, dataset)
    # 结果：{"text": "Hello", "labels": [1, 2, 3]} 和 {"text": "Test", "labels": [5, -100, 7]}
    """
    return not all(label == -100 for label in example["labels"])


def setup_tokenizer(model_args, script_args):
    """
    参数设置：构建tokenizer初始化所需的参数字典
    路径确定：智能选择tokenizer路径，优先使用指定路径，否则使用模型路径
    加载tokenizer：从预训练模型加载tokenizer
    获取对话模板：用于格式化输入文本
    特殊token设置：
    EOS（结束符）：如果不存在则使用模板的停止字符串
    BOS（开始符）：如果不存在则使用结束符
    PAD（填充符）：优先使用未知token，否则使用结束符
    调试输出：记录tokenizer信息并返回
    """
    # 构建tokenizer的初始化参数字典
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,  # 缓存目录，避免重复下载模型
        "use_fast": model_args.use_fast_tokenizer,  # 是否使用快速tokenizer
        "trust_remote_code": model_args.trust_remote_code,  # 是否信任远程代码
    }

    # 确定tokenizer的路径：优先使用指定的tokenizer路径，否则使用模型路径
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path

    # 从预训练模型加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs
    )

    # 获取对话模板，用于格式化输入文本
    prompt_template = get_conv_template(script_args.template_name)

    '''
    tokenizer.eos_token 和 tokenizer.eos_token_id 是两个相关但不同的属性：

    1.tokenizer.eos_token
        类型：字符串 (str)
        内容：结束符的实际文本，例如 "<|endoftext|>" 、 "</s>" 或 "\n"
        用途：
            在文本处理和字符串操作中使用
            用于生成对话时的文本格式化
            在数据预处理时拼接文本
    2.tokenizer.eos_token_id
        类型：整数 (int)
        内容：结束符在词汇表中的索引编号，例如 2 、 151643 等
        用途：
            在模型输入的token序列中使用
            用于模型的张量计算和推理
            在训练时标识序列结束位置
    '''
    # 检查并设置结束符（eos_token）
    if tokenizer.eos_token_id is None:
        # 如果没有结束符，使用模板的停止字符串作为结束符
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(
            f"Add eos_token: {tokenizer.eos_token}, "
            f"eos_token_id: {tokenizer.eos_token_id}"
        )

    # 检查并设置开始符（bos_token）
    if tokenizer.bos_token_id is None:
        # 如果没有开始符，使用结束符作为开始符
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(
            f"Add bos_token: {tokenizer.bos_token}, "
            f"bos_token_id: {tokenizer.bos_token_id}"
        )

    # 检查并设置填充符（pad_token）
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            # 优先使用未知token作为填充符
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # 否则使用结束符作为填充符
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Add pad_token: {tokenizer.pad_token}, "
            f"pad_token_id: {tokenizer.pad_token_id}"
        )

    # 输出调试信息
    logger.debug(f"Tokenizer: {tokenizer}")
    return tokenizer, prompt_template

def load_hf_datasets(data_args, model_args):
    """
    从 HuggingFace 数据集加载数据，并处理验证集
    """
    from datasets import load_dataset

    raw_datasets = {}
    if not data_args.dataset_name:
        return raw_datasets, False

    has_data_source = False

    dataset_names = [name.strip() for name in data_args.dataset_name.split(',') if name.strip()]
    dataset_configs = []
    if data_args.dataset_config_name:
        dataset_configs = [
            None if (c := config.strip()) in ("", "None", "none") else c
            for config in data_args.dataset_config_name.split(',')
        ]

    # 补全配置列表长度
    while len(dataset_configs) < len(dataset_names):
        dataset_configs.append(None)
    dataset_configs = dataset_configs[:len(dataset_names)]

    for i, dataset_name in enumerate(dataset_names):
        dataset_config = dataset_configs[i]
        try:
            logger.info(f"加载 HuggingFace 数据集 '{dataset_name}' (配置: {dataset_config})")
            named_datasets = load_dataset(dataset_name, dataset_config, cache_dir=model_args.cache_dir)
            if not named_datasets:
                logger.warning(f"数据集 '{dataset_name}' 加载成功但为空")
                continue
            has_data_source = True

            # 处理验证集
            if "validation" not in named_datasets:
                if "train" in named_datasets and len(named_datasets["train"]) > 0:
                    shuffled_train = named_datasets["train"].shuffle(seed=42)
                    split = shuffled_train.train_test_split(
                        test_size=data_args.validation_split_percentage / 100, seed=42
                    )
                    named_datasets["train"] = split["train"]
                    named_datasets["validation"] = split["test"]
                else:
                    logger.warning(f"数据集 '{dataset_name}' 没有训练数据，无法分割验证集")

            # 合并数据集
            for key in named_datasets:
                if key in named_datasets and len(named_datasets[key]) > 0:
                    if key in raw_datasets:
                        raw_datasets[key] = raw_datasets[key].concatenate(named_datasets[key])
                    else:
                        raw_datasets[key] = named_datasets[key]

        except Exception as e:
            logger.error(f"加载 HuggingFace 数据集 '{dataset_name}' 失败: {str(e)}")
            logger.warning(f"跳过数据集 '{dataset_name}'，继续加载其他数据集")

    return raw_datasets, has_data_source


def load_local_datasets(data_args, model_args):
    """
    从本地 JSON/JSONL 文件加载数据，并处理验证集
    """
    from datasets import load_dataset
    import os
    from glob import glob

    raw_datasets = {}
    has_data_source = False
    data_files = {}

    # 训练文件
    train_files = []
    if data_args.train_file_dir and os.path.exists(data_args.train_file_dir):
        train_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + \
                      glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
        if train_files:
            has_data_source = True
    elif data_args.train_file_dir:
        logger.warning(f"训练文件目录不存在: {data_args.train_file_dir}")

    # 验证文件
    val_files = []
    if data_args.validation_file_dir and os.path.exists(data_args.validation_file_dir):
        val_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + \
                    glob(f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
        if val_files:
            has_data_source = True
    elif data_args.validation_file_dir:
        logger.warning(f"验证文件目录不存在: {data_args.validation_file_dir}")

    if train_files or val_files:
        try:
            if train_files:
                data_files["train"] = train_files
            if val_files:
                data_files["validation"] = val_files

            logger.info("加载本地文件数据集...")
            file_datasets = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
            if not file_datasets:
                logger.warning("本地文件加载成功但数据集为空")
            else:
                has_data_source = True

            # 仅训练文件时，分割验证集
            if train_files and not val_files and "validation" not in file_datasets:
                if "train" in file_datasets and len(file_datasets["train"]) > 0:
                    shuffled_train = file_datasets["train"].shuffle(seed=42)
                    split = shuffled_train.train_test_split(
                        test_size=float(data_args.validation_split_percentage / 100), seed=42
                    )
                    file_datasets["train"] = split["train"]
                    file_datasets["validation"] = split["test"]
                else:
                    logger.warning("训练文件为空，无法分割验证集")

            # 合并数据集
            for key in file_datasets:
                if key in file_datasets and len(file_datasets[key]) > 0:
                    if key in raw_datasets:
                        raw_datasets[key] = raw_datasets[key].concatenate(file_datasets[key])
                    else:
                        raw_datasets[key] = file_datasets[key]

        except Exception as e:
            logger.error(f"加载本地文件数据集失败: {str(e)}")

    return raw_datasets, has_data_source


def load_datasets(data_args, model_args):
    """
    加载数据集（HuggingFace + 本地文件），并合并
    """
    raw_datasets = {}
    overall_has_data = False

    # HuggingFace 数据集
    hf_datasets, hf_has_data = load_hf_datasets(data_args, model_args)
    if hf_has_data:
        overall_has_data = True
        for k, v in hf_datasets.items():
            raw_datasets[k] = v

    # 本地文件
    local_datasets, local_has_data = load_local_datasets(data_args, model_args)
    if local_has_data:
        overall_has_data = True
        for k, v in local_datasets.items():
            if k in raw_datasets:
                raw_datasets[k] = raw_datasets[k].concatenate(v)
            else:
                raw_datasets[k] = v

    # 验证最终数据集
    if not overall_has_data or "train" not in raw_datasets or len(raw_datasets["train"]) == 0:
        raise ValueError("未能加载有效训练数据集，请检查配置或数据源")

    # 打印统计信息
    logger.info("=" * 50)
    logger.info("数据集加载完成，统计信息:")
    for key, dataset in raw_datasets.items():
        logger.info(f"  {key}: {len(dataset)} 条数据")
    logger.info("=" * 50)

    return raw_datasets



def process_train_dataset(train_dataset, data_args, training_args, is_main_process, tokenizer, script_args,
                          IGNORE_INDEX, prompt_template):
    """处理训练数据集"""
    # 获取训练数据集的总样本数
    max_train_samples = len(train_dataset)

    # 如果用户设置了最大训练样本数限制，则截取数据集
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        # 取数据集长度和设置的最大值中的较小值
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        # 只选择前max_train_samples个样本
        train_dataset = train_dataset.select(range(max_train_samples))

    # 如果是主进程，打印第一个训练样本的示例（用于调试）
    if is_main_process:
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

    # 使用主进程优先的上下文管理器进行数据集tokenization
    # 这确保在分布式训练中，只有主进程执行预处理，其他进程等待
    with training_args.main_process_first(desc="Train dataset tokenization"):
        # 对训练数据集进行映射处理，将文本转换为token IDs
        tokenized_dataset = train_dataset.map(
            # lambda函数：对每个样本应用预处理函数
            lambda examples: preprocess_function(examples, tokenizer, script_args.model_max_length, script_args,
                                                 IGNORE_INDEX, prompt_template),
            batched=True,  # 批量处理以提高效率
            num_proc=1,  # 使用1个进程进行处理
            remove_columns=train_dataset.column_names,  # 移除原始列，只保留tokenization后的结果
            load_from_cache_file=not data_args.overwrite_cache,  # 根据配置决定是否加载缓存
            desc="Running tokenizer on dataset" if is_main_process else None,  # 进度描述（仅主进程显示）
        )

        # 过滤掉标签为空的样本
        train_dataset = tokenized_dataset.filter(filter_empty_labels, num_proc=1)

        # 如果是主进程，输出调试信息
        if is_main_process:
            # 打印最终的训练样本数量
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")

            # 解码并打印第一个样本的input_ids（输入序列）
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")

            # 处理labels序列：将IGNORE_INDEX替换为pad_token_id以便解码
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                               for label in list(train_dataset[0]['labels'])]
            # 解码并打印第一个样本的labels（标签序列）
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    # 返回处理后的训练数据集和实际使用的样本数量
    return train_dataset, max_train_samples


def process_eval_dataset(eval_dataset, data_args, training_args, tokenizer, script_args, IGNORE_INDEX, prompt_template):
    """处理评估数据集"""
    # 获取原始评估数据集的总样本数
    max_eval_samples = len(eval_dataset)

    # 如果设置了最大评估样本数限制，则截取数据集
    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # 获取处理后的评估数据集大小
    eval_size = len(eval_dataset)

    # 记录评估样本数量信息
    logger.debug(f"Num eval_samples: {eval_size}")

    # 如果评估样本过多（超过500个），发出警告提示用户考虑减少样本数以提高训练速度
    if eval_size > 500:
        logger.warning(f"Num eval_samples is large: {eval_size}, "
                       f"training slow, consider reduce it by `--max_eval_samples=50`")

    # 输出第一个评估样本的原始数据格式，用于调试
    logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")

    # 对评估数据集进行预处理
    eval_dataset = eval_dataset.map(
        # 预处理函数：对每个样本进行tokenization，格式化为模型输入格式
        lambda examples: preprocess_function(examples, tokenizer, script_args.model_max_length, script_args,
                                             IGNORE_INDEX, prompt_template),
        batched=True,  # 批量处理提高效率
        num_proc=data_args.preprocessing_num_workers,  # 多进程处理
        remove_columns=eval_dataset.column_names,  # 移除原始列，只保留处理后的tokenized数据
        load_from_cache_file=not data_args.overwrite_cache,  # 是否从缓存加载
        desc="Running tokenizer on validation dataset",  # 进度描述
    )

    # 过滤掉空标签的样本（可能由于预处理导致标签为空）
    eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)

    # 记录过滤后的最终评估样本数量
    logger.debug(f"Num eval_samples: {len(eval_dataset)}")

    # 输出第一个tokenized后的样本解码结果，用于验证预处理是否正确
    logger.debug("Tokenized eval example:")
    logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # 返回处理后的评估数据集和最大样本数
    return eval_dataset, max_eval_samples


def setup_quantization_config(model_args, script_args, dtype, training_args):
    """设置量化配置

    量化是一种模型压缩技术，通过降低模型参数的精度来减少内存占用和计算开销。
    支持4位量化（QLoRA）和8位量化两种模式。

    Args:
        model_args: 模型参数对象，包含量化相关的配置选项
        script_args: 脚本参数对象，包含QLoRA等高级特性开关
        dtype: PyTorch数据类型（如torch.float16或torch.bfloat16）
        training_args: 训练参数对象（当前未使用但保留接口）

    Returns:
        tuple: (quantization_config, load_in_4bit, load_in_8bit)
            - quantization_config: BitsAndBytesConfig对象或None
            - load_in_4bit: 是否启用4位量化的布尔值
            - load_in_8bit: 是否启用8位量化的布尔值

    Raises:
        ValueError: 当4位和8位量化同时启用，或与DeepSpeed ZeRO-3冲突时
    """
    # 从模型参数中提取量化配置选项
    load_in_4bit = model_args.load_in_4bit  # 4位量化标志
    load_in_8bit = model_args.load_in_8bit  # 8位量化标志
    quantization_config = None  # 初始化量化配置为None

    # 检查互斥性：4位和8位量化不能同时启用
    if load_in_4bit and load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
    # 如果启用了任意一种量化模式
    elif load_in_8bit or load_in_4bit:
        # 记录量化配置信息
        logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")

        # 检查与DeepSpeed ZeRO-3的兼容性
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        # 8位量化配置
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 4位量化配置
        elif load_in_4bit:
            if script_args.qlora:
                # QLoRA模式的4位量化：启用所有优化特性
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 启用4位量化
                    bnb_4bit_compute_dtype=dtype,  # 计算时使用的数据类型
                    bnb_4bit_use_double_quant=True,  # 启用双重量化（进一步压缩）
                    bnb_4bit_quant_type="nf4"  # 使用NF4量化类型（4位归一化浮点）
                )
            else:
                # 标准4位量化模式
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 启用4位量化
                    bnb_4bit_compute_dtype=dtype,  # 计算时使用的数据类型
                )

    return quantization_config, load_in_4bit, load_in_8bit


def setup_model_kwargs(model_args, config, config_kwargs, dtype, quantization_config, training_args=None):
    """设置模型加载参数

    Args:
        model_args: 模型参数对象，包含设备映射、信任远程代码等配置
        config: 模型配置对象
        config_kwargs: 配置关键字参数
        dtype: PyTorch数据类型（如torch.float16, torch.bfloat16等）
        quantization_config: 量化配置对象

    Returns:
        dict: 包含所有模型加载参数的字典
    """
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()

    # 基础模型参数配置（不包含device_map，稍后设置）
    model_kwargs = {
        "config": config,  # 模型配置对象
        "dtype": dtype,  # 指定模型的数据类型，影响精度和内存使用
        "trust_remote_code": model_args.trust_remote_code,  # 是否信任远程代码（用于加载自定义模型）
        "quantization_config": quantization_config,  # 量化配置，用于减少内存占用
        "low_cpu_mem_usage": True,  # 启用低CPU内存使用模式
    }

    # 检查是否使用DeepSpeed ZeRO-3
    using_deepspeed_zero3 = False
    if training_args and training_args.deepspeed is not None:
        # 导入并检查DeepSpeed配置
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled
            using_deepspeed_zero3 = is_deepspeed_zero3_enabled()
        except ImportError:
            pass

    # 设置设备映射策略
    if using_deepspeed_zero3:
        # DeepSpeed ZeRO-3不支持device_map，由DeepSpeed自动管理
        logger.info("🔧 检测到DeepSpeed ZeRO-3，将让DeepSpeed自动管理设备映射")
        model_kwargs["device_map"] = None
    elif model_args.device_map == 'auto':
        if num_gpus > 1:
            # 保持自动设备映射
            model_kwargs["device_map"] = "auto"

            # 为每个GPU设置最大内存限制
            max_memory = {}
            for i in range(num_gpus):
                # 获取GPU属性信息
                gpu_props = torch.cuda.get_device_properties(i)
                total_mem = gpu_props.total_memory  # GPU总内存

                # 设置可用内存为总内存的80%，预留20%作为缓冲
                usable_mem = int(total_mem * 0.8)

                # 将内存大小转换为GiB单位并添加到字典中
                max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

            # 将最大内存配置添加到模型参数中
            model_kwargs["max_memory"] = max_memory
        else:
            # 单GPU情况，不设置device_map让其自动使用GPU:0
            model_kwargs["device_map"] = None
    else:
        # 使用用户指定的device_map
        model_kwargs["device_map"] = model_args.device_map

    return model_kwargs


def log_model_info(model):
    """记录模型信息"""
    logger.info("📊 模型分布情况:")

    # 检查模型是否使用HuggingFace的设备映射（通常用于多GPU分布式训练）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        logger.info("🔧 使用HuggingFace设备映射:")

        # 遍历并记录每个模型模块所在的设备
        for module_name, device in model.hf_device_map.items():
            logger.info(f"  {module_name}: {device}")

        # 统计每个设备上分配的模块数量
        device_count = {}
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_count[device_str] = device_count.get(device_str, 0) + 1

        logger.info("📈 设备使用统计:")
        for device, count in device_count.items():
            logger.info(f"  {device}: {count} 个模块")
    else:
        # 如果没有设备映射，则手动统计参数在各设备上的分布
        device_params = {}
        total_params = 0

        # 遍历模型的所有命名参数
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_params:
                device_params[device] = {'count': 0, 'size': 0}
            device_params[device]['count'] += 1  # 参数组数量
            device_params[device]['size'] += param.numel()  # 参数总数
            total_params += param.numel()

        logger.info("📈 参数设备分布:")
        if total_params > 0:
            for device, info in device_params.items():
                # 计算参数大小（假设float32，每个参数4字节）
                param_size_gb = info['size'] * 4 / 1024 ** 3
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")
        else:
            logger.info("  未检测到模型参数（可能使用了DeepSpeed ZeRO等优化技术）")

    # 如果CUDA可用，显示GPU内存使用情况
    if torch.cuda.is_available():
        logger.info("💾 GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # 已分配内存
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3  # 缓存内存
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # GPU总内存
            logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")


def setup_neftune(model, model_args):
    """
    设置NEFTune (Noisy Embedding Instruction Fine-Tuning)

    NEFTune的数学原理：
    ==================

    1. 核心思想：
    NEFTune通过在输入嵌入向量中添加噪声来增强模型对指令的鲁棒性和泛化能力。
    这种方法基于以下假设：在训练过程中引入适当的噪声可以防止模型过拟合，
    并提高其对输入扰动的容忍度。

    2. 数学公式：
    给定原始嵌入向量 E ∈ R^(d×V)（其中d是嵌入维度，V是词汇表大小），
    对于输入token索引i，添加噪声后的嵌入向量：

    Ẽ_i = E_i + ε_i

    其中噪声 ε_i ~ Uniform(-α/√(d×V), α/√(d×V))

    α 是控制噪声强度的超参数（neft_alpha）
    √(d×V) 是归一化因子，确保噪声幅度与嵌入层大小成反比

    3. 理论依据：
    - 噪声注入等价于L2正则化的一种形式，有助于平滑损失函数
    - 类似于数据增强，在表示空间创建更多样化的训练样本
    - 鼓励模型学习对输入扰动不敏感的鲁棒特征表示

    4. 实验发现：
    原论文表明，在指令微调阶段添加噪声可以显著提升模型在
    unseen指令上的表现，特别是在小样本学习场景中。

    参数说明：
    - model: 要进行NEFTune的模型
    - model_args: 包含neft_alpha参数的模型参数对象

    Returns:
        None (直接修改模型的前向传播方法)
    """
    # 检查是否启用了NEFTune（neft_alpha > 0）
    if model_args.neft_alpha > 0:
        # 获取模型的输入嵌入层
        input_embed = model.get_input_embeddings()

        # 确保输入嵌入层是标准的nn.Embedding类型
        if isinstance(input_embed, torch.nn.Embedding):
            def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                """
                带噪声的嵌入前向传播函数

                Args:
                    self: 嵌入层实例
                    x: 输入token索引张量 [batch_size, seq_len]

                Returns:
                    添加了均匀分布噪声的嵌入向量 [batch_size, seq_len, embedding_dim]
                """
                # 首先计算原始嵌入向量
                embeddings = input_embed.__class__.forward(self, x)

                # 计算嵌入层的总维度数（词汇表大小 × 嵌入维度）
                # 这个值用作噪声幅度的归一化因子
                dims = self.num_embeddings * self.embedding_dim

                # 计算噪声幅度：α / √(词汇表大小 × 嵌入维度)
                # 这样可以确保噪声幅度与嵌入层规模成反比，避免在大模型中噪声过大
                mag_norm = model_args.neft_alpha / (dims ** 0.5)

                # 生成与嵌入向量同形状的均匀分布噪声并添加到原始嵌入中
                # 噪声范围：[-mag_norm, mag_norm]
                embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

                return embeddings

            # 使用MethodType将噪声前向传播函数绑定到嵌入层实例
            # 这样就替换了原始的forward方法，在每次前向传播时都会添加噪声
            input_embed.forward = MethodType(noisy_forward, input_embed)

            # 记录NEFTune已启用，显示噪声强度参数
            logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
        else:
            # 如果嵌入层不是标准nn.Embedding，发出警告
            # 某些模型可能使用自定义的嵌入层实现
            logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")


def setup_model_patches(model, config, training_args):
    """设置模型补丁 - 处理不同模型类型的特殊配置和训练优化"""

    # 1. 处理 ChatGLM 和 InternLM2 模型的输出层映射
    if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
        # 将 lm_head 映射到 transformer.output_layer
        setattr(model, "lm_head", model.transformer.output_layer)
        # 保存时忽略 lm_head.weight，避免冗余
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # 2. 处理 Mixtral 模型的 DeepSpeed ZeRO-3 优化
    if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
        # 检查 DeepSpeed 版本要求
        require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        # 设置 MoE 块为叶子模块，优化内存使用
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # 3. 处理 DeepSeek V3 模型的 DeepSpeed ZeRO-3 优化
    if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
        require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
        # 手动设置每个 MoE 层为叶子模块
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    # 4. 配置梯度检查点 (Gradient Checkpointing)
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        # 启用梯度检查点以节省显存
        model.gradient_checkpointing_enable()
        # 禁用缓存以避免梯度检查点冲突
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        # 不启用梯度检查点时保持缓存开启
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # 5. 启用输入梯度计算
    # 确保模型参数可以正确计算梯度
    model.enable_input_require_grads()


def setup_peft_model(model, script_args, training_args, load_in_8bit, load_in_4bit):
    """设置PEFT模型"""
    # 记录日志：显示当前使用的是LoRA(PEFT)微调方法
    logger.info("Fine-tuning method: LoRA(PEFT)")

    # 获取模型的输出层（通常是lm_head）
    output_layer = getattr(model, "lm_head")
    # 检查输出层是否为线性层且权重数据类型不是float32
    if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
        # 定义一个后向钩子函数，将输出转换为float32精度
        def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            return output.to(torch.float32)

        # 为输出层注册前向传播钩子，确保输出为float32精度
        output_layer.register_forward_hook(fp32_forward_post_hook)

    # 检查是否指定了预训练的PEFT模型路径
    if script_args.peft_path is not None:
        # 从预训练的PEFT模型加载权重
        logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
        # 创建PeftModel实例，设置为可训练模式
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
    else:
        # 没有预训练PEFT模型，需要初始化新的PEFT模型
        logger.info("Init new peft model")

        # 如果使用了8位或4位量化，需要为量化训练准备模型
        if load_in_8bit or load_in_4bit:
            # 准备模型用于k-bit训练，包括梯度检查点设置
            model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)

        # 解析目标模块参数：如果指定了target_modules，则按逗号分割
        target_modules = script_args.target_modules.split(',') if script_args.target_modules else None

        # 如果目标模块包含'all'，则自动查找所有线性层
        if target_modules and 'all' in target_modules:
            # 根据量化类型查找所有线性层名称
            target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)

        # 解析需要保存的模块参数：如果指定了modules_to_save，则按逗号分割
        modules_to_save = script_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')

        # 记录PEFT配置信息
        logger.info(f"Peft target_modules: {target_modules}")
        logger.info(f"Peft lora_rank: {script_args.lora_rank}")

        # 创建LoRA配置对象
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
            target_modules=target_modules,  # 目标模块：应用LoRA的模块
            inference_mode=False,  # 非推理模式，启用训练
            r=script_args.lora_rank,  # LoRA rank：低秩矩阵的秩
            lora_alpha=script_args.lora_alpha,  # LoRA alpha：缩放因子
            lora_dropout=script_args.lora_dropout,  # LoRA dropout：防止过拟合
            modules_to_save=modules_to_save)  # 需要完整保存的模块

        # 使用PEFT配置包装模型，返回可训练的PEFT模型
        model = get_peft_model(model, peft_config)

    # 将所有需要梯度的参数转换为float32精度，确保训练稳定性
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # 打印可训练参数信息，显示参数数量统计
    model.print_trainable_parameters()

    # 返回配置好的PEFT模型
    return model


def train_model(trainer, training_args, max_train_samples):
    """
    训练模型
    
    Args:
        trainer: 训练器对象，负责模型训练的执行
        training_args: 训练参数配置对象，包含各种训练相关的配置
        max_train_samples: 最大训练样本数量
    
    Returns:
        dict: 训练结果指标字典，包含损失、训练样本数等训练统计信息
    """
    # 仅在主进程(进程ID为0)中打印训练开始信息
    if trainer.is_world_process_zero():
        logger.info("*** 开始模型训练 ***")

        # 获取训练数据加载器中的一个样本用于调试
        sample = next(iter(trainer.get_train_dataloader()))
        logger.debug(f"训练数据样本示例: {sample}")

        # 打印样本中的input_ids和labels的前3个元素，用于检查数据格式
        logger.debug(f"输入ID序列:\n{list(sample['input_ids'])[:3]}, \n标签序列:\n{list(sample['labels'])[:3]}")

    # 初始化检查点路径，用于断点续训
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # 执行模型训练，支持从检查点恢复训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 获取训练结果指标
    metrics = train_result.metrics

    # 添加训练样本数量到指标中
    metrics["train_samples"] = max_train_samples

    # 记录训练指标到日志
    trainer.log_metrics("train", metrics)

    # 保存训练指标到文件
    trainer.save_metrics("train", metrics)

    # 保存训练器状态，包括优化器状态、调度器状态等
    trainer.save_state()

    # 返回训练指标
    return metrics


def evaluate_model(trainer, max_eval_samples):
    """评估模型

    Args:
        trainer: 训练器对象，负责模型的训练和评估操作
        max_eval_samples: 最大评估样本数量，用于指定评估时使用的数据量

    Returns:
        dict: 包含评估指标的字典，包括loss、perplexity等指标
    """
    # 检查当前进程是否为主进程（rank 0）
    # 只有主进程才会打印日志信息，避免多进程重复输出
    if trainer.is_world_process_zero():
        logger.info("*** Evaluate ***")

    # 执行模型评估，获取评估指标
    # metric_key_prefix="eval" 为指标键名添加前缀，如"eval_loss"
    metrics = trainer.evaluate(metric_key_prefix="eval")

    # 添加评估样本数量到指标字典中
    metrics["eval_samples"] = max_eval_samples

    # 计算困惑度（Perplexity）
    # 困惑度是评估语言模型性能的重要指标，值越小表示模型预测越准确
    # 使用数学公式：perplexity = exp(loss)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        # 当loss过大时，指数运算可能导致数值溢出
        # 此时将困惑度设为无穷大，表示模型性能极差
        perplexity = float("inf")

    # 将困惑度添加到指标字典中
    metrics["perplexity"] = perplexity

    # 记录评估指标到日志系统
    # 这些指标会被保存到训练日志中，便于后续分析
    trainer.log_metrics("eval", metrics)

    # 保存评估指标到文件
    # 通常会保存到output_dir/eval_results.json等文件中
    trainer.save_metrics("eval", metrics)

    # 返回包含所有评估指标的字典
    return metrics


def setup_model_config(model_args, script_args):
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    return config, config_kwargs, dtype


def main():
    # 1. 创建参数解析器，解析四种类型的参数
    # - ModelArguments: 模型相关参数（模型路径、量化设置等）
    # - DataArguments: 数据相关参数（数据集路径、样本数量限制等）
    # - Seq2SeqTrainingArguments: 训练参数（学习率、批次大小、训练步数等）
    # - ScriptArguments: 脚本特定参数（LoRA设置、模板选择等）
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    # 2. 解析命令行参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果提供了JSON配置文件，从文件解析参数
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则从命令行参数解析
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    # 3. 处理DeepSpeed配置
    if training_args.deepspeed is not None:
        # 清空分布式状态中的DeepSpeed插件配置，避免冲突
        training_args.distributed_state.deepspeed_plugin = None

    # 4. 确定是否为主进程
    # local_rank为-1表示单GPU，0表示多GPU中的主进程
    is_main_process = training_args.local_rank in [-1, 0]

    # 5. 仅在主进程上打印配置信息
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        # 打印分布式训练和精度设置信息
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # 6. 设置随机种子，确保实验可复现
    set_seed(training_args.seed)

    # 7. 检查并优化GPU显存使用
    check_and_optimize_memory()

    # 8. 设置tokenizer和提示词模板
    tokenizer, prompt_template = setup_tokenizer(model_args, script_args)

    # 9. 设置损失计算时的忽略索引
    # 如果配置为忽略padding token的损失，使用-100（LabelSmoother.ignore_index）
    # 否则使用tokenizer的pad_token_id
    IGNORE_INDEX = (
        LabelSmoother.ignore_index
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id
    )

    # 10. 加载数据集
    raw_datasets = load_datasets(data_args, model_args)

    # 11. 处理训练数据集
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        # 检查是否有训练数据
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")

        # 打乱训练数据并获取数据集
        train_dataset = raw_datasets['train'].shuffle(seed=42)

        # 对训练数据进行预处理（tokenization、格式转换等）
        train_dataset, max_train_samples = process_train_dataset(
            train_dataset, data_args, training_args, is_main_process, tokenizer, script_args, IGNORE_INDEX,
            prompt_template
        )

    # 12. 处理验证数据集
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        # 检查是否有验证数据
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        # 打乱验证数据并获取数据集
        eval_dataset = raw_datasets['validation'].shuffle(seed=42)

        # 对验证数据进行预处理
        eval_dataset, max_eval_samples = process_eval_dataset(
            eval_dataset, data_args, training_args, tokenizer, script_args, IGNORE_INDEX, prompt_template
        )

    # 13. 加载模型
    if model_args.model_name_or_path:
        # 设置模型配置
        config, config_kwargs, dtype = setup_model_config(model_args, script_args)

        # 设置量化配置（4bit/8bit量化）
        quantization_config, load_in_4bit, load_in_8bit \
            = setup_quantization_config(model_args, script_args, dtype, training_args)

        # 设置模型加载参数
        model_kwargs = setup_model_kwargs(model_args, config, config_kwargs, dtype, quantization_config, training_args)

        # 14. 处理分布式训练设置
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1  # 是否为分布式数据并行
        if ddp:
            # 设置设备映射，每个进程使用对应的GPU
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            # 调整梯度累积步数以适应多进程
            training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size or 1

        # 15. 检查QLoRA与ZeRO-3的兼容性
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        # 打印模型加载配置
        logger.info(f"🔧 大模型训练配置:")
        logger.info(f"  model_kwargs: {model_kwargs}")

        # 16. 从预训练模型加载
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("✅ 模型加载完成")

        # 打印模型信息（参数数量、设备分布等）
        log_model_info(model)

        # 设置NEFTune噪声注入（提高泛化能力）
        setup_neftune(model, model_args)

        # 应用模型特定补丁（如ChatGLM、Mixtral等）
        setup_model_patches(model, config, training_args)

        # 17. 多GPU并行设置
        if not ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        # 18. 应用PEFT（参数高效微调）
        if script_args.use_peft:
            model = setup_peft_model(model, script_args, training_args, load_in_8bit, load_in_4bit)
        else:
            # 全参数微调模式
            logger.info("Fine-tuning method: Full parameters training")
            model = model.float()
            print_trainable_parameters(model)
    else:
        # 必须指定预训练模型路径
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    # 19. 创建数据整理器
    # 负责将batch中的数据整理成模型输入格式
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    )

    # 20. 创建训练器
    # 使用自定义的SavePeftModelTrainer来支持PEFT模型的保存
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 21. 执行训练
    if training_args.do_train:
        # 开始训练并获取训练指标
        metrics = train_model(trainer, training_args, max_train_samples)

        # 训练完成后重新启用缓存以提高推理效率
        model.config.use_cache = True
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"

        # 仅在主进程上保存模型
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")

            # 根据是否使用DeepSpeed ZeRO-3选择不同的保存方式
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # 22. 执行评估
    if training_args.do_eval:
        # 在验证集上评估模型性能
        metrics = evaluate_model(trainer, max_eval_samples)

        # 仅在主进程上打印评估结果
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()

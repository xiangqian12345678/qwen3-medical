from dataclasses import dataclass
from typing import Optional, List, Dict, Sequence

__all__ = ['Conversation', 'register_conv_template', 'get_conv_template']


@dataclass
class Conversation:
    """
    对话模板类，用于管理对话格式和保持对话历史记录。

    该类基于 ChatML 格式（Qwen、Yi、InternLM2 等模型使用），通过模板化的方式
    将消息列表转换为模型输入格式。ChatML 格式使用特殊标记来标识角色边界：
    - <|im_start|>role: 开始角色标记
    - <|im_end|>: 结束角色标记

    Attributes:
        name: 模板名称，用于注册和获取模板
        system_prompt: 系统提示词，定义模型的身份和行为
        messages: 消息列表，格式为 [[问题1, 答案1], [问题2, 答案2], ...]
        roles: 角色名称，默认为 ("user", "assistant")
        prompt: 用户问题模板，使用 {query} 作为占位符
        sep: 分隔符，用于连接对话轮次
        stop_str: 停止字符串，用于标识回复结束，默认为 <|im_end|>
    """

    # The name of this template
    name: str
    # The system prompt
    system_prompt: str
    # All messages. format: list of [question, answer]
    messages: Optional[List[Sequence[str]]]
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Conversation prompt
    prompt: str
    # Separator
    sep: str
    # Stop token, default is tokenizer.eos_token
    stop_str: Optional[str] = "</s>"

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        获取包含对话历史的完整提示字符串（不包含最后一个回复）。

        该方法将格式化后的对话列表连接成一个字符串，通常用于模型输入。
        返回的字符串包含所有历史对话，但不包含最后的模型回复部分。

        Args:
            messages: 可选的消息列表，格式为 [[问题1, 答案1], [问题2, 答案2], ...]
                     如果为None，则使用 self.messages
            system_prompt: 可选的系统提示词，如果为空字符串则使用 self.system_prompt

        Returns:
            str: 包含系统提示和所有对话历史的字符串（最后一个问题的响应前）

        Example (Qwen template):
            Input:
                history_messages = [
                    ["治疗阳痿吃什么药呢？", "男子早泄、早泄病症的再次发生，多由恣情纵欲..."]
                ]
                system_prompt = "你是一个专业的医疗助手。"

            Output:
                "<|im_start|>system\\n你是一个专业的医疗助手。<|im_end|>\\n"
                "<|im_start|>user\\n治疗阳痿吃什么药呢？<|im_end|>\\n<|im_start|>assistant\\n男子早泄、早泄病症的再次发生，多由恣情纵欲...<|im_end|>\\n"
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        获取格式化后的对话列表。

        该方法返回一个列表，其中偶数索引（0, 2, 4, ...）是用户问题，奇数索引（1, 3, 5, ...）是对应的模型回复。
        列表的最后两个元素是最新的一轮对话（问题和回答）。

        Args:
            messages: 可选的消息列表，格式为 [[问题1, 答案1], [问题2, 答案2], ...]
                     如果为None，则使用 self.messages
            system_prompt: 可选的系统提示词，如果为空字符串则使用 self.system_prompt

        Returns:
            List[str]: 格式化后的对话列表，长度为 2*n，其中第 2k 个元素是问题，第 2k+1 个元素是答案

        Example (Qwen template):
            Input:
                messages = [["你好", "你好！我是Qwen"], ["今天天气如何", "今天天气晴朗"]]
                system_prompt = "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n"

            Output:
                [
                    "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n",
                    "你好！我是Qwen",
                    "<|im_end|>\\n<|im_start|>user\\n今天天气如何<|im_end|>\\n<|im_start|>assistant\\n",
                    "今天天气晴朗"
                ]
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        格式化对话示例，将消息列表转换为 ChatML 格式的对话序列。

        ChatML 格式说明：
        - <|im_start|>role: 开始一个角色（system/user/assistant）
        - role 之后的内容是角色的文本
        - <|im_end|>: 结束当前角色
        - 不同角色之间用换行符分隔

        Args:
            messages: 可选的消息列表，格式为 [[问题1, 答案1], [问题2, 答案2], ...]
                     如果为None，则使用 self.messages
            system_prompt: 可选的系统提示词，如果为空字符串则使用 self.system_prompt

        Returns:
            List[str]: 格式化后的对话列表，包含系统提示、用户问题和助手回答的交替序列

        Example (Qwen template with multi-turn conversation):
            Input:
                messages = [["你好", "你好！我是Qwen助手"], ["感冒吃什么药？", "建议感冒多喝水"]]
                system_prompt = "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n"
                self.sep = "\\n"
                self.prompt = "<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n"

            Output:
                [
                    "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n",
                    "你好！我是Qwen助手",
                    "\\n<|im_start|>user\\n感冒吃什么药？<|im_end|>\\n<|im_start|>assistant\\n",
                    "建议感冒多喝水"
                ]

            组合后的完整格式：
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            你好<|im_end|>
            <|im_start|>assistant
            你好！我是Qwen助手<|im_end|>
            <|im_start|>user
            感冒吃什么药？<|im_end|>
            <|im_start|>assistant
            建议感冒多喝水<|im_end|>

        Example (Qwen template with empty messages):
            Input:
                messages = []
                system_prompt = ""

            Output:
                []  # 空列表，因为没有消息需要格式化
        """
        # 使用传入的系统提示词或默认的系统提示词
        system_prompt = system_prompt or self.system_prompt
        # 如果系统提示词不为空，则在末尾添加分隔符
        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
        # 使用传入的消息列表或默认的消息列表
        messages = messages or self.messages
        # 初始化格式化后的对话列表
        convs = []
        # 确保messages不为None，如果是None则设为空列表
        if not messages:
            messages = []
        # 遍历每一轮对话
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            # 如果是第一轮对话
            if turn_idx == 0:
                # 添加系统提示词和第一个用户问题
                convs.append(system_prompt + self.prompt.format(query=user_query))
                # 添加第一个机器人回答
                convs.append(bot_resp)
            # 如果不是第一轮对话
            else:
                # 添加分隔符和当前用户问题
                convs.append(self.sep + self.prompt.format(query=user_query))
                # 添加当前机器人回答
                convs.append(bot_resp)
        # 返回格式化后的对话列表
        return convs

    def append_message(self, query: str, answer: str):
        """
        追加新的对话消息。

        Args:
            query: 用户的问题
            answer: 模型的回答

        Example:
            conv.append_message("什么是感冒？", "感冒是由病毒引起的上呼吸道感染...")
        """
        self.messages.append([query, answer])


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """
    注册一个新的对话模板。

    Args:
        template: Conversation 对象，包含模板的所有配置

    Example:
        register_conv_template(
            Conversation(
                name="qwen",
                system_prompt="<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n",
                messages=[],
                roles=("user", "assistant"),
                prompt="<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n",
                sep="\\n",
                stop_str="<|im_end|>",
            )
        )
    """
    conv_templates[template.name] = template


"""Vicuna v1.1 template
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT:",
        sep="</s>",
    )
)

"""Base model template, for few shot"""
register_conv_template(
    Conversation(
        name="base",
        system_prompt="",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="{query}",
        sep="</s>",
    )
)

"""Alpaca template"""
register_conv_template(
    Conversation(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.",
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{query}\n\n### Response:\n",
        sep="\n\n",
    )
)

"""Baichuan template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py#L31
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan",
        system_prompt="",
        messages=[],
        roles=("<reserved_102>", "<reserved_103>"),
        prompt="<reserved_102>{query}<reserved_103>",
        sep="</s>",
    )
)

"""Baichuan2 template
Support: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
         https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan2",
        system_prompt="",
        messages=[],
        roles=("<reserved_106>", "<reserved_107>"),
        prompt="<reserved_106>{query}<reserved_107>",
        sep="</s>",
    )
)

"""ziya template"""
register_conv_template(
    Conversation(
        name="ziya",
        system_prompt="",
        messages=[],
        roles=("<human>", "<bot>"),
        prompt="<human>:{query}\n<bot>:",
        sep="\n",
    )
)

"""Linly template"""
register_conv_template(
    Conversation(
        name="linly",
        system_prompt="",
        messages=[],
        roles=("User", "Bot"),
        prompt="User: {query}\nBot: ",
        sep="\n",
    )
)

"""ChatGLM1 template
Support: https://huggingface.co/THUDM/chatglm-6b
source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
"""
register_conv_template(
    Conversation(
        name="chatglm",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n答：",
        sep="\n",
    )
)

"""ChatGLM2 template
Support: https://huggingface.co/THUDM/chatglm2-6b
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    Conversation(
        name="chatglm2",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n\n答：",
        sep="\n\n",
    )
)

"""ChatGLM3 template
Support: https://huggingface.co/THUDM/chatglm3-6b
source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
"""
register_conv_template(
    Conversation(
        name="chatglm3",
        system_prompt="",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|assistant|>",
        sep="\n",
        stop_str="<|user|>",
    )
)

"""Phoenix template"""
register_conv_template(
    Conversation(
        name="phoenix",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: <s>{query}</s>Assistant: ",
        sep="</s>",
    )
)

"""belle template
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_conv_template(
    Conversation(
        name="belle",
        system_prompt="",
        messages=[],
        roles=("Human", "Belle"),
        prompt="Human: {query}\n\nBelle: ",
        sep="\n\n",
    )
)

"""aquila template
Supports: https://huggingface.co/qhduan/aquilachat-7b
          https://huggingface.co/BAAI/AquilaChat2-34B
"""
register_conv_template(
    Conversation(
        name="aquila",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}###Assistant:",
        sep="###",
    )
)

"""intern template
Supports: https://huggingface.co/internlm/internlm-chat-7b
          https://huggingface.co/internlm/internlm-chat-20b
"""
register_conv_template(
    Conversation(
        name="intern",
        system_prompt="",
        messages=[],
        roles=("<|User|>", "<|Bot|>"),
        prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
        sep="<eoa>\n",
        stop_str="<eoa>",
    )
)

"""intern2 template
Supports: https://huggingface.co/internlm/internlm2-1_8b
"""
register_conv_template(
    Conversation(
        name="intern2",
        system_prompt="<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

"""StarChat template
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_conv_template(
    Conversation(
        name="starchat",
        system_prompt="<system>\n",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
        sep="<|end|>\n",
        stop_str="<|end|>",
    )
)

"""llama2 template
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
"""
register_conv_template(
    Conversation(
        name="llama2",
        system_prompt=(
            "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, "
            "toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, "
            "explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        ),
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""llama3 template
source: https://huggingface.co/meta-llama
Supports: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
chat template:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ model_answer_1 }}<|eot_id|>
"""
register_conv_template(
    Conversation(
        name="llama3",
        system_prompt=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful, excellent and smart assistant."
        ),
        messages=[],
        roles=("user", "assistant"),
        prompt=(
            "<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        sep="<|eot_id|>",
        stop_str="<|eot_id|>",
    )
)

"""llama2-zh template
source: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_conv_template(
    Conversation(
        name="llama2-zh",
        system_prompt="[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST]",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""mistral template
Supports: https://huggingface.co/mistralai/Mistral-7B-v0.1
          https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
source: https://docs.mistral.ai/llm/mistral-instruct-v0.1
"""
register_conv_template(
    Conversation(
        name="mistral",
        system_prompt="",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""XVERSE template
Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_conv_template(
    Conversation(
        name="xverse",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: ",
        sep="</s>",
    )
)

"""chatml template
chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
"""
register_conv_template(
    Conversation(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

"""deepseek template
Supports: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat
          https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat
"""
register_conv_template(
    Conversation(
        name="deepseek",
        system_prompt="",
        messages=[],
        roles=("User", "Assistant"),
        prompt="User: {query}\n\nAssistant:",
        sep="</s>",
    )
)

"""deepseek3 template
Supports: https://huggingface.co/deepseek-ai/DeepSeek-V3
"""
register_conv_template(
    Conversation(
        name="deepseek3",
        system_prompt="",
        messages=[],
        roles=("<｜User｜>", "<｜Assistant｜>"),
        prompt="<｜User｜>{query}<｜Assistant｜>",
        sep="</s>",
    )
)

"""deepseekcoder template
Supports: https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct
"""
register_conv_template(
    Conversation(
        name="deepseekcoder",
        system_prompt=(
            "You are an AI programming assistant, utilizing the Deepseek Coder model, "
            "developed by Deepseek Company, and you only answer questions related to computer science. "
            "For politically sensitive questions, security and privacy issues, "
            "and other non-computer science questions, you will refuse to answer\n"
        ),
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{{content}}\n### Response:\n",
        sep="\n",
        stop_str="<|EOT|>",
    )
)

"""Yi template
source: https://github.com/01-ai/Yi
Supports: https://huggingface.co/01-ai/Yi-34B-Chat
          https://huggingface.co/01-ai/Yi-6B-Chat
"""
register_conv_template(
    Conversation(
        name="yi",
        system_prompt="",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

"""Orion template
source: https://github.com/OrionStarAI/Orion
Supports: https://huggingface.co/OrionStarAI/Orion-14B-Chat
"""
register_conv_template(
    Conversation(
        name="orion",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: </s>",
        sep="</s>",
    )
)

"""Cohere template
source: https://huggingface.co/CohereForAI/c4ai-command-r-plus
Supports: https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit
          https://huggingface.co/CohereForAI/c4ai-command-r-plus
"""
register_conv_template(
    Conversation(
        name="cohere",
        system_prompt="<BOS_TOKEN>",
        messages=[],
        roles=("User", "Assistant"),
        prompt=(
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{query}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
        sep="</s>",
    )
)

"""Qwen template
source: https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat/blob/main/tokenizer_config.json#L18
Supports: https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat
          https://huggingface.co/Qwen/Qwen1.5-72B-Chat
          https://huggingface.co/Qwen/Qwen2-72B
          https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""
register_conv_template(
    Conversation(
        name="qwen",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="\n",
        stop_str="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="deepseek",
        system_prompt="<BOS_TOKEN>",
        messages=[],
        roles=("User", "Assistant"),
        prompt=(
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{query}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
        sep="</s>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """
    根据名称获取已注册的对话模板。

    Args:
        name: 模板名称，如 "qwen", "llama3", "chatglm3" 等

    Returns:
        Conversation: 对应的对话模板对象

    Example:
        qwen_template = get_conv_template("qwen")
        prompt = qwen_template.get_prompt([["你好", ""]])
    """
    return conv_templates[name]

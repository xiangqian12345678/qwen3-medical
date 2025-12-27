import argparse
import json
import os

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1.参数配置
parser = argparse.ArgumentParser(description="========量化困惑度测试========")
parser.add_argument(
    "--bnb_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="bnb量化后的模型路径。"
)
parser.add_argument(
    "--data_path",
    type=str,
    required=True,  # 设置为必须的参数
    help="jsonl数据集路径。"
)


# 2.功能函数定义
# 设备选择函数
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


# 3.加载评估数据
def load_jsonl_data(file_path):
    logger.info(f"Loading data from {file_path}")
    conversations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 human 和 gpt 部分的文本
                for conv in data['conversations']:
                    if conv['from'] == 'human':
                        input_text = conv['value']
                    elif conv['from'] == 'gpt':
                        target_text = conv['value']
                        conversations.append((input_text, target_text))
        return conversations
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


# 4.困惑度评估函数
def evaluate_perplexity(model, tokenizer, conversation_pairs, max_length=512):
    """
    评估模型在给定对话对上的困惑度 (Perplexity, PPL)。

    困惑度公式:
        对单个序列：
            PPL = exp( (1/T) * Σ_i L_i )
            其中 T 是序列长度，L_i 是第 i 个 token 的负对数似然 (Negative Log-Likelihood, NLL)

        对整个数据集：
            PPL_dataset = exp( Σ_j NLL_j / Σ_j T_j )
            其中 j 遍历每个样本，NLL_j 是第 j 个样本的总负对数似然，T_j 是样本长度

    参数:
        model: 预训练语言模型
        tokenizer: 对应的分词器
        conversation_pairs: 对话对列表，每个元素为 (input_text, target_text) 元组
        max_length: 输入/输出序列最大长度（默认 512）

    返回:
        float: 困惑度值，越低表示模型预测能力越好
    """

    def _compute_perplexity(nll_list, total_tokens):
        """
        根据总负对数似然和总 token 数计算困惑度
        """
        try:
            return torch.exp(torch.stack(nll_list).sum() / total_tokens)
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    model.eval()  # 设置评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    nll_list = []
    total_tokens = 0  # 统计总 token 数

    for input_text, target_text in tqdm(conversation_pairs, desc="困惑度评估"):
        # 编码输入和目标文本
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).input_ids.to(device)

        target_ids = tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).input_ids.to(device)

        # 前向传播计算交叉熵损失
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss  # 平均负对数似然
            seq_len = target_ids.size(1)  # 当前样本的 token 数
            nll_list.append(loss * seq_len)  # 样本总负对数似然
            total_tokens += seq_len

    # 计算最终困惑度
    ppl = _compute_perplexity(nll_list, total_tokens)
    print('-' * 100)
    print(f"模型:{model.name_or_path}  困惑度: {ppl:.3f}")
    print('-' * 100)

    return ppl.item()


if __name__ == "__main__":
    # 1.解析参数
    args = parser.parse_args()

    # 2.加载量化模型
    if not os.path.exists(args.bnb_path):
        logger.error(f"Model path {args.bnb_path} does not exist.")
        exit(1)

    logger.info(f"Loading BNB model from: {args.bnb_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.bnb_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.bnb_path, trust_remote_code=True)

    # 将模型移动到最佳设备
    device = get_device()
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")

    # 3.加载评估数据
    # 加载jsonl数据
    conversation_pairs = load_jsonl_data(args.data_path)

    if not conversation_pairs:
        logger.error("No valid conversation pairs found.")
        exit(1)

    # 4.开始评估
    evaluate_perplexity(model, tokenizer, conversation_pairs)

    # 5.清理模型和缓存
    del model
    del tokenizer
    torch.cuda.empty_cache()

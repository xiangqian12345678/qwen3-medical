"""
group_text_function 函数使用样例
演示如何将文本数据分组为固定大小的块，用于语言模型预训练
"""

from itertools import chain
from typing import Dict, List, Any


def group_text_function(examples, block_size):
    """
    文本分组函数：将连接的文本分割成固定大小的块
    
    Args:
        examples: 分词后的示例
        block_size: 块大小
    
    Returns:
        分组后的结果
    """
    # 检查输入是否为空
    if not examples or not any(examples.values()):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # 连接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # 检查连接后的文本是否为空
    if not concatenated_examples or not concatenated_examples[list(examples.keys())[0]]:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 我们丢弃小的余数，如果模型支持，可以添加填充而不是丢弃，您可以
    # 根据需要自定义这部分。
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按最大长度的块分割
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def example_1_basic_usage():
    """样例1：基本用法"""
    print("=== 样例1：基本用法 ===")
    
    # 模拟分词后的数据
    examples = {
        "input_ids": [
            [1, 2, 3, 4, 5],      # 第一个样本
            [6, 7, 8, 9, 10, 11], # 第二个样本
            [12, 13, 14]          # 第三个样本
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1]
        ]
    }
    
    block_size = 8
    result = group_text_function(examples, block_size)
    
    print(f"原始数据长度: {len(examples['input_ids'])} 个样本")
    print(f"块大小: {block_size}")
    print(f"连接后的总长度: {len(list(chain(*examples['input_ids'])))}")
    print(f"分组后数量: {len(result['input_ids'])} 个块")
    print(f"第一个块: {result['input_ids'][0]}")
    if len(result['input_ids']) > 1:
        print(f"第二个块: {result['input_ids'][1]}")
    else:
        print("注意：只有一个块，因为数据不足以生成第二个块")
    print()


def example_2_empty_input():
    """样例2：空输入处理"""
    print("=== 样例2：空输入处理 ===")
    
    # 空输入
    empty_examples = {
        "input_ids": [],
        "attention_mask": []
    }
    
    result = group_text_function(empty_examples, 512)
    print(f"空输入结果: {result}")
    
    # None 输入
    none_result = group_text_function(None, 512)
    print(f"None输入结果: {none_result}")
    print()


def example_3_medical_text():
    """样例3：医疗文本处理"""
    print("=== 样例3：医疗文本处理 ===")
    
    # 模拟医疗文本的token IDs
    medical_examples = {
        "input_ids": [
            # 第一个医疗样本：症状描述
            [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            # 第二个医疗样本：诊断结果
            [201, 202, 203, 204, 205, 206, 207, 208, 209],
            # 第三个医疗样本：治疗方案
            [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312]
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
    }
    
    block_size = 16
    result = group_text_function(medical_examples, block_size)
    
    print(f"医疗文本样本数: {len(medical_examples['input_ids'])}")
    print(f"块大小: {block_size}")
    print(f"总tokens数: {len(list(chain(*medical_examples['input_ids'])))}")
    print(f"生成块数: {len(result['input_ids'])}")
    
    for i, block in enumerate(result['input_ids']):
        print(f"块 {i+1}: 长度={len(block)}, 内容={block}")
    print()


def example_4_different_block_sizes():
    """样例4：不同块大小的影响"""
    print("=== 样例4：不同块大小的影响 ===")
    
    examples = {
        "input_ids": [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
    }
    
    total_tokens = len(list(chain(*examples['input_ids'])))
    print(f"总tokens数: {total_tokens}")
    
    # 测试不同的块大小
    for block_size in [4, 8, 16]:
        result = group_text_function(examples, block_size)
        used_tokens = len(result['input_ids']) * block_size
        wasted_tokens = total_tokens - used_tokens
        
        print(f"块大小 {block_size}:")
        print(f"  生成块数: {len(result['input_ids'])}")
        print(f"  使用tokens: {used_tokens}")
        print(f"  丢弃tokens: {wasted_tokens}")
        print(f"  利用率: {used_tokens/total_tokens:.2%}")
    print()


def example_5_with_labels():
    """样例5：包含labels字段的数据处理"""
    print("=== 样例5：包含labels字段 ===")
    
    examples = {
        "input_ids": [
            [1001, 1002, 1003, 1004, 1005],
            [1006, 1007, 1008, 1009, 1010, 1011]
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ],
        "labels": [
            [1001, 1002, 1003, 1004, 1005],
            [1006, 1007, 1008, 1009, 1010, 1011]
        ]
    }
    
    block_size = 6
    result = group_text_function(examples, block_size)
    
    print(f"输入块数: {len(result['input_ids'])}")
    print(f"标签块数: {len(result['labels'])}")
    print(f"输入和标签是否相同: {result['input_ids'] == result['labels']}")
    
    if len(result['input_ids']) > 0:
        print(f"第一块输入: {result['input_ids'][0]}")
        print(f"第一块标签: {result['labels'][0]}")
    else:
        print("注意：没有生成任何块，因为数据不足")
    print()


def example_6_realistic_scenario():
    """样例6：真实场景模拟"""
    print("=== 样例6：真实场景模拟 ===")
    
    # 模拟大量医疗文本数据
    import random
    
    # 生成100个样本，每个样本长度在50-200之间
    num_samples = 100
    examples = {
        "input_ids": [],
        "attention_mask": []
    }
    
    for i in range(num_samples):
        # 随机生成长度
        length = random.randint(50, 200)
        # 生成随机token IDs (模拟医疗词汇)
        tokens = [random.randint(1000, 9999) for _ in range(length)]
        mask = [1] * length
        
        examples["input_ids"].append(tokens)
        examples["attention_mask"].append(mask)
    
    block_size = 512  # 典型的块大小
    result = group_text_function(examples, block_size)
    
    total_tokens = sum(len(tokens) for tokens in examples["input_ids"])
    used_tokens = len(result["input_ids"]) * block_size
    
    print(f"原始样本数: {num_samples}")
    print(f"总tokens数: {total_tokens:,}")
    print(f"块大小: {block_size}")
    print(f"生成训练块数: {len(result['input_ids'])}")
    print(f"tokens利用率: {used_tokens/total_tokens:.2%}")
    print(f"平均每个样本生成的块数: {len(result['input_ids'])/num_samples:.2f}")
    print()


if __name__ == "__main__":
    # 运行所有样例
    example_1_basic_usage()
    example_2_empty_input()
    example_3_medical_text()
    example_4_different_block_sizes()
    example_5_with_labels()
    example_6_realistic_scenario()
    
    print("所有样例运行完成！")
import argparse
import json
import os
import shutil

import sentencepiece as spm
from transformers import AutoTokenizer


def extract_domain_tokens(domain_sp_model):
    """从SentencePiece模型中提取有效token"""
    domain_vocab_size = domain_sp_model.get_piece_size()
    domain_tokens = []

    for i in range(domain_vocab_size):
        token = domain_sp_model.id_to_piece(i)
        # 移除SentencePiece的特殊标记
        if token not in ['', '<unk>', '<s>', '</s>']:
            domain_tokens.append(token)

    return domain_tokens


def merge_vocabularies(base_tokenizer, domain_tokens):
    """合并基础分词器和领域分词器的词汇表"""
    base_vocab = base_tokenizer.get_vocab()

    # 找出新增的领域词汇
    new_tokens = []
    for token in domain_tokens:
        if token not in base_vocab:
            new_tokens.append(token)

    print(f"基础分词器词汇表大小: {len(base_vocab)}")
    print(f"领域分词器有效词汇大小: {len(domain_tokens)}")
    print(f"新增领域词汇数量: {len(new_tokens)}")

    # 创建合并后的词汇表
    merged_vocab = base_vocab.copy()
    next_id = max(base_vocab.values()) + 1

    for token in new_tokens:
        merged_vocab[token] = next_id
        next_id += 1

    print(f"合并后分词器词汇表大小: {len(merged_vocab)}")
    return merged_vocab, new_tokens


def copy_tokenizer_files(base_tokenizer_dir, output_dir):
    """复制基础分词器文件到输出目录"""
    os.makedirs(output_dir, exist_ok=True)

    files_to_copy = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
        "configuration.json"
    ]

    for file in files_to_copy:
        src_file = os.path.join(base_tokenizer_dir, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(output_dir, file))


def update_vocab_file(output_dir, merged_vocab):
    """更新vocab.json文件"""
    vocab_file = os.path.join(output_dir, "vocab.json")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)


def update_tokenizer_config(output_dir, merged_vocab):
    """更新tokenizer_config.json文件"""
    tokenizer_config_file = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_file):
        with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)

        tokenizer_config["vocab_size"] = len(merged_vocab)

        with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)


def update_added_tokens(output_dir, base_tokenizer_dir, new_tokens, merged_vocab):
    """更新added_tokens.json文件"""
    added_tokens_file = os.path.join(output_dir, "added_tokens.json")
    added_tokens = {}

    # 读取原有的added_tokens
    original_added_tokens_file = os.path.join(base_tokenizer_dir, "added_tokens.json")
    if os.path.exists(original_added_tokens_file):
        with open(original_added_tokens_file, 'r', encoding='utf-8') as f:
            added_tokens = json.load(f)

    # 添加新的tokens到added_tokens
    for token in new_tokens:
        added_tokens[token] = merged_vocab[token]

    with open(added_tokens_file, 'w', encoding='utf-8') as f:
        json.dump(added_tokens, f, ensure_ascii=False, indent=2)


def update_tokenizer_json(output_dir, merged_vocab):
    """更新tokenizer.json文件（Fast Tokenizer的关键文件）"""
    tokenizer_json_file = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_file):
        with open(tokenizer_json_file, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # 对于Fast Tokenizer，需要更新model.vocab中的词汇表
        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
            # 将新增的tokens添加到vocab中
            for token, token_id in merged_vocab.items():
                if token not in tokenizer_data['model']['vocab']:
                    tokenizer_data['model']['vocab'][token] = token_id

            # 更新词汇表大小
            tokenizer_data['model']['vocab_size'] = len(merged_vocab)

            print(f"更新tokenizer.json中的词汇表大小为: {len(merged_vocab)}")

        with open(tokenizer_json_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)


def update_config_json(output_dir, merged_vocab):
    """更新config.json文件中的vocab_size"""
    config_json_file = os.path.join(output_dir, "config.json")
    if os.path.exists(config_json_file):
        with open(config_json_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 更新vocab_size
        if "vocab_size" in config_data:
            config_data["vocab_size"] = len(merged_vocab)
            print(f"更新config.json中的vocab_size为: {len(merged_vocab)}")

        with open(config_json_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)


def create_merged_tokenizer(base_tokenizer, domain_sp_model, output_dir, base_tokenizer_dir):
    """创建合并后的Qwen3格式分词器"""

    # 1. 提取领域词汇
    domain_tokens = extract_domain_tokens(domain_sp_model)

    # 2. 合并词汇表
    merged_vocab, new_tokens = merge_vocabularies(base_tokenizer, domain_tokens)

    # 3. 复制基础分词器文件
    copy_tokenizer_files(base_tokenizer_dir, output_dir)

    # 4. 更新各个配置文件
    # "vocab.json",
    # "tokenizer_config.json",
    # "added_tokens.json",
    # "tokenizer.json",
    # "config.json",
    # "configuration.json", 直接拷贝
    # "merges.txt",         直接拷贝： 存储的是基础分词器的BPE合并规则，新增的领域词汇是通过 added_tokens.json 添加
    # "special_tokens_map.json", 自动生成： 分词模型存储的时候自动生成该文件
    update_vocab_file(output_dir, merged_vocab)
    update_tokenizer_config(output_dir, merged_vocab)
    update_added_tokens(output_dir, base_tokenizer_dir, new_tokens, merged_vocab)
    update_tokenizer_json(output_dir, merged_vocab)
    update_config_json(output_dir, merged_vocab)

    # 5. 创建新的tokenizer实例
    new_tokenizer = AutoTokenizer.from_pretrained(output_dir)

    return new_tokenizer


def main():
    parser = argparse.ArgumentParser(description="合并Qwen3基础分词器和领域分词器")
    parser.add_argument('--base_tokenizer_dir', default=None, type=str, required=True,
                        help='Qwen3基础分词器目录路径')
    parser.add_argument('--domain_sp_model_file', default='./domain_sp.model', type=str,
                        help='领域SentencePiece模型文件路径')
    parser.add_argument('--output_dir', default='./merged_tokenizer_qwen3', type=str,
                        help='输出合并后的分词器目录路径')

    args = parser.parse_args()
    print(f"参数: {args}")

    # 加载基础分词器
    print("加载基础Qwen3分词器...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_dir)
    print(f"基础分词器词汇表大小: {len(base_tokenizer)}")
    print(f"基础分词器类型: {type(base_tokenizer)}")

    # 加载领域SentencePiece模型
    print("加载领域SentencePiece模型...")
    domain_sp_model = spm.SentencePieceProcessor()
    domain_sp_model.Load(args.domain_sp_model_file)
    print(f"领域分词器词汇表大小: {domain_sp_model.get_piece_size()}")

    # 创建合并后的Qwen3格式分词器
    print("创建合并后的Qwen3兼容分词器...")
    merged_tokenizer = create_merged_tokenizer(base_tokenizer, domain_sp_model, args.output_dir,
                                               args.base_tokenizer_dir)

    # 保存合并后的分词器
    print("保存合并后的分词器...")
    merged_tokenizer.save_pretrained(args.output_dir)
    print(f"合并后的分词器已保存到: {args.output_dir}")

    # 测试分词器
    print("\n测试分词器效果:")
    text = '''this is a test, hello world. thisisatesthelloworld, 
慕容复来到河边，姑苏慕容氏在外面丢了人。
1号店一周岁了，我们一古脑儿买了10斤零食。
巴塞罗那足球俱乐部简称巴萨（Barça），是一家位于西班牙加泰罗尼亚巴塞罗那的足球俱乐部，于1899年由瑞士企业家胡安·甘伯所创立，世界球坛顶级足球俱乐部之一。俱乐部主场可容纳接近十万名观众，是全欧洲最大及世界第二大的足球场。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'''

    print(f"测试文本:\n{text}\n")

    # 基础分词器结果
    base_tokens = base_tokenizer.tokenize(text)
    base_ids = base_tokenizer.convert_tokens_to_ids(base_tokens)
    print(f"基础分词器 - tokens数量: {len(base_tokens)}")
    print(f"基础分词器 - tokens: {base_tokens[:50]}...")  # 只显示前50个token

    # 合并后分词器结果
    merged_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    merged_tokens = merged_tokenizer.tokenize(text)
    merged_ids = merged_tokenizer.convert_tokens_to_ids(merged_tokens)
    print(f"\n合并后分词器 - tokens数量: {len(merged_tokens)}")
    print(f"合并后分词器 - tokens: {merged_tokens[:50]}...")  # 只显示前50个token

    print(f"\n词汇表对比:")
    print(f"基础分词器: {len(base_tokenizer)} tokens")
    print(f"合并后分词器: {len(merged_tokenizer)} tokens")
    print(f"新增tokens: {len(merged_tokenizer) - len(base_tokenizer)}")


if __name__ == '__main__':
    main()

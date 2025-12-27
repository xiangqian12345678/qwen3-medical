import argparse
import os
import glob

import sentencepiece as spm


def load_corpus_data(corpus_path):
    """加载语料数据，支持单个文件或文件夹中的所有文件"""
    corpus_texts = []

    if os.path.isfile(corpus_path):
        # 如果是单个文件
        print(f"读取单个文件: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            # 以 UTF-8 编码打开语料文件，读取全部内容并追加到 corpus_texts 列表中
            # 一个文件的所有内容为一个字符串
            corpus_texts.append(f.read())
    elif os.path.isdir(corpus_path):
        # 如果是文件夹，读取所有txt文件
        print(f"读取文件夹: {corpus_path}")
        # 支持多种文件扩展名
        file_patterns = ['*.txt', '*.text', '*.corpus', '*.data']
        files_found = []

        for pattern in file_patterns:
            # 在 corpus_path 目录的当前层中查找所有匹配 pattern 的文件，并加入 files_found 列表
            files_found.extend(glob.glob(os.path.join(corpus_path, pattern)))
            # 递归遍历 corpus_path 及其所有子目录，查找匹配 pattern 的文件，并加入 files_found 列表
            files_found.extend(glob.glob(os.path.join(corpus_path, '**', pattern), recursive=True))

        # 去重
        files_found = list(set(files_found))

        if not files_found:
            print(f"警告: 在文件夹 {corpus_path} 中未找到任何语料文件")
            return []

        print(f"找到 {len(files_found)} 个文件:")
        for file_path in files_found:
            print(f"  - {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # 只添加非空文件
                        corpus_texts.append(content)
                    else:
                        print(f"    警告: 文件为空")
            except Exception as e:
                print(f"    错误: 无法读取文件 {file_path}: {e}")
    else:
        raise ValueError(f"输入路径不存在或不是有效的文件/文件夹: {corpus_path}")

    total_chars = sum(len(text) for text in corpus_texts)
    print(f"总共读取了 {len(corpus_texts)} 个文件，{total_chars} 个字符")
    return corpus_texts


def write_corpus_to_temp_file(corpus_texts, temp_file="temp_corpus.txt"):
    """将所有语料写入临时文件供SentencePiece使用"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in corpus_texts:
            f.write(text + '\n')
    return temp_file


def parse_training_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练领域分词器")

    # 输入相关参数
    parser.add_argument('--input', default='data/', type=str,
                        help='输入语料文件或文件夹路径 (默认: data/)')
    parser.add_argument('--output_dir', default='./', type=str,
                        help='输出模型文件目录 (默认: ./)')

    # 模型相关参数
    parser.add_argument('--model_name', default='domain_sp', type=str,
                        help='分词器模型名称前缀 (默认: domain_sp)')
    parser.add_argument('--vocab_size', default=4000, type=int,
                        help='词汇表大小 (默认: 4000)')
    parser.add_argument('--model_type', default="BPE", type=str, choices=['BPE', 'unigram', 'char', 'word'],
                        help='模型类型 (默认: BPE)')

    # 训练相关参数
    parser.add_argument('--max_sentence_length', default=16384, type=int,
                        help='最大句子长度 (默认: 16384)')
    parser.add_argument('--pad_id', default=3, type=int,
                        help='PAD token ID (默认: 3)')
    parser.add_argument('--shuffle_input', action='store_true', default=False,
                        help='是否打乱输入句子 (默认: False)')

    # 处理相关参数
    parser.add_argument('--split_digits', action='store_true', default=True,
                        help='是否分割数字 (默认: True)')
    parser.add_argument('--split_by_unicode_script', action='store_true', default=True,
                        help='是否按Unicode脚本分割 (默认: True)')
    parser.add_argument('--byte_fallback', action='store_true', default=True,
                        help='是否启用字节回退 (默认: True)')
    parser.add_argument('--allow_whitespace_only_pieces', action='store_true', default=True,
                        help='是否允许仅包含空白的片段 (默认: True)')
    parser.add_argument('--remove_extra_whitespaces', action='store_false', default=True,
                        help='是否移除多余空白 (默认: False)')
    parser.add_argument('--normalization_rule', default="nfkc", type=str,
                        help='标准化规则 (默认: nfkc)')
    parser.add_argument('--character_coverage', default=0.9995, type=float,
                        help='字符覆盖率 (默认: 0.9995)')

    # 测试相关参数
    parser.add_argument('--test_text', default='潜伏性感染又称潜在性感染。慕容复来到河边,this is a test', type=str,
                        help='测试文本 (默认: 潜伏性感染又称潜在性感染。慕容复来到河边,this is a test)')

    return parser.parse_args()


def print_training_config(args):
    """打印训练配置参数"""
    print("=" * 50)
    print("训练参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 50)


def prepare_training_data(args):
    """准备训练数据"""
    # 读取语料
    corpus_texts = load_corpus_data(args.input)
    if not corpus_texts:
        print("错误: 没有找到有效的语料文件")
        return None, None

    # 写入临时文件
    temp_file = os.path.join(args.output_dir, "temp_corpus.txt")
    write_corpus_to_temp_file(corpus_texts, temp_file)

    # 设置模型输出路径
    model_prefix = os.path.join(args.output_dir, args.model_name)

    return temp_file, model_prefix


def train_sentencepiece_model(temp_file, model_prefix, args):
    """训练SentencePiece模型"""
    print("开始训练分词器...")
    try:
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            shuffle_input_sentence=args.shuffle_input,
            train_extremely_large_corpus=True,
            max_sentence_length=args.max_sentence_length,
            pad_id=args.pad_id,
            model_type=args.model_type,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            split_digits=args.split_digits,
            split_by_unicode_script=args.split_by_unicode_script,
            byte_fallback=args.byte_fallback,
            allow_whitespace_only_pieces=args.allow_whitespace_only_pieces,
            remove_extra_whitespaces=args.remove_extra_whitespaces,
            normalization_rule_name=args.normalization_rule,
        )
        print("训练完成!")
        return True
    except Exception as e:
        print(f"训练失败: {e}")
        return False


def test_trained_model(model_prefix, test_text):
    """测试训练好的模型"""
    print("\n测试分词器:")
    try:
        sp = spm.SentencePieceProcessor()
        model_file = model_prefix + '.model'
        sp.load(model_file)

        # 测试编码
        token_pieces = sp.encode_as_pieces(test_text)
        token_ids = sp.encode_as_ids(test_text)

        print(f"测试文本: {test_text}")
        print(f"分词结果: {token_pieces}")
        print(f"ID序列: {token_ids}")
        print(f"解码结果: {sp.decode_pieces(token_pieces)}")

        # 显示词汇表信息
        vocab_size = sp.get_piece_size()
        print(f"\n词汇表大小: {vocab_size}")
        print("前10个词汇:")
        for i in range(min(10, vocab_size)):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)
            print(f"  {i}: '{piece}' (score: {score:.6f})")

        return True
    except Exception as e:
        print(f"加载或测试模型失败: {e}")
        return False


def cleanup_temp_files(temp_file):
    """清理临时文件"""
    if os.path.exists(temp_file):
        os.remove(temp_file)


def main():
    """主函数：训练领域分词器的完整流程"""
    # 解析参数
    training_args = parse_training_arguments()

    # 打印配置
    print_training_config(training_args)

    # 准备数据
    temp_corpus_file, model_output_prefix = prepare_training_data(training_args)
    if temp_corpus_file is None:
        return

    # 训练模型
    training_success = train_sentencepiece_model(temp_corpus_file, model_output_prefix, training_args)

    # 清理临时文件
    cleanup_temp_files(temp_corpus_file)

    # 测试模型
    if training_success:
        test_trained_model(model_output_prefix, training_args.test_text)


if __name__ == '__main__':
    main()

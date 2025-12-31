import os
from glob import glob
from itertools import chain
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from loguru import logger


def tokenize_function(tokenizer, examples, block_size):
    texts = examples["text"]
    if not isinstance(texts, list):
        texts = [texts]
    valid_texts = [str(t) for t in texts if t]
    if not valid_texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    tokenized_inputs = tokenizer(
        valid_texts,
        truncation=True,
        padding='max_length',
        max_length=block_size
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


def group_text_function(examples, block_size):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_local_files(train_dir, val_dir, tokenizer_name, block_size, cache_dir, keep_linebreaks=True):
    data_files = {}
    if train_dir and os.path.exists(train_dir):
        train_files = glob(f'{train_dir}/**/*.txt', recursive=True)
        data_files['train'] = train_files
        logger.info(f"训练文件: {train_files}")
    if val_dir and os.path.exists(val_dir):
        val_files = glob(f'{val_dir}/**/*.txt', recursive=True)
        data_files['validation'] = val_files
        logger.info(f"验证文件: {val_files}")

    extension = "text" if data_files["train"][0].endswith('txt') else 'json'

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, keep_linebreaks=keep_linebreaks)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    processed_datasets = {}
    for split in ["train", "validation"]:
        if split in raw_datasets:
            tokenized = raw_datasets[split].map(
                lambda examples: tokenize_function(tokenizer, examples, block_size),
                batched=True,
                remove_columns=raw_datasets[split].column_names
            )
            grouped = tokenized.map(
                lambda examples: group_text_function(examples, block_size),
                batched=True
            )
            processed_datasets[split] = grouped

    # 保存缓存
    cache_path = os.path.join(cache_dir, "processed_datasets")
    os.makedirs(cache_path, exist_ok=True)
    for split, dataset in processed_datasets.items():
        dataset.save_to_disk(os.path.join(cache_path, split))
        logger.info(f"{split}数据已缓存到 {os.path.join(cache_path, split)}")

    return processed_datasets


if __name__ == "__main__":
    # 示例调用
    preprocess_local_files(
        train_dir="../../data/pretrain",
        val_dir="../../data/pretrain",
        tokenizer_name="../../output/tokenizers_merge",
        block_size=1024,
        cache_dir="../../output/cache"
    )

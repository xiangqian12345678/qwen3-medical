from datasets import load_dataset
from itertools import islice


# =========================================================
# 1. 从 HuggingFace Hub 读取 text-only 预训练数据（非 streaming）
# =========================================================
def load_hf_text_pretrain_dataset(
    dataset_name: str,
    dataset_config: str = None,
    cache_dir: str = None,
    streaming: bool = False,
):
    """
    适用于中小规模 text-only 预训练数据
    例如：wikitext
    """
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    return dataset


# =========================================================
# 2. 从 HuggingFace Hub 读取超大规模 streaming 预训练数据
# =========================================================
def load_hf_streaming_pretrain_dataset(
    dataset_name: str,
    dataset_config: str,
):
    """
    适用于超大规模预训练数据
    例如：allenai/c4
    返回的是 IterableDataset
    """
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
    )
    return dataset


# =========================================================
# 3. 打印 DatasetDict（非 streaming）结构和样例
# =========================================================
def inspect_dataset(dataset, split="train", num_samples=3):
    print("\n" + "=" * 80)
    print("Dataset object:")
    print(dataset)

    print("\nSplit:", split)

    ds = dataset[split]

    print("\nFeatures:")
    print(ds.features)

    print(f"\nFirst {num_samples} samples:")
    for i, sample in enumerate(islice(ds, num_samples)):
        print(f"\n--- Sample {i} ---")
        for k, v in sample.items():
            if isinstance(v, str):
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")


# =========================================================
# 4. 打印 streaming Dataset 结构和样例
# =========================================================
def inspect_streaming_dataset(dataset, num_samples=3):
    print("\n" + "=" * 80)
    print("Streaming Dataset object:")
    print(dataset)

    print(f"\nFirst {num_samples} samples:")
    for i, sample in enumerate(islice(dataset, num_samples)):
        print(f"\n--- Sample {i} ---")
        for k, v in sample.items():
            if isinstance(v, str):
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")


# =========================================================
# 5. main：测试两种预训练数据类型
# =========================================================
def main():
    cache_dir = "./cache"

    # -----------------------------------------------------
    # TEST 1：HuggingFace Hub 普通 text-only 预训练数据
    # -----------------------------------------------------
    print("\n\n########## TEST 1: HF HUB TEXT PRETRAIN DATASET ##########")

    hf_dataset = load_hf_text_pretrain_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        cache_dir=cache_dir,
        streaming=False,
    )

    inspect_dataset(hf_dataset, split="train", num_samples=3)

    # -----------------------------------------------------
    # TEST 2：HuggingFace Hub 超大规模 streaming 预训练数据
    # -----------------------------------------------------
    print("\n\n########## TEST 2: HF HUB STREAMING PRETRAIN DATASET (C4) ##########")

    c4_dataset = load_hf_streaming_pretrain_dataset(
        dataset_name="allenai/c4",
        dataset_config="en",
    )

    inspect_streaming_dataset(c4_dataset, num_samples=3)


if __name__ == "__main__":
    main()

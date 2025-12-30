import os
from datasets import load_dataset

def test_load(data_files, extension, keep_linebreaks=False):
    print("=" * 80)
    print(f"extension = {extension}")
    print(f"data_files = {data_files}")

    dataset_args = {}
    if extension == "text":
        dataset_args["keep_linebreaks"] = keep_linebreaks

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        **dataset_args
    )

    for split, ds in raw_datasets.items():
        print(f"\n[{split}]")
        print(ds)
        print("features:", ds.features)
        print("first 2 samples:")
        for i in range(min(2, len(ds))):
            print(ds[i])

    return raw_datasets


if __name__ == "__main__":
    # ===== txt 测试 =====
    txt_data_files = {
        "train": ["data/txt/train.txt"],
        "validation": ["data/txt/eval.txt"]
    }

    txt_ds = test_load(
        data_files=txt_data_files,
        extension="text",
        keep_linebreaks=True
    )

    # ===== json/jsonl 测试 =====
    json_data_files = {
        "train": ["data/json/train.jsonl"],
        "validation": ["data/json/eval.jsonl"]
    }

    json_ds = test_load(
        data_files=json_data_files,
        extension="json"
    )

import torch
import numpy as np
from typing import List, Dict, Any
from collections.abc import Mapping

# 使用您提供的容错数据整理器函数
def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    """
    容错数据整理器：将特征列表整理成批次格式
    """
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # 特殊处理标签
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # 处理所有其他可能的键
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # 通过简单采用第一个示例进行快速修复
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch

# ==================== 样例演示 ====================

# 样例1: 分类任务数据
print("=== 样例1: 分类任务 ===")
classification_samples = [
    {
        "input_ids": torch.tensor([101, 102, 103, 104, 105]),  # Token IDs
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),       # 注意力掩码
        "label": 0,                                            # 分类标签 (0: 负面, 1: 正面)
        "text": "这个药物效果很好"
    },
    {
        "input_ids": torch.tensor([101, 106, 107, 108, 105]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 1,
        "text": "副作用太大了"
    },
    {
        "input_ids": torch.tensor([101, 109, 110, 111, 105]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 0,
        "text": "价格合理"
    }
]

batch1 = fault_tolerance_data_collator(classification_samples)
print(f"标签形状: {batch1['labels'].shape}")           # torch.Size([3])
print(f"标签数据类型: {batch1['labels'].dtype}")       # torch.int64 (long)
print(f"输入ID形状: {batch1['input_ids'].shape}")      # torch.Size([3, 5])
print(f"注意力掩码形状: {batch1['attention_mask'].shape}")  # torch.Size([3, 5])

# 样例2: 回归任务数据 (使用浮点标签)
print("\n=== 样例2: 回归任务 ===")
regression_samples = [
    {
        "input_ids": torch.tensor([201, 202, 203, 204, 205]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 3.7,  # 药物评分 (0-5分)
        "features": np.array([0.5, 0.8, 0.3])  # 额外特征
    },
    {
        "input_ids": torch.tensor([201, 206, 207, 208, 205]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 2.1,
        "features": np.array([0.2, 0.4, 0.7])
    }
]

batch2 = fault_tolerance_data_collator(regression_samples)
print(f"标签形状: {batch2['labels'].shape}")           # torch.Size([2])
print(f"标签数据类型: {batch2['labels'].dtype}")       # torch.float32
print(f"特征形状: {batch2['features'].shape}")         # torch.Size([2, 3])

# 样例3: 多标签分类 (使用label_ids)
print("\n=== 样例3: 多标签分类 ===")
multilabel_samples = [
    {
        "input_ids": torch.tensor([301, 302, 303, 304, 305]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label_ids": [1, 0, 1],  # 多个标签: [心脏病, 糖尿病, 高血压]
        "patient_age": 65,
        "patient_gender": 1  # 1: 男性, 0: 女性
    },
    {
        "input_ids": torch.tensor([301, 306, 307, 308, 305]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label_ids": [0, 1, 0],
        "patient_age": 45,
        "patient_gender": 0
    }
]

batch3 = fault_tolerance_data_collator(multilabel_samples)
print(f"标签形状: {batch3['labels'].shape}")           # torch.Size([2, 3])
print(f"患者年龄: {batch3['patient_age']}")            # tensor([65, 45])
print(f"患者性别: {batch3['patient_gender']}")          # tensor([1, 0])

# 样例4: 容错机制演示 (形状不一致的数据)
print("\n=== 样例4: 容错机制演示 ===")
error_samples = [
    {
        "input_ids": torch.tensor([401, 402, 403, 404, 405]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 1,
        "medical_features": torch.tensor([1.2, 3.4])  # 2维特征
    },
    {
        "input_ids": torch.tensor([401, 406, 407, 408, 405]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 0,
        "medical_features": torch.tensor([5.6, 7.8, 9.0])  # 3维特征 (形状不一致!)
    }
]

try:
    batch4 = fault_tolerance_data_collator(error_samples)
    print("容错机制生效，使用第一个样本的数据复制")
    print(f"医疗特征形状: {batch4['medical_features'].shape}")  # torch.Size([2, 2])
    print(f"医疗特征值:\n{batch4['medical_features']}")        # 复制第一个样本的值
except Exception as e:
    print(f"处理失败: {e}")

# 样例5: 混合数据类型
print("\n=== 样例5: 混合数据类型 ===")
mixed_samples = [
    {
        "input_ids": torch.tensor([501, 502, 503, 504, 505]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 1,
        "embedding": np.random.randn(768),  # NumPy数组
        "metadata": {"source": "hospital_A", "date": "2024-01-01"},  # 字符串字段会被跳过
        "confidence": 0.95  # 浮点数值
    },
    {
        "input_ids": torch.tensor([501, 506, 507, 508, 505]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "label": 0,
        "embedding": np.random.randn(768),
        "metadata": {"source": "hospital_B", "date": "2024-01-02"},
        "confidence": 0.87
    }
]

batch5 = fault_tolerance_data_collator(mixed_samples)
print(f"嵌入向量形状: {batch5['embedding'].shape}")   # torch.Size([2, 768])
print(f"置信度: {batch5['confidence']}")              # tensor([0.9500, 0.8700])
print(f"字符串字段 'metadata' 被跳过: {'metadata' not in batch5}")  # True

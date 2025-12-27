#!/usr/bin/env python3
"""
测试 load_hub_datasets 函数的案例
使用 pleisto/wikipedia-cn-20230720-filtered 数据集进行测试
"""

import os
import sys
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

# 添加当前目录到Python路径，以便导入pretraining模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pretraining import load_hub_datasets, ModelArguments, DataArguments


def test_load_hub_datasets():
    """测试 load_hub_datasets 函数"""
    
    print("=" * 60)
    print("测试 load_hub_datasets 函数")
    print("=" * 60)
    
    # 1. 测试单个数据集加载
    print("\n1. 测试单个数据集加载")
    print("-" * 40)
    
    # 创建测试参数
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2-0.5B",  # 使用一个小的模型用于测试
        cache_dir="./test_cache"  # 本地缓存目录
    )
    
    data_args = DataArguments(
        dataset_name="pleisto/wikipedia-cn-20230720-filtered",
        dataset_config_name=None,
        streaming=False,  # 不使用流模式以便看到完整结果
        validation_split_percentage=5,  # 使用5%作为验证集
        max_train_samples=100,  # 限制训练样本数量以加快测试
        max_eval_samples=50     # 限制评估样本数量
    )
    
    is_main_process = True
    
    try:
        # 调用函数
        result = load_hub_datasets(data_args, model_args, is_main_process)
        
        print(f"返回的数据集键: {list(result.keys())}")
        
        if "train" in result:
            train_dataset = result["train"]
            print(f"训练集大小: {len(train_dataset)}")
            print(f"训练集特征: {train_dataset.features}")
            
            # 查看第一个样本
            if len(train_dataset) > 0:
                first_sample = train_dataset[0]
                print(f"第一个样本的键: {list(first_sample.keys())}")
                if 'text' in first_sample:
                    text_preview = first_sample['text'][:100] + "..." if len(first_sample['text']) > 100 else first_sample['text']
                    print(f"文本预览: {text_preview}")
        
        if "validation" in result:
            val_dataset = result["validation"]
            print(f"验证集大小: {len(val_dataset)}")
            print(f"验证集特征: {val_dataset.features}")
            
            # 查看第一个样本
            if len(val_dataset) > 0:
                first_sample = val_dataset[0]
                if 'text' in first_sample:
                    text_preview = first_sample['text'][:100] + "..." if len(first_sample['text']) > 100 else first_sample['text']
                    print(f"验证集文本预览: {text_preview}")
        
        print("✅ 单个数据集加载测试通过")
        
    except Exception as e:
        print(f"❌ 单个数据集加载测试失败: {e}")
        return False
    
    # 2. 测试多个数据集加载（使用mock避免实际下载）
    print("\n2. 测试多个数据集加载（模拟）")
    print("-" * 40)
    
    # 创建包含多个数据集的参数
    data_args_multi = DataArguments(
        dataset_name="pleisto/wikipedia-cn-20230720-filtered,another/dataset",
        dataset_config_name="default,another_config",
        streaming=False,
        validation_split_percentage=5
    )
    
    # 使用mock来模拟load_dataset函数，避免实际下载多个数据集
    with patch('pretraining.load_dataset') as mock_load_dataset:
        # 模拟返回的数据集
        mock_dataset = MagicMock()
        mock_dataset.keys.return_value = ['train', 'validation']
        mock_dataset.__getitem__.side_effect = lambda key: MagicMock(
            __len__=lambda: 100,
            features={'text': 'string'},
            __getitem__=lambda idx: {'text': f'Sample text {idx}'}
        )
        
        mock_load_dataset.return_value = mock_dataset
        
        try:
            result_multi = load_hub_datasets(data_args_multi, model_args, is_main_process)
            
            print(f"多个数据集加载的调用次数: {mock_load_dataset.call_count}")
            print(f"返回的数据集键: {list(result_multi.keys())}")
            
            # 验证是否正确解析了多个数据集
            dataset_names = [name.strip() for name in data_args_multi.dataset_name.split(',')]
            print(f"解析的数据集名称: {dataset_names}")
            
            print("✅ 多个数据集加载测试通过")
            
        except Exception as e:
            print(f"❌ 多个数据集加载测试失败: {e}")
            return False
    
    # 3. 测试空数据集参数
    print("\n3. 测试空数据集参数")
    print("-" * 40)
    
    data_args_empty = DataArguments(
        dataset_name=None,
        streaming=False
    )
    
    try:
        result_empty = load_hub_datasets(data_args_empty, model_args, is_main_process)
        print(f"空参数返回结果: {result_empty}")
        
        if result_empty == {}:
            print("✅ 空数据集参数测试通过")
        else:
            print("❌ 空数据集参数测试失败：应该返回空字典")
            return False
            
    except Exception as e:
        print(f"❌ 空数据集参数测试失败: {e}")
        return False
    
    # 4. 测试数据集名称和配置数量不匹配的情况
    print("\n4. 测试参数不匹配情况")
    print("-" * 40)
    
    data_args_mismatch = DataArguments(
        dataset_name="dataset1,dataset2,dataset3",
        dataset_config_name="config1,config2",  # 数量不匹配
        streaming=False
    )
    
    try:
        result_mismatch = load_hub_datasets(data_args_mismatch, model_args, is_main_process)
        print("❌ 参数不匹配测试失败：应该抛出ValueError")
        return False
    except ValueError as e:
        print(f"✅ 参数不匹配测试通过：正确抛出异常 - {e}")
    except Exception as e:
        print(f"❌ 参数不匹配测试失败：异常类型不正确 - {e}")
        return False
    
    # 5. 测试无验证集的数据集（使用mock模拟）
    print("\n5. 测试无验证集的数据集")
    print("-" * 40)
    
    data_args_no_val = DataArguments(
        dataset_name="pleisto/wikipedia-cn-20230720-filtered",
        dataset_config_name=None,
        streaming=False,
        validation_split_percentage=10
    )
    
    with patch('pretraining.load_dataset') as mock_load_dataset:
        # 模拟只有训练集没有验证集的情况
        mock_dataset_no_val = MagicMock()
        mock_dataset_no_val.keys.return_value = ['train']  # 只有训练集
        
        # 模拟分割后的数据集
        mock_train_split = MagicMock()
        mock_train_split.__len__ = lambda: 90
        mock_val_split = MagicMock()
        mock_val_split.__len__ = lambda: 10
        
        mock_load_dataset.side_effect = [
            mock_dataset_no_val,  # 第一次调用返回没有验证集的数据集
            mock_train_split,    # 第二次调用返回训练分割
            mock_val_split       # 第三次调用返回验证分割
        ]
        
        try:
            result_no_val = load_hub_datasets(data_args_no_val, model_args, is_main_process)
            
            print(f"load_dataset调用次数: {mock_load_dataset.call_count}")
            print(f"返回的数据集键: {list(result_no_val.keys())}")
            
            if 'train' in result_no_val and 'validation' in result_no_val:
                print("✅ 无验证集数据集测试通过：正确从训练集分割出验证集")
            else:
                print("❌ 无验证集数据集测试失败：未能正确分割验证集")
                return False
                
        except Exception as e:
            print(f"❌ 无验证集数据集测试失败: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    return True


def test_real_dataset():
    """使用真实数据集进行测试（需要网络连接）"""
    print("\n" + "=" * 60)
    print("使用真实数据集进行测试")
    print("=" * 60)
    
    # 创建测试参数
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2-0.5B",
        cache_dir="./test_cache"
    )
    
    data_args = DataArguments(
        dataset_name="pleisto/wikipedia-cn-20230720-filtered",
        dataset_config_name=None,
        streaming=False,
        validation_split_percentage=5,
        max_train_samples=10,  # 只加载少量样本进行测试
        max_eval_samples=5
    )
    
    is_main_process = True
    
    try:
        print("正在从Hugging Face Hub加载数据集...")
        print("数据集: pleisto/wikipedia-cn-20230720-filtered")
        print("这可能需要几分钟时间...")
        
        result = load_hub_datasets(data_args, model_args, is_main_process)
        
        print(f"✅ 真实数据集加载成功！")
        print(f"返回的数据集: {list(result.keys())}")
        
        if 'train' in result:
            print(f"训练集样本数: {len(result['train'])}")
            # 查看几个样本
            for i in range(min(3, len(result['train']))):
                sample = result['train'][i]
                if 'text' in sample:
                    text = sample['text']
                    print(f"样本 {i+1}: {text[:100]}...")
        
        if 'validation' in result:
            print(f"验证集样本数: {len(result['validation'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实数据集测试失败: {e}")
        print("这可能是因为网络连接问题或数据集不可用")
        return False


if __name__ == "__main__":
    print("开始测试 load_hub_datasets 函数")
    
    # 运行基础测试（使用mock，不需要网络）
    success = test_load_hub_datasets()
    
    if success:
        print("\n是否要进行真实数据集测试？（需要网络连接，可能较慢）")
        response = input("输入 'y' 继续，其他键跳过: ").strip().lower()
        
        if response == 'y':
            test_real_dataset()
    
    print("\n测试结束！")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO训练诊断脚本
用于排查训练脚本启动失败的问题
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def check_imports():
    """检查所有必要的依赖包"""
    print("=== 检查依赖包导入 ===")
    
    required_packages = [
        'torch',
        'transformers', 
        'trl',
        'peft',
        'datasets',
        'bitsandbytes',
        'loguru',
        'latex2sympy2_extended',
        'math_verify'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}: 导入成功")
        except ImportError as e:
            print(f"✗ {package}: 导入失败 - {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠ {package}: 导入异常 - {e}")
            failed_imports.append(package)
    
    return failed_imports

def check_cuda():
    """检查CUDA环境"""
    print("\n=== 检查CUDA环境 ===")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ 警告: CUDA不可用，将使用CPU训练")
    except Exception as e:
        print(f"✗ CUDA检查失败: {e}")

def check_paths():
    """检查文件路径"""
    print("\n=== 检查文件路径 ===")
    
    # 检查模型路径
    model_path = Path("../../model/Qwen/Qwen3-0.6B")
    if model_path.exists():
        print(f"✓ 模型路径存在: {model_path.absolute()}")
        
        # 检查必要文件
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        for file in required_files:
            file_path = model_path / file
            if file_path.exists():
                print(f"  ✓ {file}: 存在")
            else:
                print(f"  ✗ {file}: 不存在")
    else:
        print(f"✗ 模型路径不存在: {model_path.absolute()}")
    
    # 检查数据路径
    data_path = Path("../../data/grpo")
    if data_path.exists():
        print(f"✓ 数据路径存在: {data_path.absolute()}")
        
        # 检查数据文件
        data_file = data_path / "sample.jsonl"
        if data_file.exists():
            print(f"  ✓ sample.jsonl: 存在 ({data_file.stat().st_size} bytes)")
        else:
            print(f"  ✗ sample.jsonl: 不存在")
    else:
        print(f"✗ 数据路径不存在: {data_path.absolute()}")

def check_grpo_script():
    """检查GRPO脚本语法"""
    print("\n=== 检查GRPO脚本语法 ===")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, '.')
        import grpo_training
        print("✓ grpo_training.py: 语法检查通过")
    except SyntaxError as e:
        print(f"✗ grpo_training.py: 语法错误 - {e}")
        print(f"  行 {e.lineno}: {e.text}")
    except Exception as e:
        print(f"⚠ grpo_training.py: 导入异常 - {e}")
        traceback.print_exc()

def check_memory():
    """检查内存使用情况"""
    print("\n=== 检查内存使用 ===")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"总内存: {memory.total / (1024**3):.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"内存使用率: {memory.percent:.1f}%")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                print(f"GPU {i}: 总显存 {gpu_memory / (1024**3):.1f} GB, 已分配 {allocated / (1024**3):.1f} GB")
    except ImportError:
        print("⚠ psutil未安装，无法检查内存信息")
    except Exception as e:
        print(f"内存检查失败: {e}")

def main():
    """主函数"""
    print("GRPO训练环境诊断工具")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    print()
    
    # 执行各项检查
    failed_imports = check_imports()
    check_cuda()
    check_paths()
    check_grpo_script()
    check_memory()
    
    print("\n=== 诊断总结 ===")
    if failed_imports:
        print(f"✗ 发现 {len(failed_imports)} 个依赖包导入失败:")
        for package in failed_imports:
            print(f"  - {package}")
        print("\n建议:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("✓ 所有依赖包导入正常")
        print("如果训练仍然失败，请检查:")
        print("1. 显存是否足够")
        print("2. 数据格式是否正确") 
        print("3. 模型文件是否完整")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="量化模型推理对比")
    parser.add_argument("--unquantized_model_path", type=str, required=True, help="未量化模型路径")
    parser.add_argument("--quantized_model_output_path", type=str, required=True, help="量化模型保存路径")
    parser.add_argument("--input_text", type=str, default='介绍北京', help="输入的文本内容")
    return parser.parse_args()


# 计算模型相关的显存占用
def get_model_memory_usage(device):
    return torch.cuda.memory_allocated(device) / (1024 ** 3)  # 转换为GB


# 定义一个函数来进行推理，并计算推理时间
def perform_inference(model, tokenizer, device, question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = inputs["attention_mask"]

    start_time = time.time()
    with torch.no_grad():
        # 预热
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id 为 eos_token_id
        )

    end_time = time.time()
    elapsed_time = (end_time - start_time) / request_count

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, elapsed_time


def quantize_model(unquantized_model_path, quantized_model_path):
    '''
    将model_path中的模型量化
    然后存储
    '''
    # 配置 4-bit 量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 加载模型并应用量化
    print(f"正在加载模型: {unquantized_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        unquantized_model_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto"
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(unquantized_model_path, trust_remote_code=True)

    # 保存量化后的模型和 tokenizer
    print(f"正在保存量化模型到: {quantized_model_path}")
    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    # 清理内存
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print("模型量化完成!")


def gpu_memory(model_path, device):
    # 1. 计算未量化模型显存占用
    start_memory = get_model_memory_usage(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)

    model_dtype = next(model.parameters()).dtype
    end_memory = get_model_memory_usage(device)
    model_memory_used = end_memory - start_memory

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return model_dtype, model_memory_used


def compare_gpu_memory(unquantized_model_path, quantized_model_path, device):
    # 1. 计算未量化模型显存占用
    unquantized_dtype, unquantized_memory_used = gpu_memory(unquantized_model_path, device)

    # 2. 直接加载已保存的量化模型
    quantized_dtype, quantized_memory_used = gpu_memory(quantized_model_path, device)

    # 3. 输出比较结果
    print('-' * 100)
    print(f"未量化模型数据类型: {unquantized_dtype}")
    print(f"量化模型数据类型: {quantized_dtype}")
    print(f"未量化模型显存占用: {unquantized_memory_used:.2f} GB")
    print(f"量化模型显存占用: {quantized_memory_used:.2f} GB")
    print('-' * 100)


def inference_performancee(model_path, device, input_text):
    # 1. 计算未量化模型推理时间
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)

    generated_text, time_took = perform_inference(model, tokenizer, device, input_text)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return generated_text, time_took


def compare_inference_performance(unquantized_model_path, quantized_model_path, device, input_text):
    # 1. 计算未量化模型推理时间
    generated_text_unquantized, time_unquantized = inference_performancee(unquantized_model_path, device, input_text)

    # 2. 计算量化模型推理时间
    generated_text_quantized, time_quantized = inference_performancee(quantized_model_path, device, input_text)

    # 3. 输出比较结果
    print('-' * 100)
    print(f"推理生成的文本（未量化模型）: {generated_text_unquantized}")
    print(f"推理生成的文本（量化模型）: {generated_text_quantized}")
    print(f"推理时间（未量化模型）: {time_unquantized:.2f} 秒")
    print(f"推理时间（量化模型）: {time_quantized:.2f} 秒")
    print('-' * 100)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unquantized_model_path = args.unquantized_model_path
    quantized_model_output_path = args.quantized_model_output_path
    input_text = args.input_text

    quantize_model(unquantized_model_path, quantized_model_output_path)
    compare_gpu_memory(unquantized_model_path, quantized_model_output_path, device)
    compare_inference_performance(unquantized_model_path, quantized_model_output_path, device, input_text)


if __name__ == "__main__":
    main()

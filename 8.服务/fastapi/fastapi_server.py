import argparse
import os
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from peft import PeftModel
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
)

from template import get_conv_template


class Item(BaseModel):
    input: str = Field(..., max_length=2048)


class MedicalChatAPI:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer, self.device = self.load_model()
        self.prompt_template = get_conv_template(args.template_name)
        self.stop_str = self.tokenizer.eos_token if self.tokenizer.eos_token else self.prompt_template.stop_str

    @torch.inference_mode()  # 推理模式：关闭梯度计算，减少显存占用、加速推理
    def _stream_generate_answer(
            self,
            prompt,
            do_print=True,  # 是否边生成边打印到 stdout
            max_new_tokens=None,  # 生成的最大新 token 数
            repetition_penalty=None,  # 重复惩罚系数
            context_len=2048,  # 上下文最大长度（模型 context window）
            stop_str=None,  # 停止生成的字符串（如 "</s>"、"###"）
    ):
        """流式生成答案（内部方法）"""

        # 如果未显式传参，则使用全局配置
        max_new_tokens = max_new_tokens or self.args.max_new_tokens
        repetition_penalty = repetition_penalty or self.args.repetition_penalty
        stop_str = stop_str or self.stop_str

        # 创建流式输出器：
        # - timeout：模型超过多久没输出就抛异常
        # - skip_prompt：不把 prompt 本身输出
        # - skip_special_tokens：过滤 <eos> 等特殊 token
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 对 prompt 做 tokenizer 编码，得到 input_ids 和 attention_mask
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]  # shape: [seq_len]
        attention_mask = inputs["attention_mask"][0]  # shape: [seq_len]

        # 计算最大可用的 prompt 长度
        # 需要给生成的 token 预留 max_new_tokens + 额外 buffer（8）
        max_src_len = context_len - max_new_tokens - 8

        # 如果 prompt 太长，只保留最后一段（右截断）
        input_ids = input_ids[-max_src_len:]
        attention_mask = attention_mask[-max_src_len:]

        # 构造 generate 所需参数
        generation_kwargs = dict(
            input_ids=input_ids.unsqueeze(0).to(self.device),  # [1, seq_len]
            attention_mask=attention_mask.unsqueeze(0).to(self.device),
            max_new_tokens=max_new_tokens,  # 最大生成 token 数
            num_beams=1,  # greedy decoding
            repetition_penalty=repetition_penalty,  # 重复惩罚
            streamer=streamer,  # 关键：流式输出
        )

        # 子线程：在单独线程中执行 generate
        # 原因：generate 是阻塞的，streamer 需要并发消费输出
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""  # 累积完整生成结果

        # 主线程：不断从 streamer 中读取新生成的文本片段
        # 流式输出： 服务端 stdout可以看到打印的流式输出
        for new_text in streamer:
            stop = False

            # 检查是否包含 stop_str
            pos = new_text.find(stop_str)
            if pos != -1:
                new_text = new_text[:pos]  # 截断到 stop_str 之前
                stop = True

            # 累加到最终结果
            generated_text += new_text

            # 实时打印（常用于 CLI / server log）
            if do_print:
                print(new_text, end="", flush=True)

            # 命中停止条件，提前终止
            if stop:
                break

        # 换行，保证输出美观
        if do_print:
            print()

        # 客户端： 返回完整生成文本（不包含 stop_str），不能看到流式输出
        return generated_text

    def load_model(self):
        if self.args.only_cpu is True:
            self.args.gpus = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus
        load_type = 'auto'
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            device = torch.device('cpu')
        
        # 先加载模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True,
        )
        try:
            base_model.generation_config = GenerationConfig.from_pretrained(self.args.base_model,
                                                                            trust_remote_code=True)
        except OSError:
            print("Failed to load generation config, use default.")
        
        if self.args.lora_model:
            model = PeftModel.from_pretrained(base_model, self.args.lora_model, dtyp=load_type, device_map='auto')
            print("Loaded lora model")
        else:
            model = base_model
        if device == torch.device('cpu'):
            model.float()
        model.eval()
        
        # 模型加载完成后，再初始化tokenizer
        # 如果tokenizer_path为空，则使用base_model路径
        if self.args.tokenizer_path is None:
            self.args.tokenizer_path = self.args.base_model
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
        except Exception as e:
            print(f"Failed to load tokenizer with force_download: {e}")
            print("Trying to load tokenizer without force_download...")
            
        if self.args.resize_emb:
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            print(f"Vocab of the base model: {model_vocab_size}")
            print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
            if model_vocab_size != tokenzier_vocab_size:
                print("Resize model embeddings to fit tokenizer")
                model.resize_token_embeddings(tokenzier_vocab_size)

        print(tokenizer)
        return model, tokenizer, device

    def predict(self, sentence):
        """
        医疗对话预测函数，使用Qwen模板进行推理
        
        Args:
            sentence (str): 用户输入的医疗相关问题或描述
            
        Returns:
            str: 模型生成的医疗建议或回答
            
        Examples:
            >>> api = MedicalChatAPI(args)
            >>> # 输入样例1：症状咨询
            >>> input1 = "我最近经常头痛，伴有轻微发热，应该怎么办？"
            >>> result1 = api.predict(input1)
            >>> print(result1)
            # 可能输出：建议您注意休息，多喝水，如果症状持续或加重，建议及时就医...
            
            >>> # 输入样例2：药物咨询
            >>> input2 = "感冒了可以服用阿司匹林吗？"
            >>> result2 = api.predict(input2)
            >>> print(result2)
            # 可能输出：阿司匹林可以缓解感冒症状，但建议在医生指导下使用...
        """
        # 构造对话历史，这里只有一轮：
        # [用户输入, 模型回复占位符]，回复先用空字符串占位
        history = [[sentence, '']]

        # 根据对话历史和 system prompt 生成最终送入模型的 prompt
        # prompt_template 通常负责把多轮对话拼成模型可理解的文本格式
        prompt = self.prompt_template.get_prompt(
            messages=history,
            system_prompt=self.args.system_prompt
        )

        # 调用内部的流式生成方法进行推理
        # do_print=False 表示不在生成过程中实时打印到 stdout
        response = self._stream_generate_answer(
            prompt=prompt,
            do_print=False,
        )

        # 去掉首尾多余空白字符，返回最终生成结果
        return response.strip()


def create_app(api_instance):
    # 创建 FastAPI 应用实例
    app = FastAPI()

    # 添加 CORS 中间件，允许跨域访问
    # 这里是完全放开（*），适合内部服务 / 调试阶段
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带 cookie
        allow_methods=["*"],  # 允许所有 HTTP 方法（GET/POST/...）
        allow_headers=["*"]  # 允许所有请求头
    )

    @app.get('/')
    async def index():
        # 根路径健康检查 / 引导接口
        # 通常用于快速确认服务是否正常
        return {"message": "index, docs url: /docs"}

    @app.post('/chat')
    async def chat(item: Item):
        """
        聊天接口
        :param item: 请求体，包含用户输入（item.input）
        :return: 模型生成的回复
        """
        try:
            # 调用底层 API 实例进行预测（同步调用）
            response = api_instance.predict(item.input)

            # 按统一格式封装返回结果
            result_dict = {'response': response}

            # 记录成功日志，方便排查线上问题
            logger.debug(f"Successfully get result, q:{item.input}")

            return result_dict
        except Exception as e:
            # 捕获所有异常，防止接口直接 500 崩掉
            logger.error(e)

            # 这里返回 None，FastAPI 会返回 null
            # 实际生产中建议返回明确的错误结构
            return None

    # 返回构建好的 FastAPI 应用
    return app


def parse_arguments():
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()

    # 基础模型路径，默认指向本地 Qwen3-0.6B 模型目录
    parser.add_argument('--base_model', default="../../model/Qwen/Qwen3-0.6B", type=str, required=False)

    # LoRA 微调模型路径，如果为空，则直接使用 base_model 推理
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")

    # 分词器路径，默认使用 base_model 的分词器
    parser.add_argument('--tokenizer_path', default=None, type=str)

    # 提示模板名称，用于构建模型输入 prompt，例如 qwen, alpaca, vicuna 等
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: qwen, alpaca, vicuna, baichuan, chatglm2 etc.")

    # 系统提示词，用于指示模型的角色或行为
    parser.add_argument('--system_prompt', default="you are a helpfull assistant", type=str)

    # 重复惩罚系数，用于生成文本时抑制重复
    parser.add_argument("--repetition_penalty", default=1.0, type=float)

    # 生成的最大新 token 数量
    parser.add_argument("--max_new_tokens", default=512, type=int)

    # 是否调整模型的 token embedding 大小
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')

    # 使用的 GPU 设备编号，多个 GPU 可用逗号分隔
    parser.add_argument('--gpus', default="0", type=str)

    # 是否只使用 CPU 推理
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')

    # 启动服务的端口号
    parser.add_argument('--port', default=8801, type=int)

    # 解析命令行参数并返回
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    api_instance = MedicalChatAPI(args)
    app = create_app(api_instance)
    uvicorn.run(app=app, host='0.0.0.0', port=args.port, workers=1)


if __name__ == '__main__':
    main()

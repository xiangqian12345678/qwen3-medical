import argparse
import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from template import get_conv_template


# -------------------------
# 参数解析
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str)
    parser.add_argument('--system_prompt', default="", type=str)
    parser.add_argument('--context_len', default=2048, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--only_cpu', action='store_true')
    parser.add_argument('--resize_emb', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', default=8081, type=int)
    return parser.parse_args()


# -------------------------
# 模型相关
# -------------------------
def get_device(only_cpu):
    return torch.device("cuda") if torch.cuda.is_available() and not only_cpu else torch.device("cpu")


def load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def load_base_model(base_model_path, load_type):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=load_type,
        trust_remote_code=True,
    )
    try:
        model.generation_config = GenerationConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        )
    except OSError:
        print("⚠️ Failed to load generation config, using default.")
    return model


def resize_model_embeddings(model, tokenizer):
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def load_lora_model(base_model, lora_model_path, load_type):
    return PeftModel.from_pretrained(
        base_model, lora_model_path, device_map="auto", torch_dtype=load_type
    )


def setup_model(model, device):
    if device.type == "cpu":
        model.float()
    model.eval()
    return model


# -------------------------
# 生成函数（messages → prompt → answer）
# -------------------------
import re

def strip_qwen_think(text: str) -> str:
    # 移除 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def create_predict_function(
    model,
    tokenizer,
    prompt_template,
    system_prompt,
    context_len,
    max_new_tokens,
    device,
):
    def predict(messages):
        """
        messages: List[{"role": "user"|"assistant", "content": "..."}]
        """

        # messages → [[user, assistant], ...]
        pairs = []
        cur_user = None

        for msg in messages:
            if msg["role"] == "user":
                cur_user = msg["content"]
            elif msg["role"] == "assistant" and cur_user is not None:
                pairs.append([cur_user, msg["content"]])
                cur_user = None

        # 如果最后是 user，没有 assistant（当前轮）
        if cur_user is not None:
            pairs.append([cur_user, ""])

        prompt = prompt_template.get_prompt(
            messages=pairs,
            system_prompt=system_prompt,
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0][- (context_len - max_new_tokens - 8):]
        attention_mask = inputs["attention_mask"][0][- (context_len - max_new_tokens - 8):]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        raw_answer = tokenizer.decode(
            output_ids[0][input_ids.size(0):],
            skip_special_tokens=True,
        ).strip()

        # qwen3模型有think部分，这部分过程，需要删除，以防止得不到答案
        answer = strip_qwen_think(raw_answer)

        return answer

    return predict



# -------------------------
# Gradio UI（核心修复点）
# -------------------------
def launch_gradio_interface(predict_func, share, port):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 💬 Chatbot Demo")

        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(
            placeholder="Ask me question",
            lines=6,
            show_label=False,
        )

        with gr.Row():
            send_btn = gr.Button("发送")
            clear_btn = gr.Button("清空")

        def user_input_and_generate(user_message, history):
            history = history or []

            # 1️⃣ 先把 user 放进 messages
            history.append({"role": "user", "content": user_message})

            # 2️⃣ 用「完整 messages」生成
            answer = predict_func(history)

            # 3️⃣ 再 append assistant
            history.append({"role": "assistant", "content": answer})

            return "", history

        send_btn.click(
            user_input_and_generate,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        clear_btn.click(lambda: [], outputs=chatbot)

        demo.launch(
            share=share,
            inbrowser=True,
            server_name="0.0.0.0",
            server_port=port,
        )


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    device = get_device(args.only_cpu)
    load_type = "auto"

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = load_tokenizer(args.tokenizer_path)
    base_model = load_base_model(args.base_model, load_type)

    if args.resize_emb:
        resize_model_embeddings(base_model, tokenizer)

    if args.lora_model:
        model = load_lora_model(base_model, args.lora_model, load_type)
    else:
        model = base_model

    model = setup_model(model, device)

    prompt_template = get_conv_template(args.template_name)

    predict_func = create_predict_function(
        model=model,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        system_prompt=args.system_prompt,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    launch_gradio_interface(predict_func, args.share, args.port)


if __name__ == "__main__":
    main()

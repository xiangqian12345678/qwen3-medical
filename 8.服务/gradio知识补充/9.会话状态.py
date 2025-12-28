# 访问地址： http://localhost:8802
import random
import gradio as gr


def chat(message, history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = str(random.randint(1, 10))
    elif message.startswith("how"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    
    # 添加新消息到历史记录（使用openai格式）
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history, history


# 设置一个对话窗
chatbot = gr.Chatbot(type="messages")
demo = gr.Interface(
    chat,
    [gr.Textbox(), gr.State()],
    [gr.Textbox(), chatbot, gr.State()],
    allow_flagging="never",
)
demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

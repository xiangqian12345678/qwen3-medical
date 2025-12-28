# 访问地址： http://localhost:8802
import gradio as gr


def welcome(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Hello World!
        Start typing below to see the output.
        """)
    inp = gr.Textbox(placeholder="What is your name?")
    out = gr.Textbox()
    # 设置change事件
    inp.change(fn=welcome, inputs=inp, outputs=out)

demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

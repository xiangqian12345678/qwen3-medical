# 访问地址： http://localhost:8802

import gradio as gr

input_textbox = gr.Textbox()
with gr.Blocks() as demo:
    # 提供示例输入给input_textbox，示例输入以嵌套列表形式设置
    gr.Examples(["hello", "bonjour", "merhaba"], input_textbox)
    # render函数渲染input_textbox
    input_textbox.render()

demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

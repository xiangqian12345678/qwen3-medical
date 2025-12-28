# 访问地址： http://localhost:8802
import gradio as gr


def greet(name):
    return "Hello " + name + "!"


demo = gr.Interface(
    fn=greet,
    # 自定义输入框
    # 具体设置方法查看官方文档
    inputs=gr.Textbox(lines=3, placeholder="Name Here...", label="my input"),
    outputs="text",
)

# 使用不同端口并设置启动参数解决冲突问题
demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

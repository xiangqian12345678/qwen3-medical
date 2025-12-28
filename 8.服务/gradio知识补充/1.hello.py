# 访问地址： http://localhost:8802

import gradio as gr


# 输入文本处理程序
def greet(name):
    return "Hello " + name + "!"


# 接口创建函数
# fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
# fn,inputs,outputs都是必填函数
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# 使用不同端口并设置启动参数解决冲突问题
demo.launch(
    server_name="0.0.0.0",    # 允许所有接口访问
    server_port=8802,         # 使用8802端口避免冲突
    share=False,              # 不创建公共链接
    show_error=True,          # 显示错误信息
    quiet=False,              # 显示启动信息
    prevent_thread_lock=False # 防止线程锁定
)

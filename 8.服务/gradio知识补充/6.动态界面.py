# 访问地址： http://localhost:8802
import gradio as gr


def calculator(num1, operation, num2):
    # 检查输入是否为空
    if num1 is None or num2 is None:
        return "请输入两个数字"
    
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            return "除数不能为零"
        return num1 / num2


iface = gr.Interface(
    calculator,
    [gr.Number(), gr.Radio(["add", "subtract", "multiply", "divide"]), gr.Number()],
    gr.Number(),
    live=True,
)

iface.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

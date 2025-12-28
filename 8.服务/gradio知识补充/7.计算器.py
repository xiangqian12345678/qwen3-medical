# 访问地址： http://localhost:8802
import gradio as gr


# 一个简单计算器，含实例说明
def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            # 设置报错弹窗
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2


demo = gr.Interface(
    calculator,
    # 设置输入
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    # 设置输出
    "number",
    # 设置输入参数示例
    examples=[
        [5, "add", 3],
        [4, "divide", 2],
        [-4, "multiply", 2.5],
        [0, "subtract", 1.2],
    ],
    # 设置网页标题
    title="Toy Calculator",
    # 左上角的描述文字
    description="Here's a sample toy calculator. Enjoy!",
    # 左下角的文字
    article="Check out the examples",
)
demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

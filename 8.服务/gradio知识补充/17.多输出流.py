# 访问地址： http://localhost:8802
import gradio as gr

with gr.Blocks() as demo:
    food_box = gr.Number(value=10, label="Food Count")
    status_box = gr.Textbox()


    def eat(food):
        if food > 0:
            return food - 1, "full"
        else:
            return 0, "hungry"


    gr.Button("EAT").click(
        fn=eat,
        inputs=food_box,
        # 根据返回值改变输入组件和输出组件
        outputs=[food_box, status_box]
    )

demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

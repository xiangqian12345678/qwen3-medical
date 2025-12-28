# 访问地址： http://localhost:8802
import gradio as gr
import numpy as np
import time


# 生成steps张图片，每隔1秒钟返回
def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.randint(255, size=(300, 600, 3))
        yield image


demo = gr.Interface(fake_diffusion,
                    # 设置滑窗，最小值为1，最大值为10，初始值为3，每次改动增减1位
                    inputs=gr.Slider(1, 10, value=3, step=1),
                    outputs="image")
# 生成器必须要queue函数
demo.queue()

demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

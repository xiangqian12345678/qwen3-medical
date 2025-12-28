# 访问地址： http://localhost:8802
import gradio as gr

scores = []


def track_score(score):
    scores.append(score)
    # 返回分数top3
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores


demo = gr.Interface(
    track_score,
    gr.Number(label="Score"),
    gr.JSON(label="Top Scores")
)
demo.launch(
    server_name="0.0.0.0",  # 允许所有接口访问
    server_port=8802,  # 使用8802端口避免冲突
    share=False,  # 不创建公共链接
    show_error=True,  # 显示错误信息
    quiet=False,  # 显示启动信息
    prevent_thread_lock=False  # 防止线程锁定
)

import gradio as gr

def greet(name):
    return "Hello " + name + "!"

iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch(
    share=False,
    server_port=8802,
    server_name="0.0.0.0",
    prevent_thread_lock=False
)
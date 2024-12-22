import gradio as gr


def another_function(input_text):
    return f"You entered: {input_text}"


iface3 = gr.Interface(
    fn=another_function,
    inputs="text",
    outputs="text",
    title="Новое окно",
    description="Введите текст, чтобы увидеть его обратно."
)

import gradio as gr

def greet(name):
  return "Hello " + name + "! This program was run on Kahan :)"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch(share=True)

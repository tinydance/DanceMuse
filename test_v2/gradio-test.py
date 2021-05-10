import gradio as gr
import os
import shutil

def test(audio):
	filename = os.path.splitext(audio)[0]
	shutil.move(audio,'test/raw_audio/')
	return_code = subprocess.call("./test/test.sh")
	video_path = "test/edited_output/"+filename+"_with_audio.mp4"
	return video_path

iface = gr.Interface(fn=test, inputs="audio", outputs="video")
iface.launch(share=True)

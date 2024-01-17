"""
Requirement
- cuDNN
- NVIDIA GPU
SEE ALSO:
https://huggingface.co/deepmind/optical-flow-perceiver

https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Optical_Flow.ipynb

TODO:
 not working yet even in Jetson.

"""

import requests
from PIL import Image
from transformers import pipeline
# from perceiver.model.vision import optical_flow  # register optical flow pipeline

frame_1 = Image.open(requests.get("https://martin-krasser.com/perceiver/flow/frame_0047.png", stream=True).raw)
frame_2 = Image.open(requests.get("https://martin-krasser.com/perceiver/flow/frame_0048.png", stream=True).raw)

optical_flow_pipeline = pipeline("optical-flow", model="krasserm/perceiver-io-optical-flow", device="cuda:0")
rendered_optical_flow = optical_flow_pipeline((frame_1, frame_2), render=True)

Image.fromarray(rendered_optical_flow).save("optical_flow.png")

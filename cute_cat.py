import requests
from PIL import Image
from transformers import pipeline

import cv2
import numpy as np

def draw_detection(image, result):
    for d in result:
        xmin = d["box"]["xmin"]
        xmax = d["box"]["xmax"]
        ymin = d["box"]["ymin"]
        ymax = d["box"]["ymax"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(image, d["label"], (xmin, ymin),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
        )

    return image


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

# Download an image with cute cats
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
image_data = requests.get(url, stream=True).raw
image = Image.open(image_data)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')
result = object_detector(image)
print(result)
cvimg = pil2cv(image)
image = draw_detection(cvimg, result)
cv2.imwrite("junk.jpg", cvimg)


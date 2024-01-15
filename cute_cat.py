import requests
from PIL import Image
from transformers import pipeline

import cv2
import numpy as np


from names2num import get_label2num_dict

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

_COLORS_UINT8 = (255 * _COLORS).astype(np.uint8)

LABEL2NUM = get_label2num_dict()

def draw_detection(image, result):
    for i, d in enumerate(result):
        xmin = d["box"]["xmin"]
        xmax = d["box"]["xmax"]
        ymin = d["box"]["ymin"]
        ymax = d["box"]["ymax"]
        r, g, b = _COLORS_UINT8[i, :]
        num = LABEL2NUM.get(d["label"], 80) - 1
        r, g, b = _COLORS_UINT8[num, :]
        color = (int(r), int(g), int(b))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(image, d["label"], (xmin, ymin),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=color,
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


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def detect_url_image(url):
    image_data = requests.get(url, stream=True).raw
    image = Image.open(image_data)

    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection', model="devonho/detr-resnet-50_finetuned_cppe5")
    result = object_detector(image)
    print(result)
    cvimg = pil2cv(image)
    oimage = draw_detection(cvimg, result)
    cv2.imwrite("junk.jpg", oimage)

def detect_for_video():
    object_detector = pipeline('object-detection', model="devonho/detr-resnet-50_finetuned_cppe5")
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("transformers", cv2.WINDOW_NORMAL)
    while True:
        ret, cvimg = cap.read()
        if cvimg is None:
            break

        image = cv2pil(cvimg)
        result = object_detector(image)
        oimage = draw_detection(cvimg, result)
        cv2.imshow("transformers", oimage)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":
    if 1:
        # Download an image with cute cats
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
        detect_url_image(url)
    else:
        detect_for_video()

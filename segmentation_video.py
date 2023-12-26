"""
HuggingFace を用いたsegmentation
"""

from pathlib import Path

from transformers import pipeline
from PIL import Image
import requests

import cv2

from cute_cat import cv2pil, pil2cv

from segmentation_color import SEGMENTATION_COLORS
from segmentation_name import SEGMENTATION_NAME2COLORS

def colorize_segmantation(results):
    pass


OUTDIR = Path(__name__).parent / "results"
OUTDIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="image segmentation")
    parser.add_argument("src", help="video source")
    args = parser.parse_args()
    # semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")  # cityscapesでの学習済みモデル

    semantic_segmentation = pipeline("image-segmentation", "facebook/maskformer-swin-tiny-coco")  # MS COCO での学習済みモデル

    print(args.src)
    if args.src.find("/dev/video") >= 0:
        src = int(args.src.replace("/dev/video", ""))
    else:
        src = args.src

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.namedWindow("transformers", cv2.WINDOW_NORMAL)
    while True:
        cap.grab()
        cap.grab()
        cap.grab()
        cap.grab()
        cap.grab()
        cap.grab()
        ret, cvimg = cap.read()
        if cvimg is None:
            break

        image = cv2pil(cvimg)
        print(f"{image.size}=")
        results = semantic_segmentation(image)
        colorized = image.copy()
        result_pil = Image.new("L", image.size)
        for i, result in enumerate(results):
            print(f"""{i} {result=}""")
            if results[i]["label"] == "undefined":
                continue
            elif results[i]["label"] in SEGMENTATION_NAME2COLORS:
                label = results[i]["label"]
                result_pil = results[i]["mask"]
                mask = results[i]["mask"]

                r, g, b = SEGMENTATION_NAME2COLORS[label]
                monocolor = Image.new("RGB", image.size, (r, g, b))
                colorized.paste(monocolor, (0, 0), mask)
            else:
                label = results[i]["label"]
                print(f"warn: no such key {label}")
        result_cv = pil2cv(colorized)
        cv2.imshow("transformers", result_cv)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

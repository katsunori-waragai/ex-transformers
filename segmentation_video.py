from pathlib import Path

from transformers import pipeline
from PIL import Image
import requests

import cv2

from cute_cat import cv2pil, pil2cv

OUTDIR = Path(__name__).parent / "results"
OUTDIR.mkdir(exist_ok=True)

semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")

cap = cv2.VideoCapture(0)
cv2.namedWindow("transformers", cv2.WINDOW_NORMAL)
while True:
    ret, cvimg = cap.read()
    if cvimg is None:
        break

    image = cv2pil(cvimg)
    results = semantic_segmentation(image)
    result_pil = results[-1]["mask"]
    result_cv = pil2cv(result_pil)
    cv2.imshow("transformers", result_cv)
    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break
cv2.destroyAllWindows()


oname = OUTDIR / "mask.png"
results[-1]["mask"].save(str(oname))

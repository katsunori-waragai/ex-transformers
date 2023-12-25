from pathlib import Path

from transformers import pipeline
from PIL import Image
import requests

OUTDIR = Path(__name__).parent / "results"
OUTDIR.mkdir(exist_ok=True)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)

semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)

oname = OUTDIR / "mask.png"
results[-1]["mask"].save(str(oname))

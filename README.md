# ex-transformers
examples for transformers

## Requirements
pillow
timm
torch


```commandline
pip install -r requirements.txt
```

## examples
https://pypi.org/project/transformers/README.md にあるサンプルをpythonファイルにしたもの
### sentiment-analysis.py
感情分析の例 

```commandline
python3 sentiment-analysis.py
(中略)
We are very happy to introduce pipeline to the transformers repository.
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
English text: good morning.
good morning.
[{'label': 'POSITIVE', 'score': 0.9998468160629272}]
English text: Traceback (most recent call last):
  File "sentiment-analysis.py", line 11, in <module>
    text = input("English text: ")
EOFError
```
Jetson Docker環境でも動作した。

cute_cat.py  
物体検出の例

```commandline
$ python cute_cat.py --image coco_sample.png 
Namespace(image='coco_sample.png', url=None, video=None)
No model was supplied, defaulted to facebook/detr-resnet-50 and revision 2729413 (https://huggingface.co/facebook/detr-resnet-50).
Using a pipeline without specifying a model name and revision in production is not recommended.
Some weights of the model checkpoint at facebook/detr-resnet-50 were not used when initializing DetrForObjectDetection: ['model.backbone.conv_encoder.model.layer4.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer1.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer2.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer3.0.downsample.1.num_batches_tracked']
- This IS expected if you are initializing DetrForObjectDetection from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DetrForObjectDetection from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.
The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.
[{'score': 0.9982202649116516, 'label': 'remote', 'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}, {'score': 0.9960021376609802, 'label': 'remote', 'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}, {'score': 0.9954745173454285, 'label': 'couch', 'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}, {'score': 0.99880051612854, 'label': 'cat', 'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}, {'score': 0.9986782670021057, 'label': 'cat', 'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

clip_example.py  

segmentation_video.py
画像セグメンテーション（動画入力)

```commandline
$ python segmentation_video.py /dev/video0
```

## Jetson AGX Orin
Docker ファイルを作成して環境を構築中。
```
bash docker_build.sh
bash docker_run.sh
```


## TODO
- 物体検出スクリプト　cute_cat.pyがJetsonのDocker環境で検出できていない。
- no detection results in Jetson Docker environment

```
pip install -q datasets transformers evaluate timm albumentations
```
に指定されているライブラリを追加した。
しかし、まだJetsonのDocker環境で検出結果がないままである。

## SEE ALSO
https://pypi.org/project/transformers/


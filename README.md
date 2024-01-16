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
sentiment-analysis.py
感情分析の例

cute_cat.py  
物体検出の例

clip_example.py  

segmentation_video.py
画像セグメンテーション（動画入力)


## Jetson AGX Orin
Docker ファイルを作成して環境を構築中。

```
bash docker_build.sh
bash docker_run.sh
```


## troubleshooting
- no detection results in Jetson Docker environment

```
pip install -q datasets transformers evaluate timm albumentations
```
に指定されているライブラリを追加した。
しかし、まだJetsonのDocker環境で検出結果がないままである。

## SEE ALSO
https://pypi.org/project/transformers/


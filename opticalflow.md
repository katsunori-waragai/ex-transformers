## NVIDIA
[NVIDIA Optical Flow SDK](https://developer.nvidia.com/optical-flow-sdk)
Optical flow

解説記事 [OpenCVのcudaoptflowモジュールからNVIDIA Optical Flow SDKを使う](https://qiita.com/dandelion1124/items/ebe51b683ed4f285a4b1)

NVIDIAの標準を使った方が格段に速いはずである。

## Optical flow の計算例
以下の例はhuggingFace の情報を元にtransformer で実装済みのモデルを実行するための参考例です。

https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Optical_Flow.ipynb

from transformers import PerceiverForOpticalFlow
を使っている。

このipynb ファイルはGPUのないノートPCでも途中まで動作する。
Gradio demo　の部分で動作に失敗する。


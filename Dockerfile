FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
RUN apt update
RUN apt install sudo
RUN apt-get install -y zip
RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get install -y libv4l-dev v4l-utils qv4l2
RUN apt-get install -y curl
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN python3 -m pip install -U pip
RUN python3 -m pip install loguru tqdm thop ninja tabulate
RUN python3 -m pip install pycocotools
RUN python3 -m pip install timm
RUN python3 -m pip install "opencv-python<4.3" pillow scipy transformers datasets evaluate timm albumentations
# RUN python3 -m pip install perceiver


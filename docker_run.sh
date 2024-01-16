#!/bin/bash
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v $HOME/github/ex-transformers:/root/ex-transformers:rw -v /tmp/.X11-unix/:/tmp/.X11-unix ex-transformer:100

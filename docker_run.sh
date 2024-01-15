#!/bin/bash
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v $HOME/github/ex-transformers:/root/ex-transformers -v /tmp/.X11-unix/:/tmp/.X11-unix ex-transformer:100

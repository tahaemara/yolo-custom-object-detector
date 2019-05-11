#!/bin/bash

xhost +"local:docker@"
sudo nvidia-docker  run  -it --entrypoint "/bin/bash" --env DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix yolo3-opencv-cuda

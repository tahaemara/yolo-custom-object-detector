# Docker Image for Darknet, OpenCV 3.3, and CUDA.

Dockerfile to build a docker image contain Darknet, OPENCV, and CUDA. Both Darknet and Opencv were compiled with CUDA.

### To build the Image 
<ol>

<li>Install nvidia docker by following the instructions <a href="https://github.com/NVIDIA/nvidia-docker/blob/master/README.md">here</a>.</li>
<li>Change directory to Dockerfile location.</li>
<li>Build the docker image  <pre>sudo nvidia-docker build -t yolo3-opencv-cuda .</pre></li>
</ol>

### To Test the image
<ol>
<li>Run this command in terminal <pre>xhost +"local:docker@"</pre></li>
<li>Run image via this command <pre>sudo nvidia-docker  run  -it  --rm --env DISPLAY=unix$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix yolo3-opencv-cuda</pre></li>
<li>The output will be this demo</li></ol>
<p align="center"><img src="https://github.com/tahaemara/yolo-custom-object-detector/blob/master/docker/test%20output.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/yolo-custom-object-detector/blob/master/docker/test%20output.png?raw=true" /></p>

### To use the image 

<pre>xhost +"local:docker@"
sudo nvidia-docker  run  -it --entrypoint "/bin/bash" --env DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix yolo3-opencv-cuda</pre>


	
 

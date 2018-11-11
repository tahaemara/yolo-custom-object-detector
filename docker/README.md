# Docker Image for Darknet, OpenCV 3.3, and CUDA.

Dockerfile to build a docker image contain Darknet, OPENCV, and CUDA. Both Darknet and Opencv were compiled with CUDA.

### To build the Image 
<ol>
<li>Install nvidia docker 

```sh
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

</li>
<li>Change directory to Dokerfile location.</li>
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


	
 

## Dockerfile to build a docker image contain Darknet, OPENCV, and CUDA.
## Author : Taha Emara, Email:taha@emaraic.com




FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ARG https_proxy
ARG http_proxy


###  OPENCV INSTALL  ###

RUN apt-get update --fix-missing && apt-get install -qy \
	cmake \
	python-numpy python-scipy python-pip python-setuptools \
	python3-numpy python3-scipy python3-pip python3-setuptools \
	wget \
	xauth \
	libjpeg-dev libtiff5-dev libjasper1 libjasper-dev libpng12-dev libavcodec-dev libavformat-dev \
	libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev \
	gfortran python2.7-dev python3-dev build-essential pkg-config 



# Build OpenCV 3.3.0
RUN \
	cd /root && \
	wget https://github.com/opencv/opencv/archive/3.3.0.tar.gz -O opencv.tar.gz && \
	tar zxf opencv.tar.gz && rm -f opencv.tar.gz && \
	wget https://github.com/opencv/opencv_contrib/archive/3.3.0.tar.gz -O contrib.tar.gz && \
	tar zxf contrib.tar.gz && rm -f contrib.tar.gz && \
	cd opencv-3.3.0 && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-3.3.0/modules \
	-D WITH_CUDA=ON \
	-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
	-D BUILD_DOCS=OFF \
	-D BUILD_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_python3=ON \
	-D WITH_1394=OFF \
	-D WITH_MATLAB=OFF \
	-D WITH_OPENCL=OFF \
	-D WITH_OPENCLAMDBLAS=OFF \
	-D WITH_OPENCLAMDFFT=OFF \
	-D CMAKE_CXX_FLAGS="-O3 -funsafe-math-optimizations" \
	-D CMAKE_C_FLAGS="-O3 -funsafe-math-optimizations" \
	.. && make && make install && \
	cd /root && rm -rf opencv-3.3.0 opencv_contrib-3.3.0

###  DARKNET INSTALL  ###
RUN     apt-get -qy install git-core

RUN  	cd / \
	&& git clone https://github.com/pjreddie/darknet \
	&& cd darknet \
	&& sed -i 's/GPU=0/GPU=1/g' Makefile \
	&& sed -i 's/OPENCV=0/OPENCV=1/g' Makefile \
	&& make 

###Download YOLO v3 tiny weights  
RUN 	cd / \
	&& mkdir weights && cd /weights \
	&& wget https://pjreddie.com/media/files/yolov3-tiny.weights 

#RUN   	export PATH=$PATH:/darknet/darknet
RUN apt-get install -qqy x11-apps 



WORKDIR /darknet
#uncomment the following line to prevent runing docker image alongwith the darknet executable file
ENTRYPOINT [ "./darknet","-nogpu", "detect" , "/darknet/cfg/yolov3-tiny.cfg" , "/weights/yolov3-tiny.weights" , "/darknet/data/dog.jpg" ] 


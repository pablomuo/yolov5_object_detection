# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Builds ultralytics/yolov5:latest image on DockerHub https://hub.docker.com/r/ultralytics/yolov5
# Image is CUDA-optimized for YOLOv5 single/multi-GPU training and inference

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.05-py3

ARG IP_TX
ARG IP_SB
ARG PORT_TX
ARG PORT_RX
ARG PORT_SB
ARG CONF_MIN

RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python<4.6.0.66' \
    --extra-index-url https://download.pytorch.org/whl/cu113

#set Gstreamer
RUN apt-get update 
RUN apt-get upgrade -y

# Install gstreamer and opencv dependencies
RUN \
	DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN \
	apt-get install -y \
	libgstreamer1.0-0 \
	gstreamer1.0-plugins-base \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav \
	gstreamer1.0-doc \
	gstreamer1.0-tools \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev


RUN apt-get install -y git

# just for testing
RUN apt-get -y install nano net-tools netcat

# setup python
RUN apt-get install -y python3-pip

# install mavlink dependencies: https://github.com/ArduPilot/pymavlink
RUN apt-get install -y gcc python3-dev libxml2-dev libxslt-dev

RUN pip3 install numpy future lxml pymavlink

# to run websockets
RUN pip3 install websockets

# get opencv and build it
RUN git clone https://github.com/opencv/opencv.git

RUN apt-get install -y build-essential libssl-dev

RUN apt-get -y install cmake
RUN apt-get -y install pkg-config
RUN apt-get -y install libgtk-3-dev
RUN \
	cd opencv && \
	git checkout 4.5.4 && \
	git submodule update --recursive --init && \
	mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D PYTHON_EXECUTABLE=$(which python3) \
	-D BUILD_opencv_python2=OFF \
	-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
	-D PYTHON3_EXECUTABLE=$(which python3) \
	-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
	-D WITH_GSTREAMER=ON \
	-D WITH_GTK=ON \
	-D BUILD_EXAMPLES=ON .. && \
	make -j$(nproc) && \
	make install && \
	ldconfig
RUN pip install imutils
RUN apt-get -y install libgtk2.0-dev

# Create working directory
RUN mkdir -p /usr/src/yolov5
WORKDIR /usr/src/yolov5

# Copy contents
#COPY . /usr/src/yolov5
#RUN git clone https://github.com/pablomuo/yolov5.git /usr/src/yolov5/project
RUN git clone https://github.com/pablomuo/yolov5_object_detection /usr/src/yolov5

# Set environment variables
ENV OMP_NUM_THREADS=8

CMD python ejecutar.py ${IP_TX} ${IP_SB} ${PORT_TX} ${PORT_RX} ${PORT_SB} ${CONF_MIN}


FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git libgl1-mesa-glx libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev

RUN apt-get install -y cmake wget unzip ninja-build

WORKDIR /

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.6.0.zip

## Install OpenCV
RUN unzip opencv.zip && rm opencv.zip

RUN mkdir /opencv

WORKDIR /opencv

RUN cmake -S ../opencv-4.6.0 -B . -GNinja \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DCMAKE_BUILD_TYPE=Release

RUN ninja

## Install LibTorch

WORKDIR /

## Based on your cuda version, you can change the cu121 to cuXXX
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip 

RUN unzip libtorch.zip && rm libtorch.zip

WORKDIR /workdir

CMD ["/bin/bash"]
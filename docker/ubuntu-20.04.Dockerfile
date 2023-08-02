#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARG CUDA_VERSION=12.0.1

# Multi-arch container support available in non-cudnn containers.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV TRT_VERSION 8.6.1.6
SHELL ["/bin/bash", "-c"]

RUN mkdir -p /home/workspace
RUN mkdir -p /home/catkin_ws/src/yolov7_trt_ros

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT. This will also pull in CUDNN
RUN v="${TRT_VERSION}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get -y install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v} libnvinfer-dispatch8=${v} libnvinfer-dispatch-dev=${v} libnvinfer-lean8=${v} \
        libnvinfer-lean-dev=${v} libnvinfer-vc-plugin8=${v} libnvinfer-vc-plugin-dev=${v} \
        libnvinfer-headers-dev=${v} libnvinfer-headers-plugin-dev=${v};

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab
RUN pip3 install --upgrade numpy

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/aarch64-linux-gnu/
ENV TRT_OSSPATH /home/workspace/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"

# RUN mkdir -p /home/catkin_ws/src/yolov7_trt_ros
WORKDIR /home/catkin_ws

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools

# install ros packages
RUN apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1*

ENV PATH="${PATH}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/share"

# INSTALLING yolov7_trt_ros
COPY . /home/catkin_ws/src/yolov7_trt_ros
RUN pip install matplotlib
RUN apt-get install -y --no-install-recommends ros-noetic-vision-msgs ros-noetic-cv-bridge
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /home/catkin_ws; catkin_make'

# setup entrypoint
COPY ./ros_entrypoint.sh .
ENTRYPOINT ["/home/catkin_ws/ros_entrypoint.sh"]
CMD ["bash"]
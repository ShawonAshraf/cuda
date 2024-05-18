from nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace
RUN apt-get update && apt-get install cmake -y
RUN apt-get install -y python3 python3-pip
RUN pip install conan

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=Ampere

WORKDIR /content

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git nano python3 python-is-python3  python3-pip && \
    apt-get install -y tmux gedit libglew-dev libassimp-dev libboost-all-dev && \
    apt-get install -y libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev && \
    apt-get install -y libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \ 
    apt-get install -y cmake ninja-build && \
    apt-get install -y python3-tk && \
    rm -rf /var/lib/apt/lists/*


RUN pip install plyfile torch torchvision torchaudio tqdm opencv-python numpy

WORKDIR /content
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive && \
    cd gaussian-splatting && \
    pip install submodules/diff-gaussian-rasterization submodules/simple-knn && \
    cd .. && \
    rm -rf gaussian-splatting

RUN pip install matplotlib open3d scipy yaspin mmcv==1.6.0 argparse lpips pytorch-msssim
RUN pip install debugpy
RUN pip install tensorboard
RUN pip install imageio[ffmpeg]
RUN pip install roma
RUN pip install pandas
RUN pip install tables
RUN pip install Flask flask-socketio
RUN pip install gunicorn eventlet
RUN pip install requests pillow
RUN pip install ray[tune]
RUN pip install optuna
RUN pip install gdown

# COPY ./webserver /content/webserver

CMD cd /content/webserver && tmux new-session -d -s webserver "python main.py" && cd .. && /bin/bash


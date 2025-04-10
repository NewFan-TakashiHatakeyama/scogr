FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set shell and avoid stuck build due to user prompt
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install base packages and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    wget \
    ninja-build \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDNN_INCLUDE_DIR=/usr/include
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Set NumPy environment variables
ENV NPY_DISABLE_CPU_FEATURES=1
ENV PYTHONWARNINGS="ignore"

# Upgrade pip and install numpy first
RUN pip3 install --no-cache-dir pip --upgrade
RUN pip3 install --no-cache-dir numpy==1.23.5

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121


# Copy and install requirements - only rebuild if requirements change
COPY requirements.txt /workspace/

# Create dataset directory structure and copy dataset
COPY dentex_dataset /workspace/datasets/dentex_dataset


RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir psutil timm

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Copy the rest of the application - this should be the last step
# We don't COPY . at this stage to avoid full rebuilds
# The docker-compose volume mount will provide the code at runtime 
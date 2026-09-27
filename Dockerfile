# syntax=docker/dockerfile:1.7

# Stateless RunPod ComfyUI image.
# CUDA 12.9 is new enough for SageAttention 2.2 / Blackwell while retaining
# broad support for Ampere, Ada and Hopper GPUs.
ARG CUDA_IMAGE=nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_SAGEATTENTION=1
ARG SAGEATTENTION_VERSION=2.2.0
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    HF_HOME=/workspace/.cache/huggingface \
    COMFY_HOME=/workspace/ComfyUI \
    CONFIG_HOME=/workspace/config \
    COMFY_PORT=8188 \
    ATTENTION_BACKEND=auto \
    ASSET_DOWNLOAD_WORKERS=4 \
    COMFY_REF=master \
    CONFIG_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
      aria2 \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      jq \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      nano \
      ninja-build \
      openssh-server \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      rsync \
      tini \
      unzip \
      wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && git lfs install --system \
    && mkdir -p /run/sshd /root/.ssh /workspace

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && python -m pip install \
      huggingface_hub \
      requests \
      rich \
      jupyterlab

# Build SageAttention into the image for the major NVIDIA architectures that
# RunPod commonly offers. This avoids recompiling it on every disposable Pod.
# Set --build-arg INSTALL_SAGEATTENTION=0 for a smaller/faster image build.
RUN if [[ "${INSTALL_SAGEATTENTION}" == "1" ]]; then \
      export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;12.1"; \
      export MAX_JOBS="2"; \
      export EXT_PARALLEL="2"; \
      python -m pip install "sageattention==${SAGEATTENTION_VERSION}" --no-build-isolation; \
    else \
      echo "Skipping SageAttention build"; \
    fi

COPY scripts /opt/runpod-comfy/scripts
COPY config /opt/runpod-comfy/default-config
COPY runpod-template.env.example /opt/runpod-comfy/runpod-template.env.example

RUN chmod +x /opt/runpod-comfy/scripts/*.sh

EXPOSE 8188 8888 22
WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null || exit 1

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "--", "/opt/runpod-comfy/scripts/entrypoint.sh"]

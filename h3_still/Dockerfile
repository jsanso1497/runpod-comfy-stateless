# syntax=docker/dockerfile:1
# Isolated build context. No other workspace code, model weights or secrets.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/h3-still/.hf \
    H3_STILL_DATA_ROOT=/workspace/h3-still \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl wget procps iproute2 openssh-client unzip git ca-certificates tini nginx apache2-utils ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/h3-still
COPY . .
RUN python validate_package.py \
    && python -m unittest discover -s checks -p '*_checks.py' -v \
    && python build_sources.py \
    && python validate_package.py
# Official PyTorch CUDA 13.0 wheels, per INT8/ConvRot H3 model-card guidance.
RUN python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt huggingface_hub jupyterlab==4.5.11 jupyter-server==2.21.1 websocket-client==1.9.0 \
    && python -m pip check \
    && python -m compileall -q /opt/h3-still \
    && python -m pip freeze > /opt/h3-still/build-pip-freeze.txt
# Python dependencies not pinned upstream are resolved here and recorded above.
# CI starts actual ComfyUI on CPU and checks authentication before publishing.
WORKDIR /workspace
EXPOSE 8188 8888
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8189/system_stats', timeout=8)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/h3-still/runtime/recovery_launcher.py"]

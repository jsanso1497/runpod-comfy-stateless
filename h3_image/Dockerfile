# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/h3-image/.hf \
    H3_IMAGE_ROOT=/workspace/h3-image NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl git ca-certificates tini openssl ffmpeg libgl1 libglib2.0-0 build-essential procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/h3-image
COPY . .
RUN test -s models.lock.json \
    && python3 -m unittest discover -s tests -v \
    && git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && COMMIT=$(python3 -c "import json; print(json.load(open('sources.json'))['comfy_commit'])") \
    && git -C /opt/ComfyUI fetch --depth 1 origin "$COMMIT" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "$COMMIT"
RUN python3 -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130 \
    && python3 -m pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt -r /opt/ComfyUI/manager_requirements.txt \
         huggingface_hub 'jupyterlab==4.5.11' 'jupyter-server==2.21.1' 'websocket-client==1.9.0' \
    && python3 -m pip check \
    && python3 -m unittest discover -s tests -v \
    && cp -a nodes/H3ImageWorkspace /opt/ComfyUI/custom_nodes/ \
    && python3 -m compileall -q /opt/h3-image \
    && python3 -m pip freeze > /opt/h3-image/build-pip-freeze.txt
WORKDIR /workspace
EXPOSE 8188 8888
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8189/system_stats', timeout=8)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python3", "/opt/h3-image/runtime/service_launcher.py"]

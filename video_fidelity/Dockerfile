# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FAITHFUL_APP_ROOT=/opt/video-fidelity FAITHFUL_SEEDVR_ROOT=/opt/seedvr2 \
    FAITHFUL_DATA_ROOT=/workspace/faithful-video \
    HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=/opt/video-fidelity
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates tini libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/video-fidelity
COPY . .
RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth 1 origin 52f98af2e2e42c421070a3e147c161c47cdeaf22 \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && git init /opt/seedvr2 \
    && git -C /opt/seedvr2 remote add origin https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
    && git -C /opt/seedvr2 fetch --depth 1 origin 4490bd1f482e026674543386bb2a4d176da245b9 \
    && git -C /opt/seedvr2 checkout --detach FETCH_HEAD \
    && python -m pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt \
    && python -m pip install -c constraints.txt -r /opt/seedvr2/requirements.txt \
    && python -m pip install -c constraints.txt 'aiohttp>=3.12,<4' 'filelock>=3.18,<4' 'huggingface-hub>=0.34,<2' av==17.0.0 \
    && ln -s /opt/video-fidelity/nodes /opt/ComfyUI/custom_nodes/faithful_video \
    && python -m compileall -q /opt/video-fidelity \
    && python -m pip freeze > /opt/video-fidelity/environment.freeze.txt \
    && python validate_package.py
# Only the local node is registered with ComfyUI. SeedVR2 runs in its own worker.
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8188/_health >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/video-fidelity/boot.py"]

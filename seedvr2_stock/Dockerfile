# One upstream ComfyUI install, one official SeedVR2 extension, existing Nginx auth.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG COMFY_COMMIT=2c75dc1b379a59e0880619d029607af5169a62ef
ARG SEEDVR_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH HF_HUB_DISABLE_TELEMETRY=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git ffmpeg curl ca-certificates tini nginx apache2-utils \
    libgl1 libglib2.0-0 libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
WORKDIR /opt/seedvr2-stock
COPY constraints.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -c constraints.txt torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
       --index-url https://download.pytorch.org/whl/cu128
RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth 1 origin "$COMFY_COMMIT" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "$COMFY_COMMIT" \
    && git init /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler \
    && git -C /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler remote add origin https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
    && git -C /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler fetch --depth 1 origin "$SEEDVR_COMMIT" \
    && git -C /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler rev-parse HEAD)" = "$SEEDVR_COMMIT" \
    && python -m pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt \
    && python -m pip install -c constraints.txt -r /opt/ComfyUI/custom_nodes/seedvr2_videoupscaler/requirements.txt \
    && python -m pip check
COPY . ./
RUN bash -n start.sh && bash -n checks/ci_start.sh \
    && python -m compileall -q /opt/seedvr2-stock \
    && python validate.py --upstream /opt/ComfyUI \
    && python -m pip freeze > environment.freeze.txt
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=5 \
    CMD curl --fail --silent --max-time 8 http://127.0.0.1:8189/system_stats >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/opt/seedvr2-stock/start.sh"]

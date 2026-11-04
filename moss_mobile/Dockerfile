# syntax=docker/dockerfile:1
# Separate image/context: does not install Qwen Image, Ollama, SAM or restoration stacks.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG COMFY_REF=v0.43.0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 MOSS_DATA_ROOT=/workspace/moss-mobile \
    HF_HOME=/workspace/moss-mobile/.hf
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tini nginx apache2-utils libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${COMFY_REF}" https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI rev-parse HEAD > /opt/comfy-source-commit.txt \
    && rm -rf /opt/ComfyUI/.git
WORKDIR /opt/moss-mobile
COPY . .
RUN python -m pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt huggingface_hub torchaudio==2.9.1 \
    && mkdir -p /opt/ComfyUI/custom_nodes/moss_mobile \
    && cp nodes/__init__.py /opt/ComfyUI/custom_nodes/moss_mobile/__init__.py \
    && python -m unittest discover -s tests -v \
    && python validate_package.py \
    && python -m pip freeze > /opt/moss-mobile/build-pip-freeze.txt
# Dependencies without exact upstream pins are resolved at build time and recorded above.
# GPU inference is a separate deployment validation step, not implied by this build.
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8189/system_stats', timeout=4)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/moss-mobile/runtime/launch.py"]

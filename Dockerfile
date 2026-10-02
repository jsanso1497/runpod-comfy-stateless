# syntax=docker/dockerfile:1.7

# Reuse the working Torch/CUDA/SeedVR2 foundation, not the rolling :latest tag.
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:56b51cf52ade07f8b6fc27f1c737ba20706deb589a731984af8f949e12019065
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV COMFY_HOME=/workspace/ComfyUI \
    CONFIG_HOME=/workspace/config \
    COMFY_PORT=8188 \
    COMFY_REF=master \
    ATTENTION_BACKEND=pytorch \
    VRAM_MODE=auto \
    ENABLE_DYNAMIC_VRAM=auto \
    ASSET_DOWNLOAD_WORKERS=2 \
    COMFY_MAX_UPLOAD_MB=1024 \
    TINI_SUBREAPER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTORCH_ALLOC_CONF=backend:cudaMallocAsync

# Keep the existing models and workflow manifests already in the repository.
COPY config /opt/runpod-comfy/default-config
COPY scripts/bootstrap.sh scripts/comfy_http_fix.py /opt/runpod-comfy/scripts/
RUN test -f /opt/video-restore/constraints.txt \
 && test -d /opt/seedvr2 \
 && bash -n /opt/runpod-comfy/scripts/bootstrap.sh \
 && python -m py_compile /opt/runpod-comfy/scripts/comfy_http_fix.py \
 && python /opt/runpod-comfy/scripts/comfy_http_fix.py self-test \
 && chmod +x /opt/runpod-comfy/scripts/bootstrap.sh

LABEL org.opencontainers.image.title="Stateless ComfyUI + SeedVR2" \
      org.opencontainers.image.description="ComfyUI, SeedVR2, Manager installation, safe RunPod HTTP compatibility and startup diagnostics" \
      org.opencontainers.image.version="comfy-seedvr2-1.1.0"

EXPOSE 8188 8888 22
WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${COMFY_PORT}/" >/dev/null \
   && curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null || exit 1

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "-s", "--", "/opt/runpod-comfy/scripts/entrypoint.sh"]
CMD []

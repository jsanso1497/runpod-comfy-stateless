# syntax=docker/dockerfile:1.7

# ComfyUI-first update for the SeedVR2 video-restoration test.
# Reuses the already-built/restoration-tested image, but restores normal ComfyUI
# as the service on port 8188. SeedVR2 source and Python dependencies are already
# present in the parent image at /opt/seedvr2.
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
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTORCH_ALLOC_CONF=backend:cudaMallocAsync

# Replace the old restoration web-app configuration with the ComfyUI-first one.
COPY config /opt/runpod-comfy/default-config
COPY scripts/bootstrap.sh /opt/runpod-comfy/scripts/bootstrap.sh
RUN chmod +x /opt/runpod-comfy/scripts/bootstrap.sh

LABEL org.opencontainers.image.title="Stateless ComfyUI + SeedVR2" \
      org.opencontainers.image.description="Normal ComfyUI UI with pinned SeedVR2 node, automatic 7B/VAE download, and prepared 4K/1080p video workflows" \
      org.opencontainers.image.version="comfy-seedvr2-1.0.0"

EXPOSE 8188 8888 22
WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null || exit 1

# The parent image used a purpose-built restoration web UI. Restore the original
# stateless ComfyUI entrypoint instead.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "--", "/opt/runpod-comfy/scripts/entrypoint.sh"]
CMD []

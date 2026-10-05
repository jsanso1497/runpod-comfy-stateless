# syntax=docker/dockerfile:1.7
# Reuse the proven Torch 2.9 / CUDA 12.9 / SeedVR2 foundation. Not a rolling tag.
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:56b51cf52ade07f8b6fc27f1c737ba20706deb589a731984af8f949e12019065
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION=local-uncommitted

# Official Linux NVIDIA Ollama release, pinned by version AND published archive hash.
ARG OLLAMA_VERSION=0.35.1
ARG OLLAMA_ARCHIVE_SHA256=9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV COMFY_HOME=/workspace/ComfyUI \
    CONFIG_HOME=/workspace/config \
    COMFY_PORT=8188 \
    COMFY_REF=pinned \
    ENABLE_OLLAMA=1 \
    OLLAMA_NO_CLOUD=1 \
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

# Ollama is a separate executable/runtime. It does not replace Torch or its CUDA wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends zstd ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl --fail --location --retry 3 --connect-timeout 30 --max-time 1200 \
    "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    -o /tmp/ollama.tar.zst \
 && printf '%s  %s\n' "${OLLAMA_ARCHIVE_SHA256}" /tmp/ollama.tar.zst | sha256sum -c - \
 && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
 && rm /tmp/ollama.tar.zst \
 && test -x /usr/bin/ollama

# Existing SeedVR2 config/workflows stay in the repository and are included here too.
RUN mkdir -p /opt/runpod-comfy && printf '%s\n' "$SOURCE_REVISION" > /opt/runpod-comfy/source-revision.txt
COPY shared_loras /opt/shared-loras
RUN python -m unittest discover -s /opt/shared-loras/tests -p 'test_*.py'
COPY config /opt/runpod-comfy/default-config
COPY scripts/entrypoint.sh scripts/hardware_check.py scripts/bootstrap.sh scripts/prepare_image.sh scripts/catalog.py scripts/install_nodes.py scripts/check_workflows.py scripts/check_rebalance_runtime.py scripts/check_single_person_runtime.py scripts/comfy_http_fix.py /opt/runpod-comfy/scripts/
COPY scripts/local_nodes /opt/runpod-comfy/scripts/local_nodes
COPY scripts/ollama_service.py /opt/runpod-comfy/scripts/
COPY Dockerfile /opt/runpod-comfy/Dockerfile
COPY tests /opt/runpod-comfy/tests
COPY config /opt/runpod-comfy/config
RUN bash -n /opt/runpod-comfy/scripts/bootstrap.sh \
 && bash -n /opt/runpod-comfy/scripts/prepare_image.sh \
 && python -m compileall -q /opt/runpod-comfy/scripts \
 && chmod +x /opt/runpod-comfy/scripts/bootstrap.sh /opt/runpod-comfy/scripts/prepare_image.sh /opt/runpod-comfy/scripts/entrypoint.sh

RUN python /opt/runpod-comfy/scripts/ollama_service.py check-install
RUN /opt/runpod-comfy/scripts/prepare_image.sh
LABEL org.opencontainers.image.title="ComfyUI Quality - Krea Rebalance + H3 RefMod + Ollama" \
      org.opencontainers.image.description="Krea-only Rebalance with grounded Identity Edit; H3 Full RefMod; local abliterated vision prompting" \
      org.opencontainers.image.version="comfy-quality-3.2.0"
EXPOSE 8188
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
 CMD curl -fsS "http://127.0.0.1:${COMFY_PORT}/" >/dev/null \
  && curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null || exit 1
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "-s", "--", "/opt/runpod-comfy/scripts/entrypoint.sh"]
CMD []

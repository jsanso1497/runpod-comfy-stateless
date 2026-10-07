# syntax=docker/dockerfile:1.7
# Same immutable Torch/CUDA foundation as the user-supplied repository.
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:56b51cf52ade07f8b6fc27f1c737ba20706deb589a731984af8f949e12019065
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION=local-uncommitted
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV COMFY_HOME=/workspace/ComfyUI COMFY_PORT=8188 COMFY_REF=pinned \
    FILEBROWSER_PORT=8888 COMFY_MAX_UPLOAD_MB=1024 VRAM_MODE=auto \
    FLUX_PHOTO_DOWNLOAD_MODELS=1 SYNC_SHARED_LORAS=1 \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TINI_SUBREAPER=1 \
    PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
COPY file_manager/ /opt/file-manager/
RUN bash /opt/file-manager/install.sh
COPY shared_loras/ /opt/shared-loras/
COPY flux_photo/ /opt/flux-photo/
COPY config/lora_links.txt /opt/flux-photo/config/lora_links.txt
COPY scripts/comfy_http_fix.py /opt/flux-photo/comfy_http_fix.py
# The FLUX workflow installs no additional KREA/H3/Ollama/SeedVR services and uses no remote inference API.
RUN set -Eeuo pipefail \
 && test -f /opt/video-restore/constraints.txt \
 && git init -q /opt/flux-comfy-bundle \
 && git -C /opt/flux-comfy-bundle remote add origin https://github.com/Comfy-Org/ComfyUI.git \
 && git -C /opt/flux-comfy-bundle fetch --depth 1 origin 65787d668397d230bf5839d69a0a7239e2dad378 \
 && git -C /opt/flux-comfy-bundle checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/flux-comfy-bundle rev-parse HEAD)" = 65787d668397d230bf5839d69a0a7239e2dad378 \
 && PIP_CONSTRAINT=/opt/video-restore/constraints.txt python -m pip install \
      -r /opt/flux-comfy-bundle/requirements.txt -r /opt/flux-comfy-bundle/manager_requirements.txt \
 && mkdir -p /opt/flux-comfy-bundle/custom_nodes/ComfyUI-FluxPhoto \
 && cp -a /opt/flux-photo/node/. /opt/flux-comfy-bundle/custom_nodes/ComfyUI-FluxPhoto/ \
 && python /opt/flux-photo/comfy_http_fix.py apply /opt/flux-comfy-bundle \
 && python -m pip check \
 && bash -n /opt/flux-photo/start.sh \
 && chmod +x /opt/flux-photo/start.sh \
 && python -m unittest discover -s /opt/flux-photo/tests -p 'test_*.py' \
 && python /opt/flux-photo/check.py --build-smoke --comfy-home /opt/flux-comfy-bundle \
 && printf '%s\n' "$SOURCE_REVISION" > /opt/flux-photo/source-revision.txt
LABEL org.opencontainers.image.title="FLUX Photo - Klein 9B Distilled Reference Inpaint" \
      org.opencontainers.image.version="flux-photo-1.0.0" \
      org.opencontainers.image.description="Full-precision local reference editing, manual masks, protected crop/stitch, optional LoRAs and image comparison."
EXPOSE 8188 8888
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
 CMD curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null \
  && curl -fsS "http://127.0.0.1:${FILEBROWSER_PORT}/login" >/dev/null || exit 1
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "-s", "--", "/opt/file-manager-venv/bin/python", "/opt/file-manager/service.py", "supervise", "--", "/opt/flux-photo/start.sh"]
CMD []

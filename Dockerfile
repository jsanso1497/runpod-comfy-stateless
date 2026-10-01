# syntax=docker/dockerfile:1.7
# Additive update: reuse James's successfully published GPU foundation.
# This digest is the EXISTING base, not the new restoration image.
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:e444d980f1c796405ba6478ce26682fd73c692fc56ea4ed51d727f31029730e8
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SEEDVR2_REF=4490bd1f482e026674543386bb2a4d176da245b9
ENV SEEDVR2_HOME=/opt/seedvr2 \
    RESTORE_HOME=/workspace/video-restore \
    RESTORE_PORT=8188 \
    RESTORE_MODEL=7b \
    RESTORE_ATTENTION=sdpa \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
# Pin the engine source. Never git-pull or pip-upgrade when a paid Pod starts.
RUN git init /opt/seedvr2 \
 && git -C /opt/seedvr2 remote add origin https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
 && git -C /opt/seedvr2 fetch --depth 1 origin "${SEEDVR2_REF}" \
 && git -C /opt/seedvr2 checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/seedvr2 rev-parse HEAD)" = "${SEEDVR2_REF}"
COPY config/video_restore/constraints.txt /opt/video-restore/constraints.txt
# The inherited SageAttention wheel was built against a different Torch ABI.
# Remove it for this first restoration image and use PyTorch SDPA, which SeedVR2
# supports natively. We can benchmark a source-built SageAttention later.
RUN python -m pip uninstall -y sageattention || true
RUN python -m pip install -c /opt/video-restore/constraints.txt \
        -r /opt/seedvr2/requirements.txt requests \
 && python -m pip check \
 && python /opt/seedvr2/inference_cli.py --help > /opt/video-restore/seedvr2-help.txt \
 && python -m pip freeze > /opt/video-restore/resolved-packages.txt
COPY scripts/video_restore /opt/video-restore/app
COPY config/video_restore/models.json /opt/video-restore/models.json
RUN python -m compileall -q /opt/video-restore/app \
 && python /opt/video-restore/app/validate_install.py \
 && chmod +x /opt/video-restore/app/start.sh
LABEL org.opencontainers.image.title="Stateless Video Restoration Test" \
      org.opencontainers.image.description="SeedVR2 7B/3B, comparison tests, authenticated upload UI; no persistent volume" \
      org.opencontainers.image.version="video-restore-1.0.0"
EXPOSE 8188
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${RESTORE_PORT}/healthz" >/dev/null || exit 1
# Override the original ComfyUI bootstrap. Do not start two applications on 8188.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "--", "/opt/video-restore/app/start.sh"]
CMD []

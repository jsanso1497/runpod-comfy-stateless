# syntax=docker/dockerfile:1
# Isolated new tag. Existing KREA, H3, FLUX and Qwen Identity Dockerfiles are untouched.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG COMFY_COMMIT=52f98af2e2e42c421070a3e147c161c47cdeaf22
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 COMFY_HOME=/workspace/ComfyUI COMFY_PORT=8188 \
    QWEN_PHOTO_MODEL_PRECISION=bf16 QWEN_PHOTO_DOWNLOAD_SAM=1 VRAM_MODE=auto
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -Eeuo pipefail \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git curl tini rsync libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/ComfyUI \
 && git init -q /opt/ComfyUI \
 && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
 && git -C /opt/ComfyUI fetch --depth 1 origin "$COMFY_COMMIT" \
 && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "$COMFY_COMMIT" \
 && rm -rf /opt/ComfyUI/.git
COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/
# Keep the PyTorch/CUDA packages shipped with the tested PyTorch base image.
RUN python /opt/qwen21_photo_edit/pin_torch.py /tmp/pytorch-constraints.txt
RUN set -Eeuo pipefail \
 && python -m pip install --constraint /tmp/pytorch-constraints.txt -r /opt/ComfyUI/requirements.txt \
 && python -m pip install --constraint /tmp/pytorch-constraints.txt scipy huggingface-hub pillow \
 && python -m pip check \
 && python /opt/qwen21_photo_edit/preflight.py --comfy-home /opt/ComfyUI --sam \
 && python -m unittest discover -s /opt/qwen21_photo_edit/tests -p 'test_*.py' -q \
 && python /opt/qwen21_photo_edit/ci_validate.py \
 && python -m compileall -q /opt/qwen21_photo_edit/node \
 && bash -n /opt/qwen21_photo_edit/start.sh /opt/qwen21_photo_edit/install.sh \
 && chmod +x /opt/qwen21_photo_edit/start.sh /opt/qwen21_photo_edit/install.sh
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=5 \
 CMD curl --fail --silent "http://127.0.0.1:${COMFY_PORT}/system_stats" > /dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen21_photo_edit/start.sh"]
CMD []

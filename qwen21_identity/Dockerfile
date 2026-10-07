# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG PROFILE=identity
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    DATA_ROOT=/workspace/qwen21_identity COMFY_PORT=8188 ASSET_DOWNLOAD_WORKERS=2
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git curl tini libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY . /opt/qwen21_identity
RUN case "$PROFILE" in identity|upscale) ;; *) echo 'PROFILE must be identity or upscale'; exit 1;; esac && \
    printf '%s\n' "$PROFILE" > /opt/qwen21_identity/config/built_profile.txt && \
    python /opt/qwen21_identity/scripts/checkout_sources.py --profile "$PROFILE" && \
    if [[ "$PROFILE" == upscale ]]; then \
      python /opt/qwen21_identity/scripts/patch_seedvr2_cpu_schema.py; \
    fi
# Constrain the CUDA stack already supplied by the base image. Do not silently replace it.
RUN python -c "import importlib.metadata as m; from pathlib import Path; Path('/tmp/torch-constraints.txt').write_text(''.join(f'{n}=={m.version(n)}\n' for n in ('torch','torchvision','torchaudio') if m.packages_distributions().get(n)))" && \
    python -m pip install --constraint /tmp/torch-constraints.txt -r /opt/ComfyUI/requirements.txt && \
    if [[ -f /opt/ComfyUI/custom_nodes/ComfyUI-AusBoss/requirements.txt ]]; then \
      python -m pip install --constraint /tmp/torch-constraints.txt -r /opt/ComfyUI/custom_nodes/ComfyUI-AusBoss/requirements.txt; \
    fi && \
    if [[ "$PROFILE" == upscale ]]; then \
      python -m pip install --constraint /tmp/torch-constraints.txt -r /opt/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt; \
    fi && \
    cp -a /opt/qwen21_identity/custom_nodes/qwen21_identity_tools /opt/ComfyUI/custom_nodes/ && \
    python -m compileall -q /opt/qwen21_identity/scripts /opt/qwen21_identity/custom_nodes && \
    chmod +x /opt/qwen21_identity/scripts/*.sh && \
    python /opt/qwen21_identity/scripts/validate_workflows.py --profile "$PROFILE" && \
    python -m pip freeze > /opt/qwen21_identity/config/built_python_packages.txt
WORKDIR /opt/ComfyUI
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
  CMD curl --fail --silent "http://127.0.0.1:${COMFY_PORT}/system_stats" > /dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen21_identity/scripts/start.sh"]
CMD []

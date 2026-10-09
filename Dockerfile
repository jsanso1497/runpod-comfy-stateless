# syntax=docker/dockerfile:1
# Fresh upstream base. Never inherit historical images containing private libraries.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG WB_WORKSPACE=qwen
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WB_ROOT=/opt/workbench PYTHONPATH=/opt/workbench/src \
    WB_WORKSPACE=${WB_WORKSPACE} WB_DATA_ROOT=/workspace HF_HUB_DISABLE_TELEMETRY=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tini ffmpeg libgl1 libglib2.0-0 zstd build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/workbench
COPY requirements.txt ./
COPY catalog/ ./catalog/
COPY config/ollama.json ./config/ollama.json
COPY config/h3/settings.json ./config/h3/settings.json
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY workflows/ ./workflows/
COPY automation/ ./automation/
COPY site/ ./site/
RUN python deploy/build.py "$WB_WORKSPACE" \
    && python -m compileall -q src \
    && python -c 'from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc; import workbench.assets, workbench.runtime; from workbench.http_gateway import ACCESS_VERSION, AUTH_MODE_KEY; assert ACCESS_VERSION == "1.2.0"; app = workbench.runtime.make_app(auth_mode="none"); assert app[AUTH_MODE_KEY] == "none"'
# Optional local helpers are executable here, but no model is downloaded at build time.
RUN curl --fail --location --retry 3 \
    https://github.com/ollama/ollama/releases/download/v0.35.1/ollama-linux-amd64.tar.zst \
    -o /tmp/ollama.tar.zst \
    && echo '9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10  /tmp/ollama.tar.zst' | sha256sum -c - \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
    && rm /tmp/ollama.tar.zst
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=2m --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8188/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "-m", "workbench.runtime"]

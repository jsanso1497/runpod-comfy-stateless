# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    H3_SWAP_ROOT=/opt/h3-swap PYTHONPATH=/opt/h3-swap \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tini ffmpeg zstd libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/h3-swap
COPY pins.json assets.spec.json requirements.txt ./
COPY swapkit/ ./swapkit/
COPY custom_nodes/ ./custom_nodes/
COPY licenses/ ./licenses/
COPY LICENSE ./licenses/INTEGRATION_LICENSE
COPY workflows/ ./workflows/
RUN python -m swapkit.build \
    && curl --fail --location --retry 3 \
       https://github.com/ollama/ollama/releases/download/v0.35.1/ollama-linux-amd64.tar.zst \
       -o /tmp/ollama.tar.zst \
    && echo '9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10  /tmp/ollama.tar.zst' | sha256sum -c - \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
    && rm /tmp/ollama.tar.zst \
    && python -m compileall -q swapkit custom_nodes \
    && python -m swapkit.check --static
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8188/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "-m", "swapkit.runtime"]

CMD []

# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    H3_SWAP_ROOT=/opt/h3-swap PYTHONPATH=/opt/h3-swap \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tini ffmpeg zstd libgl1 libglib2.0-0 python3-venv \
    && rm -rf /var/lib/apt/lists/*
# The base image includes spin, a source-build helper unused by H3 inference.
# Its Click upper bound conflicts with the current Hugging Face runtime.
# Remove only that verified, unneeded tool before exposing base packages to the
# application venv. This does not install anything into protected system Python.
COPY swapkit/base_cleanup.py /tmp/h3-base-cleanup.py
RUN H3_SWAP_BUILD_CLEANUP=1 python /tmp/h3-base-cleanup.py \
    && rm /tmp/h3-base-cleanup.py
# The qualified Ubuntu-based image marks system Python as externally managed.
# Install application dependencies into a venv while reusing the base image's
# existing PyTorch/CUDA packages. Application installs stay inside the venv.
RUN python -m venv --system-site-packages /opt/h3-swap-venv
ENV VIRTUAL_ENV=/opt/h3-swap-venv
ENV PATH="/opt/h3-swap-venv/bin:${PATH}"

WORKDIR /opt/h3-swap
COPY pins.json assets.spec.json requirements.txt ./
COPY swapkit/ ./swapkit/
COPY custom_nodes/ ./custom_nodes/
COPY licenses/ ./licenses/
COPY LICENSE ./licenses/INTEGRATION_LICENSE
COPY workflows/ ./workflows/
# Separate build stages keep the actual failing command visible in CI logs.
ARG H3_BUILD_PROFILE=baseline
RUN python -m swapkit.precision --prepare-build "$H3_BUILD_PROFILE"
RUN python -m swapkit.build
RUN curl --fail --location --retry 3 \
       https://github.com/ollama/ollama/releases/download/v0.35.1/ollama-linux-amd64.tar.zst \
       -o /tmp/ollama.tar.zst \
    && echo '9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10  /tmp/ollama.tar.zst' | sha256sum -c - \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
    && rm /tmp/ollama.tar.zst
# Classic FileBrowser is private behind the authenticated 8188 gateway.
RUN curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
       https://github.com/filebrowser/filebrowser/releases/download/v2.63.23/linux-amd64-filebrowser.tar.gz \
       -o /tmp/filebrowser.tar.gz \
    && echo 'b14db2bb8033caa3f80205eb6578b2ed0744ebd9e716b790bc4a9703ce909e88  /tmp/filebrowser.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/filebrowser.tar.gz -C /usr/local/bin filebrowser \
    && chmod 755 /usr/local/bin/filebrowser \
    && rm /tmp/filebrowser.tar.gz \
    && python -m swapkit.filebrowser --smoke
# Tiny preview weights are only 22.7 MB and never replace the final H3 VAE.
RUN python -m swapkit.preview_assets --download
RUN python -m compileall -q swapkit custom_nodes \
    && python -m swapkit.check --static
WORKDIR /workspace
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8188/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "-m", "swapkit.runtime"]

CMD []

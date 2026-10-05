# syntax=docker/dockerfile:1.7
# Proven successful image from the user's build on 2026-10-05. Do NOT use latest here.
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:49d305115a3cfe5763d5c9694eb20429803cf3e1a209567506dba15de594f672
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY h3_portrait/ /opt/h3-portrait/
COPY shared_loras/ /opt/shared-loras/
COPY config/lora_links.txt /opt/shared-loras/lora_links.txt
RUN python -m unittest discover -s /opt/shared-loras/tests -p 'test_*.py'

# This changes only the new h3-portrait tag, not the existing :latest image.
# Inherited layers remain in registry storage, but unrelated node packs are NOT
# copied into the running ComfyUI tree. No GPU model weights are in this layer.
RUN test "$(cat /opt/runpod-comfy/comfy-build-ref.txt)" = "65787d668397d230bf5839d69a0a7239e2dad378" \
 && test -x /usr/bin/ollama \
 && find /opt/comfy-bundle/custom_nodes -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + \
 && cp -a /opt/h3-portrait/node /opt/comfy-bundle/custom_nodes/ComfyUI-H3Portrait \
 && chmod +x /opt/h3-portrait/start.sh \
 && bash -n /opt/h3-portrait/start.sh \
 && python -m compileall -q /opt/h3-portrait \
 && cd /opt/h3-portrait && python -m unittest discover -s tests -p 'test_*.py' \
 && python /opt/runpod-comfy/scripts/comfy_http_fix.py self-test \
 && python /opt/runpod-comfy/scripts/comfy_http_fix.py apply /opt/comfy-bundle \
 && python /opt/h3-portrait/smoke_check.py --build --wait 240
ENV H3_PORTRAIT_CONFIG=/workspace/h3-portrait/config \
    OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M \
    OLLAMA_NO_CLOUD=1 \
    ASSET_DOWNLOAD_WORKERS=2 \
    TINI_SUBREAPER=1 \
    PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="H3 Portrait Auto" \
      org.opencontainers.image.version="h3-portrait-1.1.0"
EXPOSE 8188
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
 CMD curl -fsS http://127.0.0.1:8188/system_stats >/dev/null || exit 1
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/bin/tini", "-s", "--", "/opt/h3-portrait/start.sh"]
CMD []

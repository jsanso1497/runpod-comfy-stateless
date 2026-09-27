#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /workspace /root/.ssh

# Optional SSH access. RunPod can inject PUBLIC_KEY into the template.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd
fi

# Optional JupyterLab. Prefer a token from RunPod Secrets.
if [[ -n "${JUPYTER_TOKEN:-${JUPYTER_PASSWORD:-}}" ]]; then
  JUPYTER_AUTH="${JUPYTER_TOKEN:-${JUPYTER_PASSWORD}}"
  jupyter lab \
    --allow-root \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --ServerApp.token="${JUPYTER_AUTH}" \
    --ServerApp.root_dir=/workspace \
    >/workspace/jupyter.log 2>&1 &
fi

exec /opt/runpod-comfy/scripts/bootstrap.sh

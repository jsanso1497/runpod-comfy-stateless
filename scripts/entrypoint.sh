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

# Password-protected file manager is started by the shared supervisor on 8888.
exec /opt/runpod-comfy/scripts/bootstrap.sh

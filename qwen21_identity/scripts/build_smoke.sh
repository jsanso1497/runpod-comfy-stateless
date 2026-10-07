#!/usr/bin/env bash
# CPU-only registration/schema check. Never downloads model weights or samples an image.
set -euo pipefail
KIT=/opt/qwen21_identity
PROFILE="$(cat "$KIT/config/built_profile.txt")"
cd /opt/ComfyUI
python main.py --cpu --listen 127.0.0.1 --port 8191 > /tmp/qwen21-build-smoke.log 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT
READY=0
for ((i=0;i<180;i++)); do
  if curl --fail --silent http://127.0.0.1:8191/system_stats > /dev/null; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then cat /tmp/qwen21-build-smoke.log; exit 1; fi
  sleep 2
done
if (( READY == 0 )); then cat /tmp/qwen21-build-smoke.log; exit 1; fi
if ! python "$KIT/scripts/validate_workflows.py" --profile "$PROFILE" --server http://127.0.0.1:8191; then
  cat /tmp/qwen21-build-smoke.log
  exit 1
fi

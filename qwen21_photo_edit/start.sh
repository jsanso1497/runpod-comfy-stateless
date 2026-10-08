#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "Qwen Photo startup failed at line $LINENO (exit $rc). Review the preceding error." >&2; exit "$rc"' ERR
export COMFY_HOME="${COMFY_HOME:-/workspace/ComfyUI}"
export COMFY_PORT="${COMFY_PORT:-8188}"
export PYTHONUNBUFFERED=1
case "${QWEN_PHOTO_MODEL_PRECISION:-bf16}" in
  bf16|int8|both|none) ;;
  *) echo 'QWEN_PHOTO_MODEL_PRECISION must be bf16, int8, both, or none.' >&2; exit 2 ;;
esac
case "${QWEN_PHOTO_DOWNLOAD_SAM:-1}" in
  0|1) ;;
  *) echo 'QWEN_PHOTO_DOWNLOAD_SAM must be 0 or 1.' >&2; exit 2 ;;
esac
case "${VRAM_MODE:-auto}" in
  auto|low|high|normal) ;;
  *) echo 'VRAM_MODE must be auto, normal, low, or high.' >&2; exit 2 ;;
esac
if ! [[ "$COMFY_PORT" =~ ^[0-9]+$ ]] || (( COMFY_PORT < 1024 || COMFY_PORT > 65535 )); then
  echo 'COMFY_PORT must be a TCP port from 1024 to 65535.' >&2; exit 2
fi
python - <<'PY'
import os
from pathlib import Path
root=Path(os.environ['COMFY_HOME']).resolve()
if not root.is_relative_to(Path('/workspace')) or root == Path('/workspace'):
    raise SystemExit('COMFY_HOME must be inside /workspace and cannot equal /workspace.')
PY
mkdir -p "$COMFY_HOME"/{models,input,output,temp,user/default/workflows,custom_nodes}
# Copy the pinned core but preserve user files, model weights and custom nodes on existing volumes.
rsync -a --exclude='/.git/' --exclude='/models/' --exclude='/input/' \
  --exclude='/output/' --exclude='/temp/' --exclude='/user/' --exclude='/custom_nodes/' \
  /opt/ComfyUI/ "$COMFY_HOME/"
INSTALL=(--comfy-home "$COMFY_HOME" --models "${QWEN_PHOTO_MODEL_PRECISION:-bf16}")
if [[ "${QWEN_PHOTO_DOWNLOAD_SAM:-1}" == 1 ]]; then INSTALL+=(--sam); fi
bash /opt/qwen21_photo_edit/install.sh "${INSTALL[@]}"
ARGS=(--listen 0.0.0.0 --port "$COMFY_PORT" --use-pytorch-cross-attention --max-upload-size 1024 --disable-auto-launch)
case "${VRAM_MODE:-auto}" in
  low) ARGS+=(--lowvram) ;;
  high) ARGS+=(--highvram) ;;
  auto|normal) ;;
esac
printf '\nQwen Image 2.1 protected photo edit | manual/SAM 3.1 | multi-reference | source-size stitch\n'
cd "$COMFY_HOME"
exec python main.py "${ARGS[@]}"

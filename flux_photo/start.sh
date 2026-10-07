#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; printf "\n[FLUX PHOTO STARTUP FAILED] line=%s exit=%s. See error above.\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
export COMFY_HOME="${COMFY_HOME:-/workspace/ComfyUI}"
export COMFY_PORT="${COMFY_PORT:-8188}"
if [[ "${COMFY_REF:-pinned}" != pinned && "${COMFY_REF:-pinned}" != 65787d668397d230bf5839d69a0a7239e2dad378 ]]; then
  echo 'Use COMFY_REF=pinned. Rebuild rather than updating core at startup.' >&2
  exit 2
fi
python - <<'PY'
import os
from pathlib import Path
p=Path(os.environ['COMFY_HOME']).resolve()
if not p.is_relative_to(Path('/workspace')) or p == Path('/workspace'):
    raise SystemExit('COMFY_HOME must be a subdirectory of /workspace.')
PY
mkdir -p "$COMFY_HOME"/{models,input,output,temp,user/default/workflows,custom_nodes}
rsync -a --exclude='__pycache__' --exclude='/models/' --exclude='/input/' --exclude='/output/' \
  --exclude='/temp/' --exclude='/user/' /opt/flux-comfy-bundle/ "$COMFY_HOME/"
python /opt/flux-photo/comfy_http_fix.py apply "$COMFY_HOME"
python /opt/flux-photo/runtime.py --comfy-home "$COMFY_HOME"
ARGS=(--listen 0.0.0.0 --port "$COMFY_PORT" --enable-manager --use-pytorch-cross-attention \
      --disable-dynamic-vram --max-upload-size "${COMFY_MAX_UPLOAD_MB:-1024}" --disable-auto-launch)
case "${VRAM_MODE:-auto}" in
  auto|normal) ;;
  low) ARGS+=(--lowvram) ;;
  high) ARGS+=(--highvram) ;;
  *) echo 'VRAM_MODE must be auto, normal, low or high.' >&2; exit 2 ;;
esac
if [[ -n "${COMFY_ARGS:-}" ]]; then
  echo 'COMFY_ARGS overrides are not enabled in this precision-protected template. Edit its start.sh and rebuild.' >&2
  exit 2
fi
printf '\nFLUX Photo 1.0 | 9B DISTILLED | original-canvas inpainting | local inference only\n'
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU found. Select an NVIDIA GPU for this template.')
p=torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| VRAM:', round(p.total_memory/1024**3,1), 'GiB')
PY
cd "$COMFY_HOME"
exec python main.py "${ARGS[@]}"

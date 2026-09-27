#!/usr/bin/env bash
set -Eeuo pipefail

COMFY_HOME="${COMFY_HOME:-/workspace/ComfyUI}"
CONFIG_HOME="${CONFIG_HOME:-/workspace/config}"
COMFY_REPO="${COMFY_REPO:-https://github.com/Comfy-Org/ComfyUI.git}"
COMFY_REF="${COMFY_REF:-master}"
CONFIG_REF="${CONFIG_REF:-main}"
COMFY_PORT="${COMFY_PORT:-8188}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

clone_repo() {
  local repo="$1"
  local dest="$2"
  local ref="${3:-}"

  rm -rf "$dest"

  if [[ -n "${GITHUB_TOKEN:-}" && "$repo" == https://github.com/* ]]; then
    git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" clone --filter=blob:none "$repo" "$dest"
  else
    git clone --filter=blob:none "$repo" "$dest"
  fi

  if [[ -n "$ref" ]]; then
    git -C "$dest" fetch --depth 1 origin "$ref" || true
    git -C "$dest" checkout --detach "FETCH_HEAD" 2>/dev/null || git -C "$dest" checkout "$ref"
  fi
}

log "Stateless ComfyUI bootstrap"
printf 'Container disk: '
df -h /workspace | tail -n 1

log "GPU and runtime check"
python /opt/runpod-comfy/scripts/hardware_check.py

log "Preparing configuration"
if [[ -n "${CONFIG_REPO:-}" ]]; then
  clone_repo "${CONFIG_REPO}" "$CONFIG_HOME" "$CONFIG_REF"
else
  rm -rf "$CONFIG_HOME"
  cp -a /opt/runpod-comfy/default-config "$CONFIG_HOME"
fi

log "Installing current ComfyUI"
clone_repo "$COMFY_REPO" "$COMFY_HOME" "$COMFY_REF"
python -m pip install -r "$COMFY_HOME/requirements.txt"

log "Installing custom nodes"
python /opt/runpod-comfy/scripts/install_nodes.py \
  --manifest "$CONFIG_HOME/custom_nodes.json" \
  --comfy-home "$COMFY_HOME"

log "Downloading model assets"
python /opt/runpod-comfy/scripts/download_assets.py \
  --manifest "$CONFIG_HOME/models.json" \
  --comfy-home "$COMFY_HOME"

log "Downloading LoRAs"
python /opt/runpod-comfy/scripts/download_assets.py \
  --manifest "$CONFIG_HOME/loras.json" \
  --comfy-home "$COMFY_HOME"

if [[ -d "$CONFIG_HOME/workflows" ]]; then
  log "Installing workflows"
  mkdir -p "$COMFY_HOME/user/default/workflows"
  rsync -a "$CONFIG_HOME/workflows/" "$COMFY_HOME/user/default/workflows/"
fi

log "Selecting launch settings"
LAUNCH_ARGS=(
  --listen 0.0.0.0
  --port "$COMFY_PORT"
  --enable-manager
)

ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
if [[ "$ATTENTION_BACKEND" == "auto" ]]; then
  if python /opt/runpod-comfy/scripts/hardware_check.py --sage-usable >/dev/null 2>&1; then
    ATTENTION_BACKEND="sage"
  else
    ATTENTION_BACKEND="pytorch"
  fi
fi

case "$ATTENTION_BACKEND" in
  sage)
    LAUNCH_ARGS+=(--use-sage-attention)
    ;;
  pytorch)
    LAUNCH_ARGS+=(--use-pytorch-cross-attention)
    ;;
  default|none)
    ;;
  *)
    echo "Unknown ATTENTION_BACKEND=${ATTENTION_BACKEND}. Use auto, sage, pytorch, or default." >&2
    exit 2
    ;;
esac

case "${VRAM_MODE:-auto}" in
  low) LAUNCH_ARGS+=(--lowvram) ;;
  high) LAUNCH_ARGS+=(--highvram) ;;
  normal|auto) ;;
  *) echo "Unknown VRAM_MODE=${VRAM_MODE}. Use auto, low, normal, or high." >&2; exit 2 ;;
esac

if [[ "${ENABLE_DYNAMIC_VRAM:-auto}" == "1" ]]; then
  LAUNCH_ARGS+=(--enable-dynamic-vram)
elif [[ "${ENABLE_DYNAMIC_VRAM:-auto}" == "0" ]]; then
  LAUNCH_ARGS+=(--disable-dynamic-vram)
fi

if [[ -n "${COMFY_ARGS:-}" ]]; then
  # Intentional shell splitting so advanced users can append normal CLI flags.
  read -r -a EXTRA_ARGS <<< "${COMFY_ARGS}"
  LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

log "Starting ComfyUI"
printf 'Attention backend: %s\n' "$ATTENTION_BACKEND"
printf 'Command: python main.py'
printf ' %q' "${LAUNCH_ARGS[@]}"
printf '\n'

cd "$COMFY_HOME"
exec python main.py "${LAUNCH_ARGS[@]}"

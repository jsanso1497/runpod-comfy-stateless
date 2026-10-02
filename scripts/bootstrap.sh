#!/usr/bin/env bash
set -Eeuo pipefail

COMFY_HOME="${COMFY_HOME:-/workspace/ComfyUI}"
CONFIG_HOME="${CONFIG_HOME:-/workspace/config}"
COMFY_REPO="${COMFY_REPO:-https://github.com/Comfy-Org/ComfyUI.git}"
COMFY_REF="${COMFY_REF:-master}"
CONFIG_REF="${CONFIG_REF:-main}"
COMFY_PORT="${COMFY_PORT:-8188}"
SEEDVR2_SOURCE="/opt/seedvr2"
SEEDVR2_NODE_NAME="ComfyUI-SeedVR2_VideoUpscaler"
HTTP_FIX="/opt/runpod-comfy/scripts/comfy_http_fix.py"
CONSTRAINTS="/opt/video-restore/constraints.txt"
export COMFY_HOME CONFIG_HOME COMFY_PORT
export TINI_SUBREAPER=1

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
trap 'rc=$?; printf "\n[BOOTSTRAP FAILED] line=%s exit=%s. Read the error immediately above.\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR

clone_repo() {
  local repo="$1" dest="$2" ref="${3:-main}"
  # Keep models/input/output on a process restart. Never rm -rf COMFY_HOME.
  if [[ "$dest" != /* || "$dest" == / ]]; then
    echo "Repository destination must be a non-root absolute path." >&2
    return 2
  fi
  mkdir -p "$dest"
  if [[ ! -d "$dest/.git" ]]; then
    git init -q "$dest"
  fi
  if git -C "$dest" remote get-url origin >/dev/null 2>&1; then
    git -C "$dest" remote set-url origin "$repo"
  else
    git -C "$dest" remote add origin "$repo"
  fi
  if [[ -n "${GITHUB_TOKEN:-}" && "$repo" == https://github.com/* ]]; then
    git -c http.extraHeader="Authorization: Bearer ${GITHUB_TOKEN}" -C "$dest" fetch --depth 1 origin "$ref"
  else
    git -C "$dest" fetch --depth 1 origin "$ref"
  fi
  # Fails on a bad ref instead of silently starting a different revision.
  git -C "$dest" checkout --force --detach FETCH_HEAD
  log "Checked out $(git -C "$dest" rev-parse HEAD)"
}

log "Stateless ComfyUI + SeedVR2 bootstrap | HTTP/Manager fix 1.1.0"
printf 'Container disk: '
df -h /workspace | tail -n 1

log "GPU and runtime check"
python /opt/runpod-comfy/scripts/hardware_check.py

log "Preparing configuration"
mkdir -p "$CONFIG_HOME"
if [[ -n "${CONFIG_REPO:-}" ]]; then
  clone_repo "$CONFIG_REPO" "$CONFIG_HOME" "$CONFIG_REF"
else
  cp -a /opt/runpod-comfy/default-config/. "$CONFIG_HOME/"
fi

log "Installing ComfyUI at ref: $COMFY_REF"
clone_repo "$COMFY_REPO" "$COMFY_HOME" "$COMFY_REF"
if [[ ! -f "$CONSTRAINTS" || ! -f "$COMFY_HOME/manager_requirements.txt" ]]; then
  echo "Missing constraints or ComfyUI manager_requirements.txt; refusing a partial install." >&2
  exit 2
fi

# Use the Manager version required by this ComfyUI checkout. Protect the working
# Torch/CUDA/SeedVR2 package cohort against accidental upgrades during startup.
export PIP_CONSTRAINT="$CONSTRAINTS"
log "Installing ComfyUI AND Manager requirements (with version constraints)"
python -m pip install --disable-pip-version-check \
  -r "$COMFY_HOME/requirements.txt" \
  -r "$COMFY_HOME/manager_requirements.txt"
python -m pip check
(
  cd "$COMFY_HOME"
  python -c 'import comfyui_manager; from importlib.metadata import version; print("Manager import: PASS | version=" + version("comfyui-manager"))'
)

log "Installing pinned SeedVR2 custom node"
if [[ ! -d "$SEEDVR2_SOURCE" ]]; then
  echo "Expected pinned SeedVR2 source at $SEEDVR2_SOURCE but it was not found." >&2
  exit 2
fi
mkdir -p "$COMFY_HOME/custom_nodes/$SEEDVR2_NODE_NAME"
cp -a "$SEEDVR2_SOURCE/." "$COMFY_HOME/custom_nodes/$SEEDVR2_NODE_NAME/"

log "Applying safe RunPod HTTP compatibility fix"
python "$HTTP_FIX" apply "$COMFY_HOME"

log "Downloading SeedVR2 model assets"
python /opt/runpod-comfy/scripts/download_assets.py \
  --manifest "$CONFIG_HOME/models.json" \
  --comfy-home "$COMFY_HOME"

log "Installing prepared workflows"
mkdir -p "$COMFY_HOME/user/default/workflows"
rsync -a "$CONFIG_HOME/workflows/" "$COMFY_HOME/user/default/workflows/"

LAUNCH_ARGS=(
  --listen 0.0.0.0
  --port "$COMFY_PORT"
  --enable-manager
  --use-pytorch-cross-attention
  --max-upload-size "${COMFY_MAX_UPLOAD_MB:-1024}"
)

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
  read -r -a EXTRA_ARGS <<< "$COMFY_ARGS"
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == --enable-cors-header* ]]; then
      echo "Remove --enable-cors-header from COMFY_ARGS. This patch fixes navigation without disabling the origin guard." >&2
      exit 2
    fi
  done
  LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

log "Starting ComfyUI; HTTP checks will run after the server is ready"
printf 'Command: python main.py'
printf ' %q' "${LAUNCH_ARGS[@]}"
printf '\n'
# Local-only probes of UI, proxy-style headers, API protection and node presence.
# This child does not change the public proxy, credentials or GPU job queue.
python "$HTTP_FIX" check --port "$COMFY_PORT" --timeout 240 &

cd "$COMFY_HOME"
exec python main.py "${LAUNCH_ARGS[@]}"

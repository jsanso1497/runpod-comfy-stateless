#!/usr/bin/env bash
set -Eeuo pipefail
COMFY_HOME="${COMFY_HOME:-/workspace/ComfyUI}"
CONFIG_HOME="${CONFIG_HOME:-/workspace/config}"
COMFY_PORT="${COMFY_PORT:-8188}"
SCRIPTS=/opt/runpod-comfy/scripts
BUNDLE=/opt/comfy-bundle
export COMFY_HOME CONFIG_HOME COMFY_PORT TINI_SUBREAPER=1
log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
trap 'rc=$?; printf "\n[BOOTSTRAP FAILED] line=%s exit=%s. Read the error above.\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR

log "ComfyUI Quality 3.2 | One Person Multi-Reference + Krea-only Rebalance"
BUILD_REF="$(cat /opt/runpod-comfy/comfy-build-ref.txt)"
REQUESTED_REF="${COMFY_REF:-pinned}"
if [[ "$REQUESTED_REF" != pinned && "$REQUESTED_REF" != "$BUILD_REF" ]]; then
  echo 'This image uses a validated code revision. Set COMFY_REF=pinned (remove the old master value). Model/LoRA catalogs remain editable.' >&2
  exit 2
fi
python "$SCRIPTS/hardware_check.py"
python - <<'PY_PATHS'
import os, re
from pathlib import Path
from urllib.parse import urlsplit
workspace = Path('/workspace').resolve()
paths = [Path(os.environ[x]).resolve() for x in ('COMFY_HOME', 'CONFIG_HOME')]
if any(not p.is_relative_to(workspace) or p == workspace for p in paths):
    raise SystemExit('COMFY_HOME and CONFIG_HOME must be separate subdirectories of /workspace.')
if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
    raise SystemExit('The ComfyUI and config directories must not overlap.')
url = os.environ.get('CONFIG_REPO', '')
if url:
    u = urlsplit(url)
    if u.scheme != 'https' or u.hostname != 'github.com' or u.username or u.password or u.query or u.fragment or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', u.path):
        raise SystemExit('CONFIG_REPO must be a plain GitHub HTTPS repository URL, without credentials.')
PY_PATHS

mkdir -p "$CONFIG_HOME" "$COMFY_HOME"
if [[ -n "${CONFIG_REPO:-}" ]]; then
  # Public or private HTTPS configuration repo. Credentials never enter its URL or Git config.
  [[ "$CONFIG_REPO" == https://github.com/* ]] || { echo 'CONFIG_REPO must be a GitHub HTTPS URL.' >&2; exit 2; }
  if [[ ! -d "$CONFIG_HOME/.git" ]]; then
    git -C "$CONFIG_HOME" init -q
    git -C "$CONFIG_HOME" remote add origin "$CONFIG_REPO"
  else
    git -C "$CONFIG_HOME" remote set-url origin "$CONFIG_REPO"
  fi
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    ASKPASS="$(mktemp)"
    cat > "$ASKPASS" <<'ASK'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
ASK
    chmod 700 "$ASKPASS"
    if ! GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git -C "$CONFIG_HOME" fetch --depth 1 origin "${CONFIG_REF:-main}"; then
      rm -f "$ASKPASS"; exit 2
    fi
    rm -f "$ASKPASS"
  else
    GIT_TERMINAL_PROMPT=0 git -C "$CONFIG_HOME" fetch --depth 1 origin "${CONFIG_REF:-main}"
  fi
  git -C "$CONFIG_HOME" checkout --force --detach FETCH_HEAD
else
  cp -a /opt/runpod-comfy/default-config/. "$CONFIG_HOME/"
fi

RUNTIME_REF="$(python "$SCRIPTS/catalog.py" comfy-ref --config "$CONFIG_HOME")"
[[ "$RUNTIME_REF" == "$BUILD_REF" ]] || { echo 'Config requests a different ComfyUI revision. Rebuild the image before changing runtime.json comfy_ref.' >&2; exit 2; }
python - "$CONFIG_HOME/custom_nodes.json" <<'PY'
import json, sys
from pathlib import Path
def pins(path):
    return sorted((x['name'], x['repo'], x['ref']) for x in json.loads(Path(path).read_text()) if x.get('enabled', True))
if pins(sys.argv[1]) != pins('/opt/runpod-comfy/bundled-custom-nodes.json'):
    raise SystemExit('Custom-node pins differ from this image. Rebuild once to add/update nodes; model/LoRA-only changes do not need that.')
PY

log "Installing bundled ComfyUI and pinned nodes at $BUILD_REF"
# No --delete: preserve models, uploads, saved workflows and results on a process restart.
mkdir -p "$COMFY_HOME"/{models,input,output,temp,user/default/workflows,custom_nodes}
rsync -a --exclude='__pycache__' --exclude='build-smoke.log' \
  --exclude='/models/' --exclude='/input/' --exclude='/output/' --exclude='/user/' --exclude='/temp/' \
  "$BUNDLE/" "$COMFY_HOME/"
export PIP_CONSTRAINT=/opt/video-restore/constraints.txt
python -m pip check
python "$SCRIPTS/comfy_http_fix.py" apply "$COMFY_HOME"
PROFILES="$(python "$SCRIPTS/catalog.py" profiles --config "$CONFIG_HOME")"
log "Preparing model profiles: $PROFILES"
python "$SCRIPTS/catalog.py" download --config "$CONFIG_HOME" --comfy-home "$COMFY_HOME"
# Shared user LoRAs are independent of MODEL_PROFILES. Do not auto-apply them.
python /opt/shared-loras/sync.py --links "$CONFIG_HOME/lora_links.txt" --comfy-home "$COMFY_HOME"
python "$SCRIPTS/catalog.py" workflows --config "$CONFIG_HOME" --comfy-home "$COMFY_HOME"
python /opt/krea-identity/install.py runtime --comfy-home "$COMFY_HOME"
python /opt/h3-media/install.py runtime --comfy-home "$COMFY_HOME"

# Independent optional service: a failed pull does not prevent existing Krea/SeedVR2 use.
# The helper binds ONLY loopback. Do not expose port 11434 in the RunPod template.
log "Starting optional local Ollama helper (status: /workspace/ollama/status.json)"
python "$SCRIPTS/ollama_service.py" serve &

ARGS=(--listen 0.0.0.0 --port "$COMFY_PORT" --enable-manager --use-pytorch-cross-attention --max-upload-size "${COMFY_MAX_UPLOAD_MB:-1024}" --disable-auto-launch)
case "${VRAM_MODE:-auto}" in
  auto|normal) ;;
  low) ARGS+=(--lowvram) ;;
  high) ARGS+=(--highvram) ;;
  *) echo 'VRAM_MODE must be auto, normal, low or high.' >&2; exit 2 ;;
esac
if [[ "${ENABLE_DYNAMIC_VRAM:-auto}" == 1 ]]; then ARGS+=(--enable-dynamic-vram); fi
if [[ "${ENABLE_DYNAMIC_VRAM:-auto}" == 0 ]]; then ARGS+=(--disable-dynamic-vram); fi
if [[ -n "${COMFY_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "$COMFY_ARGS"
  for arg in "${EXTRA[@]}"; do
    case "$arg" in --enable-cors-header*|--use-sage-attention|--use-flash-attention)
      echo 'Remove global CORS/attention overrides. This profile uses the safe HTTP patch and PyTorch attention.' >&2; exit 2 ;;
    esac
  done
  ARGS+=("${EXTRA[@]}")
fi
log "Starting NORMAL ComfyUI on port $COMFY_PORT. All data remains on disposable storage."
printf 'Command: python main.py'; printf ' %q' "${ARGS[@]}"; printf '\n'
python "$SCRIPTS/comfy_http_fix.py" check --port "$COMFY_PORT" --timeout 240 &
python "$SCRIPTS/check_workflows.py" server --config "$CONFIG_HOME" --port "$COMFY_PORT" --timeout 240 &
cd "$COMFY_HOME"
exec python main.py "${ARGS[@]}"

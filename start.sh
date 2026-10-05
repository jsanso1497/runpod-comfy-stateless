#!/usr/bin/env bash
set -Eeuo pipefail
export COMFY_HOME=/workspace/ComfyUI
export H3_PORTRAIT_CONFIG=/workspace/h3-portrait/config
export OLLAMA_NO_CLOUD=1 TINI_SUBREAPER=1
# This recipe deliberately does not use external CONFIG_REPO or legacy bootstrap.
unset CONFIG_REPO CONFIG_REF COMFY_ARGS MODEL_PROFILES
printf '\nH3 PORTRAIT 1.2 | Two-pass thinking | Shared link-only LoRAs | Native H3 | No OmniNode\n'
mkdir -p "$COMFY_HOME" "$H3_PORTRAIT_CONFIG" /workspace/h3-portrait
cp -a /opt/h3-portrait/{settings.json,models.json,runtime.json} "$H3_PORTRAIT_CONFIG/"
cp /opt/shared-loras/lora_links.txt "$H3_PORTRAIT_CONFIG/lora_links.txt"
# The alternate image has only our node pack enabled in this bundle.
rsync -a --exclude='__pycache__' --exclude='build-smoke.log' --exclude='/models/' --exclude='/input/' --exclude='/output/' --exclude='/user/' --exclude='/temp/' /opt/comfy-bundle/ "$COMFY_HOME/"
mkdir -p "$COMFY_HOME"/{models,input,output,temp,user/default/workflows}
if [[ ! -f "$COMFY_HOME/user/default/workflows/H3_Portrait_Auto.json" ]]; then
  cp /opt/h3-portrait/workflows/H3_Portrait_Auto.json "$COMFY_HOME/user/default/workflows/"
fi
python /opt/runpod-comfy/scripts/comfy_http_fix.py apply "$COMFY_HOME"
# Catch configuration/access errors before starting the expensive generation model.
python /opt/h3-portrait/prepare_assets.py
OLLAMA_PID=''
COMFY_PID=''
cleanup() {
  trap - TERM INT EXIT
  [[ -z "$COMFY_PID" ]] || kill "$COMFY_PID" 2>/dev/null || true
  [[ -z "$OLLAMA_PID" ]] || kill "$OLLAMA_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT
python -u /opt/h3-portrait/ollama_service.py & OLLAMA_PID=$!
cd "$COMFY_HOME"
python main.py --listen 0.0.0.0 --port 8188 --enable-manager --use-pytorch-cross-attention --max-upload-size 1024 --disable-auto-launch & COMFY_PID=$!
python /opt/h3-portrait/smoke_check.py --port 8188 --wait 240
printf '\nH3 PORTRAIT WORKFLOW READY: open HTTP 8188. Wait for H3 PORTRAIT OLLAMA READY before running.\n'
# If either critical service exits, stop its sibling rather than leave a paid idle Pod.
wait -n "$COMFY_PID" "$OLLAMA_PID"

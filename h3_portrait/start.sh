#!/usr/bin/env bash
set -Eeuo pipefail
export COMFY_HOME=/workspace/ComfyUI
export H3_PORTRAIT_CONFIG=/workspace/h3-portrait/config
export OLLAMA_NO_CLOUD=1 TINI_SUBREAPER=1
# This recipe deliberately does not use external CONFIG_REPO or legacy bootstrap.
unset CONFIG_REPO CONFIG_REF COMFY_ARGS MODEL_PROFILES OPENROUTER_API_KEY LLM_KEY
profile="${H3_PORTRAIT_PROFILE:-full}"
if [[ "$profile" == "lite" ]]; then
  printf '\nH3 PORTRAIT 1.5.3 | Role-aware + local Reference Pack video + H3 Ref2VA stills | profile=lite\n'
else
  printf '\nH3 PORTRAIT 1.5.3 | Role-aware + local Hearmeman Reference Pack video | profile=full\n'
fi
python /opt/h3-portrait/verify_release.py --check-manifest /opt/h3-portrait/release-manifest.json --node-copy /opt/comfy-bundle/custom_nodes/ComfyUI-H3Portrait
mkdir -p "$COMFY_HOME" "$H3_PORTRAIT_CONFIG" /workspace/h3-portrait
cp -a /opt/h3-portrait/{settings.json,models.json,runtime.json} "$H3_PORTRAIT_CONFIG/"
cp /opt/shared-loras/lora_links.txt "$H3_PORTRAIT_CONFIG/lora_links.txt"
# H3 plus the isolated Krea/SeedVR2 node capability are bundled; weights remain opt-in.
rsync -a --exclude='__pycache__' --exclude='build-smoke.log' --exclude='/models/' --exclude='/input/' --exclude='/output/' --exclude='/user/' --exclude='/temp/' /opt/comfy-bundle/ "$COMFY_HOME/"
python /opt/h3-portrait/verify_release.py --check-manifest /opt/h3-portrait/release-manifest.json --node-copy "$COMFY_HOME/custom_nodes/ComfyUI-H3Portrait"
python /opt/h3-portrait/patch_refpack_local.py --check "$COMFY_HOME/custom_nodes/ComfyUI-MiniMaxRefPack"
mkdir -p "$COMFY_HOME"/{models,input,output,temp,user/default/workflows}
python /opt/h3-portrait/install_workflows.py --home "$COMFY_HOME" --settings "$H3_PORTRAIT_CONFIG/settings.json"
python /opt/runpod-comfy/scripts/comfy_http_fix.py apply "$COMFY_HOME"
# Catch configuration/access errors before starting the expensive generation model.
python /opt/h3-portrait/prepare_assets.py
python /opt/krea-identity/install.py runtime --comfy-home "$COMFY_HOME"
python /opt/h3-media/install.py runtime --comfy-home "$COMFY_HOME"
OLLAMA_PID=''
COMFY_PID=''
cleanup() {
  trap - TERM INT EXIT
  [[ -z "$COMFY_PID" ]] || kill "$COMFY_PID" 2>/dev/null || true
  [[ -z "$OLLAMA_PID" ]] || kill "$OLLAMA_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT
rm -f /workspace/h3-portrait/ollama-ready.json
python -u /opt/h3-portrait/ollama_service.py & OLLAMA_PID=$!
python /opt/h3-portrait/wait_ready.py --pid "$OLLAMA_PID"
# Validate before starting the process; a bad value cannot be silently ignored.
python /opt/h3-portrait/launch_options.py > /workspace/h3-portrait/launch-options.txt
mapfile -t MEMORY_FLAGS < /workspace/h3-portrait/launch-options.txt
cd "$COMFY_HOME"
python main.py --listen 0.0.0.0 --port 8188 --enable-manager --use-pytorch-cross-attention --max-upload-size 1024 --disable-auto-launch "${MEMORY_FLAGS[@]}" & COMFY_PID=$!
python /opt/h3-portrait/smoke_check.py --port 8188 --wait 240
printf '\nH3 PORTRAIT 1.5.3 READY: model downloads verified; open HTTP 8188. Protected model loaders installed.\n'
# If either critical service exits, stop its sibling rather than leave a paid idle Pod.
wait -n "$COMFY_PID" "$OLLAMA_PID"

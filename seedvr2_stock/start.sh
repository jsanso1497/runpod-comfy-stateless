#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/seedvr2-stock
APP=/opt/seedvr2-stock
COMFY=/opt/ComfyUI

# A standard authenticated reverse proxy, not a custom ComfyUI or inference server.
python - <<'CHECK'
import os
p=os.environ.get('COMFY_PASSWORD','')
if not 16 <= len(p.encode('utf-8')) <= 72 or '\n' in p or '\r' in p or '{{' in p:
    raise SystemExit('Set COMFY_PASSWORD to a unique password of 16-72 UTF-8 bytes before starting.')
import torch
if not torch.cuda.is_available():
    raise SystemExit('A CUDA GPU is required. No CPU or lower-precision fallback is enabled.')
print('GPU:', torch.cuda.get_device_name(0))
CHECK
umask 077
printf '%s\n' "$COMFY_PASSWORD" | htpasswd -i -B -c /run/seedvr2-stock.htpasswd workbench >/dev/null
chown root:www-data /run/seedvr2-stock.htpasswd
chmod 640 /run/seedvr2-stock.htpasswd
unset COMFY_PASSWORD
mkdir -p "$ROOT"/{input,output,temp,user/default/workflows/00_SeedVR2_Stock} /workspace/models/SEEDVR2
# Reuse the ordinary upstream model cache, including compatible weights already on this volume.
if [[ -L "$COMFY/models/SEEDVR2" ]]; then
    [[ "$(readlink -f "$COMFY/models/SEEDVR2")" == /workspace/models/SEEDVR2 ]] || { echo 'Unexpected model-cache symlink.' >&2; exit 1; }
elif [[ -d "$COMFY/models/SEEDVR2" ]]; then
    rmdir "$COMFY/models/SEEDVR2" || { echo 'Refusing to replace a nonempty model directory.' >&2; exit 1; }
    ln -s /workspace/models/SEEDVR2 "$COMFY/models/SEEDVR2"
elif [[ ! -e "$COMFY/models/SEEDVR2" ]]; then
    ln -s /workspace/models/SEEDVR2 "$COMFY/models/SEEDVR2"
else
    echo 'Unexpected model-cache path.' >&2; exit 1
fi
for file in "$APP"/workflows/*.json; do
    target="$ROOT/user/default/workflows/00_SeedVR2_Stock/$(basename "$file")"
    if [[ ! -e "$target" ]]; then cp "$file" "$target"; fi
done
nginx -t -c "$APP/nginx.conf"
comfy_pid=''; nginx_pid=''
cleanup() {
    trap - EXIT INT TERM
    [[ -z "$nginx_pid" ]] || kill "$nginx_pid" 2>/dev/null || true
    [[ -z "$comfy_pid" ]] || kill "$comfy_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cd "$COMFY"
python main.py --listen 127.0.0.1 --port 8189 \
    --input-directory "$ROOT/input" --output-directory "$ROOT/output" \
    --temp-directory "$ROOT/temp" --user-directory "$ROOT/user" \
    --use-pytorch-cross-attention &
comfy_pid=$!
nginx -c "$APP/nginx.conf" -g 'daemon off;' &
nginx_pid=$!
echo 'Open RunPod HTTPS port 8188. Login: workbench. Models download on the first Run.'
set +e
wait -n "$comfy_pid" "$nginx_pid"
status=$?
exit "$status"

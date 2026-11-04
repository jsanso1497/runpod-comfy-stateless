#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/seedvr2-stock
APP=/opt/seedvr2-stock
COMFY=/opt/ComfyUI
# Platform setup only. Rendering, queue, progress, preview and downloads are upstream.
python - <<'CHECK'
import os
p=os.environ.get('COMFY_PASSWORD','')
if not 16 <= len(p.encode('utf-8')) <= 72 or any(c in p for c in ('\n','\r','\x00','{{')):
    raise SystemExit('Set COMFY_PASSWORD to a unique 16-72 byte password or a resolved RunPod secret.')
import torch
if not torch.cuda.is_available():
    raise SystemExit('A CUDA GPU is required. No CPU or lower-precision fallback is enabled.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
CHECK
umask 077
printf '%s\n' "$COMFY_PASSWORD" | htpasswd -i -B -c /run/seedvr2-stock.htpasswd workbench >/dev/null
chown root:www-data /run/seedvr2-stock.htpasswd
chmod 640 /run/seedvr2-stock.htpasswd
unset COMFY_PASSWORD
mkdir -p "$ROOT"/{input,output,temp,user/default/workflows/00_SeedVR2_Stock} /workspace/models/SEEDVR2
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
# Archive only the two previously shipped factory graphs, and only if unmodified.
# User-authored workflows, models, videos and outputs are not removed.
python - <<'SETUP'
import hashlib, os, re, shutil
from pathlib import Path
app=Path('/opt/seedvr2-stock'); root=Path('/workspace/seedvr2-stock')
folder=root/'user/default/workflows/00_SeedVR2_Stock'
previous={
 '01_SeedVR2_7B_FP16_1080p.json':'56d6711d9d1e50db44a9bd3ecd84940539039e07',
 '02_SeedVR2_7B_FP16_4K.json':'684c62a7b8180ca5f62e99e3adc9dd7a17177fa3'}
for name, expected in previous.items():
    path=folder/name
    if path.is_file() and not path.is_symlink():
        b=path.read_bytes(); h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if h==expected:
            backup=root/'workflow-backups/pre-clarity-v3'/name
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists(): shutil.copy2(path,backup)
            if backup.read_bytes()==b: path.unlink()
for src in (app/'workflows').glob('*.json'):
    dst=folder/src.name
    if not dst.exists(): shutil.copy2(src,dst)
pod=os.environ.get('RUNPOD_POD_ID','')
config=(app/'nginx.conf').read_text()
if pod:
    if not re.fullmatch(r'[a-z0-9]{6,64}',pod): raise SystemExit('Invalid RUNPOD_POD_ID; refusing nginx configuration.')
    config=config.replace('proxy_set_header Host $http_host;',f'proxy_set_header Host {pod}-8188.proxy.runpod.net;')
Path('/run/seedvr2-stock-nginx.conf').write_text(config)
SETUP
nginx -t -c /run/seedvr2-stock-nginx.conf
comfy_pid=''; nginx_pid=''
cleanup() {
    trap - EXIT INT TERM
    [[ -z "$nginx_pid" ]] || kill "$nginx_pid" 2>/dev/null || true
    [[ -z "$comfy_pid" ]] || kill "$comfy_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cd "$COMFY"
python -u main.py --listen 127.0.0.1 --port 8189 \
    --input-directory "$ROOT/input" --output-directory "$ROOT/output" \
    --temp-directory "$ROOT/temp" --user-directory "$ROOT/user" \
    --cache-none --disable-auto-launch --use-pytorch-cross-attention &
comfy_pid=$!
ready=0
for ((i=0; i<240; i++)); do
    kill -0 "$comfy_pid" 2>/dev/null || { echo 'ComfyUI exited during startup.' >&2; exit 1; }
    if curl -fsS --max-time 2 http://127.0.0.1:8189/system_stats >/dev/null; then ready=1; break; fi
    sleep 1
done
[[ "$ready" == 1 ]] || { echo 'ComfyUI startup timed out.' >&2; exit 1; }
nginx -c /run/seedvr2-stock-nginx.conf -g 'daemon off;' &
nginx_pid=$!
echo 'SeedVR2 Video Clarity: RunPod HTTPS 8188. Username: workbench.'
echo 'Workflow: 00_SeedVR2_Stock / 01_Video_Clarity. Models download on first Run.'
echo 'Progress, logs, MP4 preview and downloads are handled by ComfyUI.'
echo 'Outputs: /workspace/seedvr2-stock/output/SeedVR2'
set +e
wait -n "$comfy_pid" "$nginx_pid"
status=$?
exit "$status"

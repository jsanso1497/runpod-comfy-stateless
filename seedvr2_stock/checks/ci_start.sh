#!/usr/bin/env bash
# CI ONLY: CPU native I/O and actual Nginx. No model weights or SeedVR2 inference.
set -euo pipefail
APP=/opt/seedvr2-stock
DATA=/workspace/seedvr2-stock
mkdir -p "$DATA"/{input,output,temp,user,fixtures} /run
: "${COMFY_PASSWORD:?Missing CI password}"
umask 077
printf '%s\n' "$COMFY_PASSWORD" | htpasswd -i -B -c /run/seedvr2-stock.htpasswd workbench >/dev/null
chown root:www-data /run/seedvr2-stock.htpasswd
chmod 640 /run/seedvr2-stock.htpasswd
unset COMFY_PASSWORD
p1=''; p2=''
cleanup(){ trap - EXIT INT TERM; [[ -z "$p1" ]] || kill "$p1" 2>/dev/null || true; [[ -z "$p2" ]] || kill "$p2" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
# Disable custom nodes only in this CPU fixture; upstream model selectors need a GPU.
python -u /opt/ComfyUI/main.py --cpu --disable-all-custom-nodes --listen 127.0.0.1 --port 8189 \
 --input-directory "$DATA/input" --output-directory "$DATA/output" \
 --user-directory "$DATA/user" --temp-directory "$DATA/temp" --cache-none --disable-auto-launch &
p1=$!
nginx -t -c "$APP/nginx.conf"
nginx -c "$APP/nginx.conf" -g 'daemon off;' &
p2=$!
wait -n "$p1" "$p2"

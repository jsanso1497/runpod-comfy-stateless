#!/usr/bin/env bash
set -euo pipefail
KIT=/opt/qwen21_native
COMFY=/opt/ComfyUI
export DATA_ROOT="${DATA_ROOT:-/workspace/qwen21_native}"
export COMFY_PORT="${COMFY_PORT:-8188}"
export ASSET_DOWNLOAD_WORKERS="${ASSET_DOWNLOAD_WORKERS:-2}"
export ENABLE_BFS_COMPARISONS="${ENABLE_BFS_COMPARISONS:-0}"
PROFILE="$(cat "$KIT/config/built_profile.txt")"
if [[ "$PROFILE" != identity && "$PROFILE" != upscale ]]; then echo 'Invalid baked profile' >&2; exit 1; fi
if [[ ! "$COMFY_PORT" =~ ^[0-9]+$ ]] || (( COMFY_PORT < 1024 || COMFY_PORT > 65535 )); then echo 'Invalid COMFY_PORT' >&2; exit 1; fi
if [[ "$ENABLE_BFS_COMPARISONS" != 0 && "$ENABLE_BFS_COMPARISONS" != 1 ]]; then echo 'ENABLE_BFS_COMPARISONS must be 0 or 1' >&2; exit 1; fi
mkdir -p "$DATA_ROOT"/{models,input,output,user,temp,logs}
rm -f "$DATA_ROOT/logs/runtime_schema_check.json"
extra=()
if [[ "$ENABLE_BFS_COMPARISONS" == 1 ]]; then extra+=(--include-bfs); fi
python "$KIT/scripts/download_models.py" --models-dir "$DATA_ROOT/models" --profile "$PROFILE" --workers "$ASSET_DOWNLOAD_WORKERS" "${extra[@]}"
python "$KIT/scripts/install_into_comfy.py" --comfy-dir "$COMFY" --user-dir "$DATA_ROOT/user" --profile "$PROFILE" "${extra[@]}"
python - <<'PY'
import os,yaml
from pathlib import Path
root=Path(os.environ['DATA_ROOT']).resolve()
profile=Path('/opt/qwen21_native/config/built_profile.txt').read_text().strip()
config={'qwen21_native':{'base_path':str(root/'models'),'diffusion_models':'diffusion_models','text_encoders':'text_encoders','vae':'vae','loras':'loras'}}
if profile=='upscale':
    config['qwen21_native']['seedvr2']='SEEDVR2'
    source=root/'models/SEEDVR2';source.mkdir(exist_ok=True)
    link=Path('/opt/ComfyUI/models/SEEDVR2')
    if not link.exists() and not link.is_symlink():link.symlink_to(source,target_is_directory=True)
(root/'extra_model_paths.yaml').write_text(yaml.safe_dump(config))
PY
# BEGIN QWEN21 TORSO LOCK ADDON
python "$KIT/addons/torso_lock/INSTALL.py" --comfy-dir "$COMFY" --user-dir "$DATA_ROOT/user"
# END QWEN21 TORSO LOCK ADDON
cd "$COMFY"
echo "Starting native Qwen 2.1: profile=$PROFILE, BFS opt-in=$ENABLE_BFS_COMPARISONS"
exec python "$KIT/scripts/serve_checked.py" --profile "$PROFILE" "${extra[@]}" -- \
  python main.py --listen 0.0.0.0 --port "$COMFY_PORT" \
  --input-directory "$DATA_ROOT/input" --output-directory "$DATA_ROOT/output" \
  --user-directory "$DATA_ROOT/user" --temp-directory "$DATA_ROOT/temp" \
  --extra-model-paths-config "$DATA_ROOT/extra_model_paths.yaml" \
  --use-pytorch-cross-attention "$@"

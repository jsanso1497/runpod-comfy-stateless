#!/usr/bin/env bash
set -euo pipefail
KIT=/opt/qwen21_identity
COMFY=/opt/ComfyUI
export DATA_ROOT="${DATA_ROOT:-/workspace/qwen21_identity}"
export COMFY_PORT="${COMFY_PORT:-8188}"
export ASSET_DOWNLOAD_WORKERS="${ASSET_DOWNLOAD_WORKERS:-2}"
PROFILE="$(cat "$KIT/config/built_profile.txt")"
if [[ "$PROFILE" != identity && "$PROFILE" != upscale ]]; then
  echo 'Invalid baked image profile' >&2; exit 1
fi
if [[ ! "$COMFY_PORT" =~ ^[0-9]+$ ]] || (( COMFY_PORT < 1024 || COMFY_PORT > 65535 )); then
  echo 'COMFY_PORT must be between 1024 and 65535' >&2; exit 1
fi
mkdir -p "$DATA_ROOT"/{models,input,output,user,temp,logs}
python "$KIT/scripts/download_models.py" --models-dir "$DATA_ROOT/models" --profile "$PROFILE" --workers "$ASSET_DOWNLOAD_WORKERS"
python - <<'PY'
import os, shutil
from pathlib import Path
import yaml
root=Path(os.environ['DATA_ROOT']).resolve()
kit=Path('/opt/qwen21_identity')
profile=(kit/'config/built_profile.txt').read_text().strip()
model_paths={'qwen21_identity':{'base_path':str(root/'models'),'diffusion_models':'diffusion_models','text_encoders':'text_encoders','vae':'vae','loras':'loras'}}
if profile=='upscale':model_paths['qwen21_identity']['seedvr2']='SEEDVR2'
(root/'extra_model_paths.yaml').write_text(yaml.safe_dump(model_paths))
out=root/'user/default/workflows/Qwen21 Identity v1.0'
out.mkdir(parents=True,exist_ok=True)
for source in (kit/'workflows').glob('*.json'):
    if source.name.startswith('06_') and profile!='upscale':continue
    target=out/source.name
    # Preserve saved personal edits on persistent storage; versioned folder avoids migration surprises.
    if not target.exists():shutil.copy2(source,target)
# SeedVR2 also uses its default directory for validation cache. Point that directory at our single copy.
if profile=='upscale':
    source=root/'models/SEEDVR2';source.mkdir(exist_ok=True)
    link=Path('/opt/ComfyUI/models/SEEDVR2')
    if not link.exists() and not link.is_symlink():link.symlink_to(source,target_is_directory=True)
PY
cd "$COMFY"
echo "Starting isolated Qwen 2.1 profile: $PROFILE. No unrelated model downloads."
# No eval, no arbitrary auto-updates, no inherited model manifests, no external model APIs.
exec python main.py --listen 0.0.0.0 --port "$COMFY_PORT" \
  --input-directory "$DATA_ROOT/input" --output-directory "$DATA_ROOT/output" \
  --user-directory "$DATA_ROOT/user" --temp-directory "$DATA_ROOT/temp" \
  --extra-model-paths-config "$DATA_ROOT/extra_model_paths.yaml" \
  --use-pytorch-cross-attention "$@"

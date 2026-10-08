#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${COMFY_HOME:-/workspace/ComfyUI}"
MODELS=bf16
SAM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --comfy-home) HOME_DIR="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --sam) SAM=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$MODELS" in none|bf16|int8|both) ;; *) echo 'Use --models none, bf16, int8, or both.' >&2; exit 2;; esac
PYTHON="${PYTHON:-python}"
CHECK=(--comfy-home "$HOME_DIR")
if [[ "$SAM" == 1 ]]; then CHECK+=(--sam); fi
"$PYTHON" "$HERE/preflight.py" "${CHECK[@]}"
# Keep backups outside custom_nodes, so ComfyUI does not load duplicate classes.
"$PYTHON" - "$HERE" "$HOME_DIR" <<'PY'
from pathlib import Path
import json, shutil, sys, time
root, home = map(lambda v: Path(v).resolve(), sys.argv[1:])
if not (home/'main.py').is_file():raise SystemExit('Not a ComfyUI installation: '+str(home))
destination=home/'custom_nodes'/'ComfyUI-Qwen21-PhotoEdit'
if destination.exists():
    backup=home.parent/'qwen21-photo-backups'/str(time.time_ns())
    backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(destination,backup)
    print('Existing add-on backed up to:', backup)
shutil.copytree(root/'node',destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
workflows=home/'user/default/workflows/Qwen21_Photo_Edit'
workflows.mkdir(parents=True,exist_ok=True)
for source in (root/'workflows').glob('*.json'):
    if source.name.endswith('.api.json'): continue
    target=workflows/source.name
    if target.exists():
        print('Preserved existing user workflow:',target)
    else:
        shutil.copy2(source,target)
print('Custom nodes:',destination)
print('Workflows:',workflows)
PY
DOWNLOAD=(--comfy-home "$HOME_DIR" --models "$MODELS")
if [[ "$SAM" == 1 ]]; then DOWNLOAD+=(--sam); fi
"$PYTHON" "$HERE/download_models.py" "${DOWNLOAD[@]}"
printf '\nInstalled. Restart ComfyUI, then refresh the browser. Do not terminate the pod.\n'
printf 'Open Qwen21_Photo_Edit. Begin with run_edit OFF to inspect the mask.\n'

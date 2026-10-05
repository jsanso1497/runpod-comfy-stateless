#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPTS=/opt/runpod-comfy/scripts
CONFIG=/opt/runpod-comfy/default-config
CORE=/opt/comfy-bundle
CONSTRAINTS=/opt/video-restore/constraints.txt
PIN="$(python "$SCRIPTS/catalog.py" comfy-ref --config "$CONFIG")"
test -d /opt/seedvr2
test -f "$CONSTRAINTS"
mkdir -p "$CORE"
git -C "$CORE" init -q
git -C "$CORE" remote add origin https://github.com/Comfy-Org/ComfyUI.git
git -C "$CORE" fetch --depth 1 origin "$PIN"
git -C "$CORE" checkout --detach FETCH_HEAD
test "$(git -C "$CORE" rev-parse HEAD)" = "$PIN"
export PIP_CONSTRAINT="$CONSTRAINTS"
python -m pip install --disable-pip-version-check -r "$CORE/requirements.txt" -r "$CORE/manager_requirements.txt"
python -m pip check
(cd "$CORE" && python -c 'import comfyui_manager; print("Manager import PASS")')
mkdir -p "$CORE/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler"
cp -a /opt/seedvr2/. "$CORE/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/"
python "$SCRIPTS/install_nodes.py" --manifest "$CONFIG/custom_nodes.json" --comfy-home "$CORE" --constraints "$CONSTRAINTS"
cp -a "$SCRIPTS/local_nodes/". "$CORE/custom_nodes/"
python -m pip check
python "$SCRIPTS/check_rebalance_runtime.py" --comfy-home "$CORE"
python "$SCRIPTS/check_single_person_runtime.py" --comfy-home "$CORE"
python -m unittest discover -s /opt/runpod-comfy/tests -p "test_*.py"
python "$SCRIPTS/comfy_http_fix.py" self-test
python "$SCRIPTS/comfy_http_fix.py" apply "$CORE"
python "$SCRIPTS/catalog.py" validate --config "$CONFIG" --comfy-home "$CORE"
python "$SCRIPTS/check_workflows.py" static --config "$CONFIG"
python "$SCRIPTS/check_workflows.py" build-smoke --config "$CONFIG" --comfy-home "$CORE" --port 18188 --timeout 240
printf '%s\n' "$PIN" > /opt/runpod-comfy/comfy-build-ref.txt
cp "$CONFIG/custom_nodes.json" /opt/runpod-comfy/bundled-custom-nodes.json
python -m pip freeze > /opt/runpod-comfy/resolved-packages.txt
printf '\nBuild checks PASS. Model weights and GPU generation are NOT tested during this build.\n'

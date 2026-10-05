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
# Run every suite belonging to the current ComfyUI package explicitly.
# Folder-merge updates can leave tests from the retired upload UI beside these
# suites (for example test_restore.py). Broad discovery imports that legacy app
# and its pytest dependency even though the app is not shipped by this recipe.
# Do not install legacy dependencies or suppress failures in active test suites.
CURRENT_TEST_MODULES=(
  test_krea_rebalance
  test_ollama_h3
  test_quality
  test_single_person
  test_update
  test_user_directed_h3
)
TEST_ROOT=/opt/runpod-comfy/tests
for test_module in "${CURRENT_TEST_MODULES[@]}"; do
  if [[ ! -f "$TEST_ROOT/$test_module.py" ]]; then
    printf 'BUILD_TEST_SELECTION: Missing required suite: %s/%s.py\n' \
      "$TEST_ROOT" "$test_module" >&2
    exit 2
  fi
done
printf '\nRunning current ComfyUI suites (legacy upload-app tests excluded):\n'
printf '  %s\n' "${CURRENT_TEST_MODULES[@]}"
(
  cd "$TEST_ROOT"
  python -m unittest "${CURRENT_TEST_MODULES[@]}"
)
python "$SCRIPTS/comfy_http_fix.py" self-test
python "$SCRIPTS/comfy_http_fix.py" apply "$CORE"
python "$SCRIPTS/catalog.py" validate --config "$CONFIG" --comfy-home "$CORE"
python "$SCRIPTS/check_workflows.py" static --config "$CONFIG"
python "$SCRIPTS/check_workflows.py" build-smoke --config "$CONFIG" --comfy-home "$CORE" --port 18188 --timeout 240
printf '%s\n' "$PIN" > /opt/runpod-comfy/comfy-build-ref.txt
cp "$CONFIG/custom_nodes.json" /opt/runpod-comfy/bundled-custom-nodes.json
python -m pip freeze > /opt/runpod-comfy/resolved-packages.txt
printf '\nBuild checks PASS. Model weights and GPU generation are NOT tested during this build.\n'

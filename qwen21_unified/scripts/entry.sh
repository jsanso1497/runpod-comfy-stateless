#!/usr/bin/env bash
set -Eeuo pipefail
export COMFY_PORT="${COMFY_PORT:-8188}"
export QWEN_PHOTO_MODEL_PRECISION="${QWEN_PHOTO_MODEL_PRECISION:-bf16}"
export QWEN_PHOTO_DOWNLOAD_SAM="${QWEN_PHOTO_DOWNLOAD_SAM:-1}"
if [[ "$QWEN_PHOTO_MODEL_PRECISION" != "bf16" && "$QWEN_PHOTO_MODEL_PRECISION" != "int8" && "$QWEN_PHOTO_MODEL_PRECISION" != "both" && "$QWEN_PHOTO_MODEL_PRECISION" != "none" ]]; then
  echo 'Invalid QWEN_PHOTO_MODEL_PRECISION' >&2; exit 1
fi
# SAM workflow is bundled; avoid falsely advertising it if checkpoint is skipped.
if [[ "$QWEN_PHOTO_MODEL_PRECISION" == "int8" ]]; then
  echo 'Photo INT8 alone does not supply the BF16 model required by the Native and Torso workflows. Use bf16 or both, or manually supply BF16 and select none.' >&2; exit 1
fi
if [[ "$QWEN_PHOTO_DOWNLOAD_SAM" != 1 ]]; then
  echo 'WARNING: SAM3 workflow installed, but SAM3 weight auto-download is disabled.' >&2
fi
python /opt/qwen21_unified/scripts/runtime_prepare.py
# Preserve the Photo image's original startup, checkpoint selection, ports, and model installer.
exec python /opt/qwen21_unified/scripts/supervise.py -- /opt/qwen21_photo_edit/start.sh "$@"

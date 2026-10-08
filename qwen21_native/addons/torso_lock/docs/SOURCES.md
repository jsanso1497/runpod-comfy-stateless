# Implementation references

Reviewed October 7, 2026. These are source contracts, not evidence that this
new integration has been GPU-tested or benchmarked on the user's photographs.

- Native Qwen 2.1 ComfyUI workflow, model filenames, slot roles, sampling, resolution,
  cache behavior: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1
- Existing suite's pinned Qwen encoder source:
  https://github.com/Comfy-Org/ComfyUI/blob/b00c6e95279053474955540ba4f551646722b9aa/comfy_extras/nodes_qwen.py
- Existing suite's pinned AusBoss crop/stitch implementation:
  https://github.com/ausboss/ComfyUI-AusBoss/blob/ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb/nodes/node_inpaint_crop_stitch.py
- Existing suite's crop/stitch documentation:
  https://github.com/ausboss/ComfyUI-AusBoss/blob/ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb/js/docs/AUSBOSS_NODES_CropForInpaint.md

No model files, source photographs, credentials, font files or downloaded
upstream code archives are redistributed in this addon.

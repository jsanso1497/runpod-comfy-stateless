# Primary source references

Checked public documentation: October 7, 2026. Code pins and model hashes are
retained from the prior kit. No automatic source upgrades are performed.

- Native ComfyUI Qwen 2.1 instructions: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1
- Qwen model author documentation: https://github.com/QwenLM/Qwen-Image-2.1
- Native encoder implementation: https://github.com/Comfy-Org/ComfyUI/blob/b00c6e95279053474955540ba4f551646722b9aa/comfy_extras/nodes_qwen.py
- AusBoss crop/stitch implementation: https://github.com/ausboss/ComfyUI-AusBoss/blob/ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb/nodes/node_inpaint_crop_stitch.py
- SeedVR2 node integration: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tree/4490bd1f482e026674543386bb2a4d176da245b9
- BFS body adapter limitations: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md

The workflows are original integrations built around these nodes, not a claim
to redistribute AusBoss's exact Person Swap workflow. Third-party licenses and
base-model terms remain applicable. Native-vs-BFS image quality has not been
benchmarked on the user's photos.

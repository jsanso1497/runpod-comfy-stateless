# Upstream references and attribution

Reviewed October 7, 2026. The package contains original integration code and workflow graphs. Third-party source is fetched from its own repository during the Docker build and retains its license files. Model weights are downloaded separately, not redistributed inside this ZIP. The code is not endorsed by AusBoss, Comfy, Qwen or BFS.

## Native Qwen support

- Official ComfyUI Qwen Image 2.1 guide: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1
- Native encoder and lossless cache node, pinned source: https://github.com/Comfy-Org/ComfyUI/blob/b00c6e95279053474955540ba4f551646722b9aa/comfy_extras/nodes_qwen.py
- Core loaders, pinned source: https://github.com/Comfy-Org/ComfyUI/blob/b00c6e95279053474955540ba4f551646722b9aa/nodes.py
- Comfy-distributed model files: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main

The implementation uses native encoding with independently sized references and its matching output latent. BF16 is a precision selection, not a separate "higher-resolution edition." The build adds no prompt enhancer.

## BFS adapters

- Body guide: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md
- Head guide: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md
- Adapter files: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/tree/main

The documented body reference is square, standing, front-facing and plain-background, approximately 0.59 MP, with a larger scene input. The body adapter transfers clothing as well as physique. Pose preservation is approximate. Standard head v1.1 is the selected default; the alternative adapter is not downloaded.

## AusBoss

- Repository: https://github.com/ausboss/ComfyUI-AusBoss
- Pinned crop-and-stitch implementation: https://github.com/ausboss/ComfyUI-AusBoss/blob/ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb/nodes/node_inpaint_crop_stitch.py
- Pinned memory cleanup: https://github.com/ausboss/ComfyUI-AusBoss/blob/ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb/nodes/node_free_memory.py
- Crop help: https://github.com/ausboss/ComfyUI-AusBoss/blob/main/js/docs/AUSBOSS_NODES_CropForInpaint.md
- Stitch help: https://github.com/ausboss/ComfyUI-AusBoss/blob/main/js/docs/AUSBOSS_NODES_StitchInpaint.md

This package uses actual AusBoss crop, stitch and cleanup nodes. It does not also install the separate lquesada crop pack because that would duplicate functionality. Blended boundary pixels are part of the edit; untouched means outside the actual blend footprint.

## Optional SeedVR2

- Node repository: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- Pinned source: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tree/4490bd1f482e026674543386bb2a4d176da245b9
- Full-precision 7B model and checksum: https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_7b_fp16.safetensors
- VAE and checksum: https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/ema_vae_fp16.safetensors

The optional graph uses batch 1 for a still image and full-precision FP16 weights. The output may be a more detailed candidate without being a more faithful identity match. No independent benchmark or personal image test was performed for this package.

## Deployment and source-image prompting

- Runpod template REST request: https://docs.runpod.io/api-reference/templates/POST/templates
- Official PyTorch image family: https://hub.docker.com/r/pytorch/pytorch/tags
- ChatGPT image creation/editing help: https://help.openai.com/en/articles/8932459-creating-images-in-chatgpt

The supplied prompts are original instructions, not quoted official prompts or guarantees of image-generation behavior. They prioritize primary-reference authority, independent reference roles and comparison against the genuine originals.

## License scope

Source/model licenses remain those of their respective publishers. This integration does not relicense third-party weights, strip notices or disable license conditions. Personal use is the stated project context. No lower-quality substitute was selected on licensing grounds.

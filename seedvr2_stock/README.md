# SeedVR2 Stock FP16

Prepared November 4, 2026. A standalone ComfyUI build with the ordinary upstream SeedVR2 nodes.

## What it does

Load Video -> Get Video Components -> SeedVR2 -> Create Video -> Save Video.
Two model-loader nodes feed SeedVR2. That is the complete seven-node processing graph.
The extra note is not a processing node. Source audio and frame rate are wired directly into
Create Video. There is no prompt, Qwen pass, source blend, face restoration, custom
streaming worker, frame interpolation, scene detector, or additional sharpening model.

The standard 7B FP16 checkpoint is selected, not Sharp, FP8, INT8, or GGUF. The VAE is FP16.
FP16 describes the checkpoint precision; unmodified upstream code controls compute dtypes.
No claim is made that SeedVR2 is the best upscaler for every possible clip.

## Install into your local GitHub repository

Download `install_seedvr2_stock_fp16.py` to Downloads. Open a terminal at the root of your
local `runpod-comfy-stateless` repository:

```sh
python3 -c "from pathlib import Path; import runpy; runpy.run_path(str(Path.home() / 'Downloads' / 'install_seedvr2_stock_fp16.py'), run_name='__main__')"
```

The installer only adds this folder and `.github/workflows/build-seedvr2-stock.yml`.
It verifies its payload and refuses conflicting existing files. It does not fetch, stage,
commit, push, trigger Actions, or modify other workflows. Re-running it is safe if the
installed files are unchanged. A `--check` flag previews changes when running it directly.

Review and commit the files in GitHub Desktop, then manually Push origin. In GitHub,
run **Actions > Build SeedVR2 Stock FP16 > Run workflow**. After that build succeeds,
copy the image digest from its summary into RunPod.

The future image tag is:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:seedvr2-stock-fp16
```

This package has NOT published that image. Do not deploy the tag before a successful build.
A private GHCR package needs your existing GHCR registry credentials on RunPod.

## RunPod setup

| Field | Value |
| --- | --- |
| GPU | Recommended first target: A100 80 GB, especially for 4K |
| Host RAM | 128 GB or more preferred |
| Container disk | 30 GB |
| Volume | 100 GB or more mounted at `/workspace` |
| Image | Exact digest from the successful new build |
| HTTP port | 8188 only |
| Start command override | Leave blank |
| Username | workbench |

These are planning allocations, not measured minimums. An A40 48 GB is a lower-cost
1080p test candidate, but no cost, availability, speed, or fit guarantee is made.
Do not deploy this over a running Pod with unsaved files. Keep original recordings backed up.
A network volume is useful when weights and outputs need to survive deletion/replacement of a Pod.

Required environment variable:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

Use an actual secret containing a unique password of 16-72 UTF-8 bytes, or enter that password
as the environment value. An unresolved secret reference prevents startup. No HF token,
CivitAI token, WB_TASKS, WB_TOOLBOX, Ollama setting, or old video-fidelity setting is required.
The selected model files are public. Connect through RunPod's HTTPS link for HTTP port 8188.
Do not expose internal port 8189.

## Use it

Open **Workflows > 00_SeedVR2_Stock**, choose the 1080p or 4K JSON, upload the video in
**1. Upload video**, then Run. The first run downloads the two selected weights automatically.
The image contains no model weights, so it remains independent of those large downloads.

| Setting | 1080p | 4K |
| --- | --- | --- |
| DiT | seedvr2_ema_7b_fp16.safetensors | Same |
| VAE | ema_vae_fp16.safetensors | Same |
| Short edge / long-edge cap | 1080 / 1920 | 2160 / 3840 |
| Temporal batch | 9 | 9 |
| Uniform batch / temporal overlap | On / 3 | Same |
| Color correction | LAB | LAB |
| Added input / latent noise | 0 / 0 | Same |
| VAE encode tile / overlap | 1024 / 128 | Same |
| VAE decode tile / overlap | 768 / 128 | Same |
| DiT block swap / offload | 32 / CPU | Same |
| Attention | PyTorch SDPA | Same |
| torch.compile | Not connected | Not connected |

The aspect ratio is retained. These are bounding targets, not a forced 16:9 crop, an
exact-size canvas guarantee, or an automatic 2x multiplier. For ordinary 16:9 footage,
the targets are 1920x1080 and 3840x2160. A non-16:9 clip can have other dimensions.
The conservative 9-frame batch is an engineering starting point, not a measured quality optimum.
Larger batches such as 13 or 17 offer more temporal context but require more memory.
Block swapping moves weights to CPU; it does not replace them with quantized weights.

## Keep the stock limitation in mind

This workflow decodes the entire selected clip into RAM and holds output frames in RAM.
A small temporal batch reduces GPU working memory; it DOES NOT make long videos RAM-bounded.
Start with an approximately five-second source clip at 4K and inspect the result before a
longer render. As an illustration, 300 RGB float32 4K frames alone occupy about 29.9 GB
(decimal), before input frames, model weights, processing buffers, and export copies.
Do not interpret this as a tested five-second maximum or a guarantee that any duration fits.

Use progressive, square-pixel, SDR, constant-frame-rate source video. The stock graph
has no custom media validation. Do not use it as an HDR/log, VFR timing, multi-audio-track,
subtitle, timecode, alpha, or archival metadata-preservation pipeline. Source FPS and the
ordinary decoded audio track are reattached; audio is not promised to be bit-for-bit copied.
The stock Save Video output is a compressed MP4, not a lossless master. No custom encoding
or timestamps layer has been added. Check audio sync and motion after export.

SeedVR2 is generative. Standard 7B can still alter small features or invent texture.
FP16 does not guarantee faithful identities, real recovered text, or flicker-free motion.
There is no source blending here: you receive the actual model output.

If CUDA memory runs out, try temporal batch 5 first; do not use 1 for video temporal
processing. For a VAE-stage OOM, reduce the offending encode/decode tile to 512. For a
DiT-stage OOM, increase block swapping to 36. These are upstream controls, not an
automatic fallback. Do not silently change the model or precision.

## Storage

```text
/workspace/models/SEEDVR2/                 # Shared upstream model cache
/workspace/seedvr2-stock/input/            # Uploaded source video
/workspace/seedvr2-stock/output/SeedVR2/    # Saved video
/workspace/seedvr2-stock/user/             # Workflows and settings
/workspace/seedvr2-stock/temp/             # ComfyUI temporary files
```

Outputs and model files are not automatically deleted. Stop the Pod when finished;
closing the browser does not stop the instance. Export important files before terminating it.

## Existing live Pod

The two JSON files can be loaded directly into a normal ComfyUI installation that already
has `ComfyUI-SeedVR2_VideoUpscaler` v2.5.24 and the stock video nodes. Use **Load workflow**
or drag the JSON onto the canvas. Having a standalone SeedVR2 CLI elsewhere on disk is
not the same as having its nodes registered with ComfyUI. The earlier Faithful Video
container does not register these upstream nodes, so the standalone new build is the
clean path for that container. This package does not mutate or restart a live Pod.

## Verification

Locally: JSON structure, all links/types, model selections, audio/FPS routing, Python
syntax, shell syntax, installer payload checks, and installer dry-run/conflict/idempotency
behavior. Docker and CUDA are not available in the authoring environment, so neither
image build nor model inference/VRAM/visual quality is certified here.

During your build: check the graph against actual pinned upstream execute signatures,
then run unmodified core Load/Get/Create/Save video nodes on a synthetic one-second
CPU clip and check output frame count, dimensions, and presence of audio. Custom nodes
are disabled for this media-only check because upstream GPU selectors require a GPU.
No upstream source is patched and no AI inference runs in CI. An image is published only
after those checks succeed. A green build is not a GPU quality test.

## Primary sources

- Upstream integration and settings: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- Source template: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/example_workflows/SeedVR2_HD_video_upscale.json
- Model repository: https://huggingface.co/numz/SeedVR2_comfyUI
- Stock core video nodes: https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy_extras/nodes_video.py
- PyTorch CUDA wheels: https://pytorch.org/get-started/previous-versions/
- RunPod storage: https://docs.runpod.io/pods/storage/overview

See `sources.json` for source pins and the explicit changes from the upstream template.

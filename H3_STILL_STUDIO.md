# H3 Still Studio: generate, edit, or use references

Release 2026-11-06.1. Prepared against repository main f14db6a9d34e31ed5515294f79595e94bc8b1796.

This update consolidates the earlier padding and browser-terminal repairs. Do not stack those older installers on top of this update. It updates only the dedicated H3 Still workspace, not Qwen, MOSS, SeedVR2, the separate H3 video workspace, or their image tags.

## Install on your Mac

Save `update_h3_still_studio.py` in Downloads. In a terminal opened inside your local repository:

```bash
python3 "$HOME/Downloads/update_h3_still_studio.py"
```

The installer uses Python 3.8-compatible syntax and standard-library functions. It checks every change before writing, backs up recognized files under Git's metadata directory, and refuses unknown local edits. It does not fetch, stage, commit, push, build, or deploy. `--dry-run` previews the update.

Review and commit in GitHub Desktop, then manually Push origin there. Run **Actions > Build H3 Still Developer Workspace > Run workflow** on the updated branch. Start a NEW run, not a rerun of an older commit. After it succeeds, use the image digest printed in that run's summary.

## RunPod

Use an A100 80 GB or comparable compatible GPU and at least 128 GB host RAM as the first-test planning target, not a validated minimum. Use 30 GB container disk and preferably a 150 GB persistent volume at `/workspace`. Expose HTTP **8188,8888**, no TCP ports required, and leave the entrypoint/start command overrides blank. The image uses PyTorch 2.9.1 CUDA 13.0 and needs a compatible host driver.

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Only COMFY_PASSWORD is required. It must resolve to 16-72 UTF-8 bytes without newlines. HF_TOKEN is optional for the public model repositories. Do not add legacy `WB_WORKSPACE`, prompt-enhancer settings, or Ollama settings.

ComfyUI: HTTPS 8188, username `workbench`, your password. JupyterLab: HTTPS 8888, log in with the same password, then Launcher > Terminal. Jupyter starts before model preparation and remains available if ComfyUI stops. This is independent of RunPod's managed Web Terminal toggle; its original failure was not diagnosed from logs. The updater does not claim to repair the provider's toggle itself.

Preserve `/workspace` when changing images. A stopped Pod is not the same as a terminated Pod. Do not terminate an ordinary Pod volume containing needed data. Stop the running Pod when finished; no automatic billing-stop feature is included.

## One workflow, three jobs

Open **Workflows > 00_H3_Studio_v1 > 00_H3_Studio_Auto_INT8**. The new image seeds this exact file without overwriting your saved graphs. The older factory folder is preserved separately. The same JSON is bundled for direct import AFTER the new image is built. Importing it into the old image will report missing Studio helper nodes; the installer and rebuild are required once.

| Your inputs | Auto mode does | Checkpoint |
| --- | --- | --- |
| No active photo | Generate from the prompt | FL2VA INT8 |
| One active photo | Edit that photo | FL2VA INT8, matching the developer's editing route |
| Two to nine active photos | Generate using references | REF2VA INT8 |

For **one reference and a new scene**, choose **References** rather than Auto. For strict behavior, the explicit Generate/Edit/References modes validate their required photo counts instead of silently ignoring photographs.

Upload photos in the reference slots. Unused slots can remain `(none)`. You can temporarily switch an uploaded photo's `enabled` toggle OFF without deleting it. Slots 4-9 are collapsed initially; expand them when needed. Blank slots never produce black/placeholder photos and never call the image loader.

Fill slots left to right where practical. Active photos are numbered consecutively in slot order. If only slots 3 and 9 are active, those photos become `<Picture 1>` and `<Picture 2>`. The **Route + photo-tag report** shows this mapping and the selected model after execution. Replacing a file under the same name invalidates its image cache.

Edit the large prompt field, refer to photographs as `<Picture 1>`, `<Picture 2>`, etc., then Run. There is no prompt enhancer, background language-model service, or automatic prompt rewrite.

## Size and sampling

Generation and References use the selected output aspect and `generation_megapixels`, initially 4 MP. Increase to 8 MP for a high-resolution comparison. Edit follows the original photo's aspect with `edit_megapixels`, initially 2.5 MP. Dimensions are rounded to 32-pixel multiples and constrained to the Fizgig latent node's limits. Rounding can slightly change the aspect ratio; pixel-perfect alignment to the original is not promised.

The default **Quality** preset is 50 steps, er_sde/simple, with Turbo disabled. Its use for reference editing is an integration starting point, not a benchmark-proven optimum. The **Developer fast** preset uses 20 steps and the existing v4 step600 EMA Turbo LoRA at 0.38, and is valid only for Generate/Edit on FL2VA. Choosing it for References raises a clear error rather than applying an FL2VA adapter to REF2VA.

References have an independent megapixel cap, initially 2.5 MP per photo. Small supporting photos are not enlarged for invented detail, apart from necessary 32-pixel alignment. Single-photo Edit follows the developer's resize-to-output approach. Start reference testing with two photographs. Nine inputs are supported by the graph and upstream reference node, but nine high-resolution references fitting the planned GPU has NOT been verified.

The native H3 conditioning node's unused video latent is discarded. It is called with the minimum valid length of five for image-only conditioning, avoiding the original unused 124-frame allocation. The sampler receives the separate Fizgig **one-frame** latent. Only the Fizgig decoder feeds the final PNG.

Final images: `/workspace/h3-still/output/H3_Studio/`. Edit also saves a separate comparison with the resized original on the left and generated result on the right. The original file remains untouched. The comparison is skipped for generation/reference composition, where there is no single before-image to compare.

## LoRAs

Two separate H3-only adapter slots are included and OFF by default. Upload trusted compatible `.safetensors` files through Jupyter to:

```text
/workspace/h3-still/models/loras
```

Refresh ComfyUI's node/model definitions or restart ComfyUI so newly uploaded adapters appear. Select a file, set its documented strength and enable the slot. Turbo remains independently controlled by the sampling preset. Use `(none)`, OFF, or strength 0 to bypass a personal LoRA. No private identity or style LoRA is downloaded automatically.

The helper delegates to the native model-only loader. It rejects a completely unmatched adapter when native patch counts are available. Matching some keys is not proof of full compatibility or of likeness preservation. Check warnings and make an on/off comparison. Qwen, FLUX, and Krea adapters are not H3 adapters. Verify a personal adapter's compatibility with FL2VA versus REF2VA.

## Quantized model policy and storage

Kept: FL2VA pruned INT8/ConvRot, NVFP4/AWQ required H3 encoder, INT8/ConvRot video VAE and the existing Turbo LoRA. Added: **REF2VA pruned INT8/ConvRot**, about 20.97 GB, for the References route. No BF16 base checkpoint, BF16 encoder, alternate VAE, P-H3 stack, or prompt-upsample model is added.

The cache is about 61 GB in total. On an intact previous cache, only the approximately 21 GB reference checkpoint is new. Only the chosen diffusion model is loaded by the routing helper for each run. Actual peak CPU/GPU memory depends on ComfyUI's cache/offloading and the workload; it is not established by the download total.

Model revisions and SHA-256 hashes are locked. Existing verified downloads are reused. Inputs, outputs, private adapters and personal workflows are never deleted by this update.

## Limits and validation

These are whole-image generative edits, not protected-mask inpainting or proven faithful restoration. Reference generation can change identity, anatomy, details or composition. A syntactically valid graph does not establish image quality.

The provided local report records executed offline tests. The build must separately pass actual pinned ComfyUI CPU startup, installed schemas, native resizing, zero/one/two/nine-photo routing, disabled-slot handling, native conditioning with synthetic encoder/VAE fixtures, one-frame latent geometry, password gateway/WebSocket checks, and a real Jupyter terminal command before publishing the image.

Those CPU tests do NOT load actual model weights. GPU sampling, LoRA image quality, 8 MP memory, nine-reference memory, mobile layout and the deployed RunPod instance are not certified. Run one text-only image, one edit, and a two-reference generation first, then test a compatible personal LoRA before increasing size/reference count.

## Source and license notes

The ComfyUI and Fizgig source pins are unchanged. Studio helpers perform input selection/routing only; model loading, LoRA application, conditioning, sampling and decoding remain upstream implementations. RMBG is no longer installed in the image just for resizing. Old source provenance is retained.

Model rights are unchanged. The H3 Community License contains territorial and other requirements; confirm authorization before deployment, especially for work in excluded territories or client use. This package contains no model weights and does not grant model rights. Do not treat the MIT license on Fizgig's nodes as the model license.

Primary references:
- https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/10d5171f6e45b0757a03df977f53a660e060c33d
- https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy_extras/nodes_minimax_h3.py
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/Comfy-Org/MiniMax-H3/commit/3f57e8291d2ef846f9a074b1b76d2767db434abe
- https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

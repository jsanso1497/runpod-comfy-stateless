# H3 First + Last Frame, Identity HQ

Release: 2026-10-19.1. This is a source update, not a GPU-certified or user-approved workflow.

## What was added

`HFL` adds two presets of one workflow: portrait (768 x 1344) and landscape (1344 x 768). Both use the existing `minimax_h3_ref2va_bf16.safetensors` checkpoint. They do not substitute FL2VA weights.

The first photograph enters the native reference-conditioning node as `<Picture 1>` and a native Add Guide node at frame 0. The last photograph enters as `<Picture 2>` and a second Add Guide at frame -1. The positive conditioning is chained through both guides into the sampler. The video VAE is connected to reference conditioning, so references reach the diffusion model as well as the text encoder. An optional third image socket accepts a close facial reference. This follows the current native multiframe approach [1, 2].

Generation saves an RGB16 FFV1 Matroska master and an MP4 browser preview. `HFL4K` is a separate SeedVR2 finishing workflow. It reads the native MKV directly from ComfyUI output storage, offloads Comfy-managed models, restores/upscales the frames, and saves a separate master and preview. Original audio and frame rate travel around the upscaler, not through a second H3 generation pass.

The helpers do not implement a new diffusion sampler, face swap, face restoration or endpoint compositor. All H3 reference conditioning, guide injection, sampling and native VAE decoding remain visible native nodes.

## Install locally

Use Python 3.10 or newer. Save and extract the ZIP in Downloads. Run `apply_update.py --repo .` from the root of your existing `runpod-comfy-stateless` checkout. The installer verifies the local origin, checks package hashes, merges the two catalog entries and the H3 model configuration, and applies three small source integrations. It does not fetch, commit, push, build an image or download weights.

Existing files changed:

- `catalog/models.json`: H3 core pin, helper registration, HFL/HFL4K startup selections and hardware planning values. Other model rows and all external-node pins are preserved.
- `catalog/tasks.json`: add HFL and HFL4K, without removing older workflows.
- `deploy/build.py`: verify the exact inspected native H3 source when building H3.
- `src/workbench/runtime_recovery.py`: use dense PyTorch attention and disable the optional Comfy compiler by default only for H3. Other workspaces keep their previous launch behavior.
- `src/workbench/runtime.py`: expose MKV files in the existing media browser alongside MP4/MOV.

The new source files, workflow JSONs, API exports, tests and this guide are added. Qwen/SAM weights, the approved-workflow catalog, the SAM-local-edit image and the existing SeedVR2 source pin are not changed. The sole approved SAM workflow keeps its approval; the new H3 workflows are explicitly unapproved pending your review.

Backups are saved beside the extracted installer, outside your repository, in `backups/`. The installer prints the exact backup path. It refuses unexpected source anchors, unrelated local H3 core pins, new-file collisions, symlink escapes and conflicting task IDs before writing. Repeat application of the same package is a no-op. A write failure rolls back completed writes. It never resets or cleans the checkout.

To roll back the source patch:

```bash
python3 "$HOME/Downloads/H3_First_Last_HQ_Update_2026-10-19/apply_update.py" --repo . --restore "/the/exact/backup/path/printed/by/the/installer"
python3 scripts/build_site.py
```

Restore refuses to overwrite edits made after installation. Regenerated site output is not in the source backup, so rebuild the site after a restore.

## Validate, publish and launch

After applying locally:

```bash
python3 -m unittest discover -s tests -p 'test_h3_first_last_hq.py' -v
python3 scripts/build_site.py
python3 scripts/validate.py --no-imports
git diff --check
git status --short
```

The optional CPU media tests require numpy, torch, ffmpeg and ffprobe. They skip on a machine without those dependencies; they do not download models. The graph and launch tests need only Python. The repository's existing CI remains the full integration gate. Review all changes and commit/push manually in GitHub Desktop.

After your push, run **Actions > Build HQ workspace > Run workflow > workspace: h3**. Use the image digest published by that successful build. An old `h3-hq` image will not contain these helpers or the new native core pin. Do not run this H3 graph in the separate SAM-local-edit image.

For the rebuilt RunPod image, set:

```text
WB_WORKSPACE=h3
WB_TASKS=HFL,HFL4K
WB_TOOLBOX=standard
WB_DISABLE_COMFY_COMPILER=1
```

Keep the existing authentication configuration and Hugging Face secret mapping. Public HTTP port remains 8188; the managed Comfy backend remains 127.0.0.1:8189. Do not start a second Comfy process. With these task selections, normal startup prepares the H3 and SeedVR2 groups. No Ollama or turbo model is newly required. An existing explicit WB_TASKS value must be changed; image defaults do not override an environment variable.

After model preparation, run this read-only check inside the Pod terminal:

```bash
python -m workbench.h3_first_last_contract --live
```

It checks registered node names, selected options and prepared-model visibility. It does not submit a prompt or certify runtime memory or image quality. Then run one complete 124-frame generation and the separate finish, and inspect their outputs. A successful software build alone is insufficient.

## Using the workflow

Open the HFL portrait or landscape preset. Upload your actual photographs into both Load Image nodes; the example filenames are placeholders, not bundled media. Edit the motion paragraph in node 10. Begin with one continuous shot, modest movement and a visible face. Avoid unrelated appearance instructions. The first and last images should show the same person with compatible framing, outfit, lighting and scene.

Guides use an aspect-preserving cover crop. They do not stretch faces. Match the images' aspect ratio to the selected canvas to avoid unintended cropping. Portrait and landscape here are the native 7:4 or 4:7 canvas orientations, not exact 16:9 or 9:16 dimensions.

For a third face reference, add a Load Image node, connect IMAGE to `ref_images.ref_image_2`, and append this to the prompt:

```text
<Picture 3> shows the same subject. Use it only to reinforce facial identity, not pose, outfit, lighting or background.
```

Leave that socket unconnected to run with only two images. There is no required unused third Load Image node. Use a sharp, unfiltered photograph of the same person rather than an AI-restored face with invented detail.

Review the native video before finishing, including the face during motion. Matching two endpoint images does not guarantee that every intermediate frame preserves likeness. This graph does not conceal endpoint drift by pasting the original photographs over generated frames.

For the finish, refresh ComfyUI after generation, open HFL4K and select the native MKV in node 1. The dropdown reads ComfyUI input/output storage, so no download/re-upload is necessary. It rejects ordinary compressed video and already-upscaled high-resolution masters. The browser may not play FFV1; use the MP4 for preview and the MKV for finishing or editing.

## Default settings and tradeoffs

| Setting | Value |
| --- | --- |
| Diffusion model | Existing full H3 Ref2VA BF16 |
| Text encoder | Existing H3-specific Qwen3-VL 32B BF16, CLIP type `minimax` |
| VAEs | Native video FP16; native audio FP32 |
| Reference sizing | `max`: reference pipeline retains up to a 2048-pixel short edge, without inventing resolution in a small source [2] |
| Timeline | 124 frames at 24 fps: 5.1667 seconds |
| Sampler / scheduler | `res_multistep` / `simple` |
| Steps | 30 as a quality-first starting point, not a measured optimum |
| Guidance | BasicGuider; no extra classifier-free-guidance branch |
| Flow shifts | Video 12; audio 3 |
| Attention | Dense PyTorch SDPA; no additional sparse/Sage patch |
| Turbo / extra LoRAs | None |
| Output master | RGB rounded to 16 bits, then losslessly FFV1-encoded; original float PCM audio retained |

The current Comfy guide puts open-weight H3 around a 1 MP native canvas and recommends native generation followed by upscaling rather than claiming that a larger diffusion canvas creates more detail. It also advises against very short turbo schedules for tight reference matching. The 30-step selection here is a deliberate setting above the standard 20-step baseline, not a benchmark result [3].

Comfy's current templates favor smaller/pruned model builds for practical memory use. This patch deliberately retains your already configured full BF16 assets, rather than introducing a checkpoint change. That does **not** prove that full BF16 looks better than unquantized pruned BF16. Nor does it guarantee perfect identity. The full model files and their configured hashes remain unchanged [4, 5].

Keep the same seed, inputs and prompt when testing 40 or 50 steps. Compare rather than assume more steps improve every shot. The current reference template also notes `beta` and `normal` as scheduler alternatives for reference-heavy prompts; the shipped baseline stays `simple` to match the main guide. Treat scheduler changes as A/B tests, not a second automatic pass [6].

### Resolution and finishing

HFL4K sets a 2160-pixel short edge and caps the long edge at 3840, preserving the input ratio. For the included presets, that is approximately 2160 x 3780 portrait or 3780 x 2160 landscape. It is a 4K-class upscale, not native 4K generation, an exact 16:9 UHD crop, or new ground-truth detail.

The existing SeedVR2 7B FP16 model is retained. The finish uses nine-frame temporal batches, four-frame overlap, uniform final-batch padding, LAB color correction, tiled VAE processing, CPU offload, and zero added input/latent noise. These use the pinned maintainer interfaces; larger batches can improve context at a substantial memory cost [7, 8, 9]. Nine frames is a starting memory/temporal compromise, not a certified optimum. Try 13 or 17 only after a successful test with sufficient headroom.

The current H3 guide also describes a community H3 latent-upscale/refine route. This update does not install it or claim to benchmark it. The separate SeedVR2 route reuses your existing stack, avoids an additional high-resolution H3 denoising pass, retains the native master, and passes the original audio around the generative finish. This is an integration choice, not evidence that SeedVR2 preserves identity better in every case.

For an experimental larger delivery raster, the exposed upscaler controls can be changed to `resolution=4320` and `max_resolution=7680`. That is an approximately 8K-class upscale, not native 8K detail. Memory and identity are untested, and the extra pixels may make artifacts larger rather than improve the subject. The default remains 4K-class because your stated goal includes likeness, not only pixel dimensions.

Do not accept a finish that changes recognizable facial features. Keep the original MKV and compare beginning, middle, end and motion transitions. Turning noise controls to zero reduces one source of variation but does not make a generative upscaler identity-neutral.

## Hardware planning

The configurator now uses a quality-first H3 planning target of an H200-class GPU with 141 GB VRAM, 256 GB system RAM and a 300 GB persistent volume. NVIDIA lists 141 GB for the H200 [10]. The RAM/storage values and suitability for these exact graphs are planning estimates, not measurements. The existing catalog estimates about 141 GB for H3 plus SeedVR2 assets before cache overhead, references and outputs. RGB16 masters can be large.

An 80 GB GPU may work with offloading and constrained reference sizes, but this package does not certify full-size dual-reference or third-reference peak memory. On an out-of-memory error, first determine whether it occurs in reference encoding, generation, VAE decoding or finishing. Do not blindly increase canvas size, load more models, or switch to a low-step turbo LoRA as an identity fix.

## Validation and provenance

Inspected repository snapshot: `87144865e22a7a7555e56b62c09cb6f8a4b58090`.

New H3-only Comfy core pin: `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`.

Native H3 file Git blob: `1ad3c6a4095fdcb756d6222f08ba589452b72091`.

Retained SeedVR2 source pin: `4490bd1f482e026674543386bb2a4d176da245b9`.

Checks cover JSON graph structure, reciprocal typed links, dependency cycles, API-export parity, reference/guide routing, full-precision choices, retained audio/FPS, source-pin contracts, installer conflicts/rollback, and real FFmpeg CPU RGB16/audio round-trips. These checks do not run H3 or SeedVR2 inference. This environment did not build the Docker image, import the complete Comfy stack, download model weights, run the full original repository test suite, or measure identity and memory on a GPU. Those remain live validation steps.

Before client/commercial use, review the applicable MiniMax license. Open-weight availability is not by itself a grant of all commercial rights; the current Comfy overview includes a commercial-licensing section [3].

## Inspected developer and maintainer sources

Accessed for this update on October 19, 2026. The source pins above are authoritative for the code contract; web documentation can change later.

1. Comfy native multiframe workflow and reference/guide distinction: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe
2. Comfy native workflows and reference controls: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
3. Comfy H3 overview, native resolution, sampling and limitations: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
4. Existing full Ref2VA BF16 model metadata: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors
5. Existing H3 text-encoder metadata: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
6. Official reference template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
7. SeedVR2 maintainer implementation: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/video_upscaler.py
8. SeedVR2 DiT interface: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/dit_model_loader.py
9. SeedVR2 VAE interface: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/interfaces/vae_model_loader.py
10. NVIDIA H200 specifications: https://www.nvidia.com/en-us/data-center/h200/
11. Pinned native H3 implementation: https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy_extras/nodes_minimax_h3.py
12. MiniMax developer repository: https://github.com/MiniMax-AI/MiniMax-H3

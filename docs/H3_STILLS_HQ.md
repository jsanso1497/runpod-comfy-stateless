# H3 true single-image stills HQ

Research checked: October 20, 2026. Source candidate, not GPU-certified.

## Which workflow to open

In the H3 workspace, open **HSTILL: True single-image still: Ref2VA identity HQ**.

- **01_H3_Reference_Still_Identity_2p5MP.json** is the single-reference edit baseline. Start here when likeness matters more than pixel count.
- **02_H3_Reference_Still_Native_8MP.json** generates the still directly at approximately 8.4 MP. It is not an upscaled video frame. Compare its face to the 2.5 MP result before treating it as better.

Both load one authoritative photo, preserve its approximate aspect ratio, and share one size control between conditioning and the sampled latent. Enter the requested edit in the prompt. The default prompt asks for photographic clarity while preserving identity and the scene; it cannot guarantee exact reconstruction.

The optional **HSTXT: True single-image still: text-only FL2VA HQ** generates from a written prompt, without a source identity. It requires a separate FL2VA checkpoint. Do not substitute Ref2VA into this text-only recipe and assume it is the same configuration.

## The important current-core finding

Older Fizgig instructions recommend replacing both the normal H3 latent and the stock decoder. The current pinned ComfyUI source already contains the decoder fix.

At ComfyUI commit `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`, `comfy/ldm/minimax/vae.py` handles a T=1 latent by repeating it into `tokens_chunk_size` copies and selecting the first real decoded frame. With the default H3 parameters, this is five latent copies and pixel frame 3. That is the same grouping and frame selection as Fizgig Still Decode. The native implementation also selects the needed frame inside the tiled decode rather than retaining the complete decoded group on the output canvas.

Therefore this package uses **FizgigH3StillLatent + current native VAEDecode**, not an obsolete claim that the current stock decoder cannot decode a still. The external pack also installs its own decoder, but that decoder is not used by these graphs. Source-blob checks protect this choice during the H3 image build.

Do not open these graphs on an older, unverified core. On old versions, the native decoder may not contain this fix.

## Settings and evidence boundaries

All three still graphs use 50 steps, `er_sde`, `simple`, native H3 flow shifts 12/3, BasicGuider, denoise 1, and a fixed seed of 42. No Turbo LoRA, sparse-attention patch, face restoration, skin-texture modification or generative upscale is inserted.

The maintainer's published highest-quality example is an approximately 8 MP **FL2VA** still using a true single-frame latent, 50 steps and no Turbo. The same maintainer describes approximately 2.5 MP as useful for reference editing. These are published example settings, not an independent ranking or a controlled identity benchmark.

This package keeps your full BF16 diffusion weights and H3-specific BF16 text encoder, plus the original FP16 video VAE. The maintainer's examples use quantized components. Keeping full precision avoids introducing a new quantization choice, but it is not evidence that the result will outperform those examples.

For identity/reference stills, the package uses your existing **Ref2VA** checkpoint, consistent with ComfyUI's documented model routing for reference generation. Applying the true-still recipe to Ref2VA at 8.4 MP is an adaptation, not a benchmark-verified copy of the maintainer's FL2VA example.

## Why only one reference by default

An unresolved issue filed on October 20, 2026 reports that multiple-reference edits can switch toward the second image. The report does not establish a root cause or that every checkpoint/configuration is affected. It is a reason not to market multi-reference identity retention as verified.

The delivered reference graphs consequently load exactly one image and mention only `<Picture 1>`. They do not hide additional required Load Image nodes or include references to disconnected images. Multiple-reference experimentation is outside the delivered baseline.

Reference conditioning is not a hard pixel lock, an identity-trained LoRA, or masked compositing. Face geometry, lighting, background and pose can still drift. Select the sharpest, most representative photo available. Avoid contradictory instructions such as changing age or facial structure while asking for exact identity.

## Size control

Change **H3 Still: shared output size**. `source` reads the reference image's aspect ratio without altering its pixels. Other aspect choices intentionally request a different composition. All dimensions round to multiples of 32.

Approximate 8.4 MP outputs:

| Aspect | Width x height |
|---|---:|
| 16:9 | 3872 x 2176 |
| 9:16 | 2176 x 3872 |
| 2:3 | 2368 x 3552 |
| 3:2 | 3552 x 2368 |
| 1:1 | 2880 x 2880 |

The installed latent node accepts up to 4096 pixels per axis. This is a software limit, not a quality guarantee or a measured VRAM minimum. The helper caps the long edge at that limit. Larger requested megapixels can consequently yield a lower actual pixel count on wide or tall aspects.

There is no automatic 16 MP or 8K default. Higher native resolution can change the face, composition or detail rather than simply improve it. The evidence for 8 MP text-to-image does not prove that 8 MP is always better for identity editing.

`ref_image_size=max` remains selected. This keeps more reference pixels than `match` when the input is sufficiently large; it does not invent missing source detail. The source image is not first passed through an image generator or a face enhancer.

## A real still, not an extracted clip

The sampler's input comes exclusively from **FizgigH3StillLatent**, with one temporal latent frame and batch size 1. The native conditioning node still displays `length=5` because its schema does not accept 1; its LATENT output is deliberately unused. Do not reconnect it to the sampler.

H3 still retains a tiny auxiliary audio tensor internally because it is a joint model. The workflow does not decode or export audio. Do not replace the joint latent with an ordinary image VAE-encode output.

Keep denoise at 1 for this empty-latent, reference-conditioned path. Lowering denoise here is not a conventional image-to-image preservation-strength control.

## Exports

Each run saves an RGB16 PNG master under `output/h3_still/` and a standard 8-bit PNG copy under `output/h3_still/preview/`. There is no lossy video encode in the still path.

The master preserves the decoded pixel values to 16-bit integer precision. It does not create HDR, expand gamut, or recover detail absent from the render. Some applications, including image-loading paths, may convert a 16-bit PNG to 8-bit. Use a 16-bit-capable editor for the master.

No workflow metadata is embedded by the custom master exporter. Filenames are unique; existing masters are not overwritten.

## Required customization and downloads

The local installer adds the HSTILL/HSTXT catalog entries, matching API exports, two small Workbench helpers, tests and source guards. It installs `shootthesound/ComfyUI-Fizgig-H3-Still` at commit `10d5171f6e45b0757a03df977f53a660e060c33d` in the H3 build only. That pack requires no additional Python dependencies.

**HSTILL needs no additional model weights.** It reuses the Ref2VA BF16 model, H3 Qwen3-VL 32B BF16 encoder and FP16 video VAE already in the H3 package.

**HSTXT is optional.** It adds `minimax_h3_fl2va_bf16.safetensors`, approximately 66.3 GB, to the existing diffusion-model folder. Its download uses a pinned Comfy-Org repository revision and a provider SHA256. Selecting HSTILL or the video tasks does not download this file.

No extra single-frame VAE is installed. The independently released 500K decoder's own evaluation identifies a natural-photography/perceptual-detail tradeoff compared with the official VAE; it is not established as a better identity renderer. That comparison is not a direct benchmark against these exact workflows.

## RunPod

Rebuild **Build HQ workspace**, choosing `h3`, after manually pushing the local changes through GitHub Desktop. Use the successfully published image digest, not an assumed new tag.

Use the previous H3 authentication, ports and volume settings. Keep HTTP 8188 public and 8189 internal. For the video workflows plus reference stills:

```text
WB_WORKSPACE=h3
WB_TASKS=HFL,HFL4K,HSTILL
WB_TOOLBOX=standard
WB_DISABLE_COMFY_COMPILER=1
```

For only reference stills, `WB_TASKS=HSTILL` avoids preparing SeedVR2 unless otherwise requested. For optional text-only stills, include `HSTXT` in WB_TASKS. It adds approximately 66.3 GB, so plan extra volume and cache headroom, for example a 400 GB volume when keeping the complete set. These are planning estimates, not measured capacity requirements.

After model preparation, check inside the Pod:

```bash
python -m workbench.h3_still_contract --sources /opt/ComfyUI
python -m workbench.h3_still_contract --live
```

For optional text-only generation, also run:

```bash
python -m workbench.h3_still_contract --live --text
```

These commands only inspect sources and the live schema; they do not queue an image or certify visual quality.

## What was tested

See `reports/h3-combined-validation.json` and the two combined test logs. The offline checks cover graph structure, UI/API consistency, single-frame latent routing, shared dimensions, model selection, optional-download boundaries, PNG16 pixel round-trips, schema-error reporting and local installer safety/rollback.

The Docker image was not built here. The external node was not imported against a running ComfyUI instance here. No H3 weights were downloaded or sampled. GPU memory, latency, likelihood of identity drift and visual superiority remain unmeasured. The complete repository CI suite was not run in this environment.

## Primary sources checked

1. Current native H3 VAE implementation, including the T=1 grouping fix: https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/ldm/minimax/vae.py
2. Fizgig maintainer README and examples: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still
3. Pinned no-Turbo example: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/blob/10d5171f6e45b0757a03df977f53a660e060c33d/example_workflows/h3_still_text_to_image-8MP-NoTurboVersion.json
4. ComfyUI model routing and reference settings: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
5. Single-frame decoder author's scope and limitations: https://huggingface.co/iamkaikai/MiniMax-H3-Single-Frame-VAE-500K
6. Unresolved multiple-reference edit report: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/issues/3
7. Optional FL2VA file and provider checksum: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors

The external Fizgig code is MIT-licensed and fetched from its original repository during the build. Its original license remains in that checkout. The model weights retain their upstream license and are not redistributed in this source package.

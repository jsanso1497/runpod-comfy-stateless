# Qwen Image 2.1 Photo Edit 2.0

One full-resolution source photograph, manual or native SAM 3.1 masks, a roughly 2K working crop, up to nine additional references, optional LoRAs, protected stitching, and original/output comparison.

This is an add-on for the existing RunPod. It does not replace the source repository, existing workflows, LoRA files, ComfyUI core, or installed Torch. The separate standalone-template ZIP is an optional new image build, not an update-only patch for a snapshot-locked repository.

## Install on the current pod

Upload `Qwen21_Photo_Edit_2K_SAM3_v2_0.zip` into `/workspace` using FileBrowser, then run:

```bash
cd /workspace
unzip -o Qwen21_Photo_Edit_2K_SAM3_v2_0.zip -d /workspace
bash /workspace/qwen21_photo_edit/install.sh --models bf16 --sam
```

Existing structurally complete BF16 models are reused. Only missing or incomplete selected files are downloaded. For the lower-memory variant, use `--models int8 --sam`. To install nodes/workflows without downloading weights, use `--models none`.

Restart **ComfyUI**, then refresh the browser. Do not terminate a stateless pod just to refresh nodes. Open the `Qwen21_Photo_Edit` workflow folder. Import the ordinary `.json`, not `.api.json`.

The installer checks for native `TextEncodeQwenImage21`, `QwenImage21Cache`, and, with `--sam`, `SAM3_Detect` with refinement. It stops rather than replacing your core or Torch when these interfaces are missing. A missing-interface error requires a compatible ComfyUI rebuild/update, or the standalone template. Do not install the separate Meta SAM Python package for this workflow.

After restart, check node registration with:

```bash
python /workspace/qwen21_photo_edit/preflight.py \
  --comfy-home /workspace/ComfyUI --sam --server http://127.0.0.1:8188
```

## The two main toggles

**Node 2, `auto_mask`**: OFF uses the mask painted and saved on the SOURCE Load Image. ON loads the official ComfyUI SAM 3.1 checkpoint and detects the short object phrase. It returns a full-source-coordinate mask. OFF does not import, load, or execute SAM.

**The node titled "10 | RUN EDIT", `run_edit`**: OFF runs mask/crop preparation and shows the mask overlay, with full-photo and close-up crop views. It does not request the Qwen model, LoRAs, references, or sampler. It saves no edited photograph. ON runs the edit, stitches it into the untouched source canvas, and saves the result. The graph starts with this OFF so that an incorrect automatic selection is not committed before review.

A mask changed by SAM is not automatically a good edit mask. In particular, segmentation cannot know what area should be newly exposed when clothing is removed. Inspect the shoulder/arm transition, desired new garment outline, and protected landmarks.

## Automatic masking

Use a short object description, not an edit instruction. Examples: `shirt:1`, `shirt sleeves:2`, or `jacket:1`. The native SAM text path supports comma-separated concepts and `:N` detection caps. These are starting prompts, not a guarantee that a specific garment part will be detected correctly.

`threshold=0.5` and `refine_iterations=2` match the current native image-segmentation template. More refinement is not automatically more accurate.

`instance_index=-1` combines detected instances. A nonnegative number selects that detector result. For crowded scenes or a small clothing detail in a 7K photo, paint a rough area on the SOURCE and choose `search_region=painted ROI`; SAM searches that crop plus 128 source pixels of context and maps the result back without changing the source photo.

Manual corrections in automatic mode:

| Setting | Effect |
|---|---|
| ignore | Use SAM's mask; the saved manual mask is unused except as a search ROI when selected. |
| restrict to painted area | Intersect SAM with the saved manual mask. |
| add painted area | Add the saved manual mask to SAM's selection. |
| subtract painted area | Use painted areas as protected regions, removing them from the SAM mask. |

`expand_pixels=0` is the preservation-oriented default. Raising it intentionally expands the effective edit boundary. Restrict/subtract operations are applied after expansion, so protected areas are not grown back into the mask. There is no automatic hole filling. A missing/empty selection raises an error; it never falls back to editing the whole photo.

## Crop resolution and original photo size

`resolution=2048` is an area ceiling of **2048 x 2048 pixels**, including 32-pixel alignment padding. It is not a forced square and not a limit of 2048 pixels on both axes. `max_long_side=3072` separately limits narrow/elongated crops. This uses the model's supported approximately 4.2-million-pixel output class while accommodating portrait and landscape shapes.

Only the crop is resized. The source photo remains at the loaded dimensions. Small crops are **not enlarged** by default. Turn `upscale_small_crops` on only deliberately; interpolating a small crop does not recover original detail.

Examples, away from frame edges:

| Mask bounds | Context per side | Working crop |
|---|---|---|
| 1500 x 1500 | 128 px | 1756 x 1756 content, padded to 1760 x 1760; no resampling |
| 2500 x 2500 | 128 px | 2756 x 2756 source crop, reduced to 2048 x 2048 |

Padding is stripped before a generated crop is resized back to its original rectangle. A final 7001 x 4003 source remains 7001 x 4003. In the larger example, the generated area is a 2K generation resampled back, **not newly generated native 2500-pixel detail**.

## References and prompt

The source crop is always `<image1>`. The first reference loader is `<image2>`, the next `<image3>`, and so on. Four additional reference loaders are shown. Duplicate a reference loader and connect the remaining `reference_5` through `reference_9` sockets for up to ten total images including the source. Fill slots consecutively; gaps are rejected rather than silently changing tag numbers. `None` disables an unused loader without requiring a dummy image file.

Each reference has an independent default area ceiling of 1536 squared, plus a 3072-pixel long-side cap. Smaller references are not enlarged. Raise a useful detail reference to 2048 deliberately rather than enlarging every reference automatically. Two to four relevant, nonconflicting references are a practical starting point; nine large references may be unnecessarily expensive.

Role text is appended verbatim with the appropriate image tag. Your edit prompt is not rewritten by a language model. No prompt enhancer is installed or enabled. The reference map and effective prompt are included in the verification report.

For perspective-sensitive clothing edits, a useful role is:

```text
Reference for this person's arm shape, skin texture, and distinctive features only. Reconstruct those features in the source pose and perspective. Keep existing source shoulder, elbow, wrist, and hand positions; do not copy apparent lengths or pixel sizes from this reference.
```

Prompt example:

```text
Make the shirt in <image1> naturally sleeveless. Preserve its neckline, fabric, color, torso fit, and untargeted details. Use <image2> as the arm-appearance reference, adapted to the existing pose and camera perspective of <image1>. Keep the visible source elbow, wrist, and hand positions fixed. Reconstruct only the newly exposed shoulder and upper-arm transitions. Match the source lighting and photographic texture.
```

This guides the model; it does not mathematically lock anatomy inside the editable region. Features that must be exact should remain outside the effective mask or inside a protected hole.

## Model and sampler defaults

BF16 is the quality-oriented default. INT8 is a separately selectable lower-memory preset, never a silent fallback. Both use the matching Qwen 2.1 VAE and the `qwen_image` text-encoder loader type.

The generation path delegates to ComfyUI's `TextEncodeQwenImage21`, using `resolution=0` because every input has already been prepared and padded to 32. This avoids a second uniform resize of all reference images. The source crop determines the latent size.

The sampler is **40 steps, CFG 1, Euler, simple scheduler, denoise 1**. Seed 12345 is fixed for comparable A/B tests. There is no four-step FLUX schedule or Lightning adapter. Native Qwen prefix KV caching uses `dtype=default`, not INT8/INT4 cache quantization. Use `cache_device=cpu` when memory is tight before resorting to a quantized model. The text encoder can also be moved to CPU, at a speed cost.

All three model-only LoRA slots start at `None`. Only load adapters explicitly made for Qwen-Image-2.1. FLUX, the older 20B Qwen Edit models, and their accelerated adapters are not interchangeable. No BFS or face-restoration adapter is pre-enabled. Duplicate the LoRA node in series for more slots.

## What this inpaint workflow actually does

This is **native Qwen crop editing followed by protected compositing**. Qwen generates the whole working crop from its native empty target latent and reference conditioning. The mask controls the crop selection and final replacement. It is not passed as a dedicated trained inpainting channel and is not attached as a noise mask to an empty latent. That latter shortcut would not preserve real source context.

The full-resolution source outside the effective mask never enters the target generation path. Pixels outside the final mask are copied from the loaded source. Mask holes survive. Feathering is inward-only and defaults to 12 source pixels. Color matching defaults to zero and, if enabled, applies only to the generated patch. No whole-photo sharpening, face restoration, upscale model, or global color conversion is applied.

## Output and comparison

`run_edit=ON` saves a full-size RGB PNG, the effective mask PNG, and a verification JSON sidecar. The latter records source/output dimensions, actual working size, resampling, mask mode, reference roles, and the protected-pixel check.

The comparison node offers full-photo and edit-crop views. Preview images are reduced to a maximum 1536-pixel long side; inspect the saved PNG externally at 100% for final detail assessment. Preview size has no effect on export.

Inputs should be **8-bit sRGB** photographs. Output is lossless 8-bit sRGB PNG with workflow metadata, not a RAW/16-bit/P3/Adobe-RGB roundtrip, not the original JPEG file, and not preservation of all camera metadata. Exact preservation refers to the loaded working-image pixels outside the effective mask. Keep high-bit-depth/wide-gamut masters separately.

## Dependencies and models

No external SAM Python environment, API model, Ollama, KREA, SeedVR2, or prompt-enhancer model is needed.

BF16 files:

```text
models/diffusion_models/qwen_image_2.1_bf16.safetensors
models/text_encoders/qwen3vl_8b_bf16.safetensors
models/vae/qwen_image_2.1_vae_bf16.safetensors
models/checkpoints/sam3.1_multiplex_fp16.safetensors
```

The downloader uses the official Comfy-Org repositories, checks safetensors structural completeness, and preserves an invalid existing file with an `.invalid-*` suffix rather than deleting it. That check is not a cryptographic authenticity guarantee. HF_TOKEN is read from the environment and is not written into the workflow.

## Verification and limitations

See `VALIDATION.md`, `TEST_RESULTS.txt`, and `LARGE_CANVAS_VALIDATION.json`. CPU tests check geometry, protected pixels, mask logic, lazy execution requests, reference ordering, workflow wiring, and actual PNG export. Calls into SAM/Qwen are mocked in unit tests. GPU quality, GPU memory demand, and the real browser interface have not been tested here. No model outcome is guaranteed to perfectly reproduce identity or hidden anatomy.

This kit has not changed or published your live GitHub repository or container image. Do not merge its standalone build files into a repository guarded by the older SOURCE_SNAPSHOT manifest. Use the current-pod add-on installer, or put the separate standalone-template ZIP in a new repository.

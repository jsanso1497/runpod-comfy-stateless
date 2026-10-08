# Review of the supplied Qwen Image 2.1 workflows

## Reviewed basis

The two supplied simple Qwen JSONs were inspected directly. Their SHA-256 values are in SOURCES.json. This review is not a claim to have inspected an unprovided newer private repository or benchmarked the models on the user's photographs.

## Retained

The simple workflows already selected matching Qwen Image 2.1 diffusion, Qwen3-VL, and VAE files, the qwen_image loader type, native reference conditioning, and 40 / CFG 1 / Euler / simple / denoise 1. Those settings are consistent with the current official path. There is no evidence here that an exotic sampler or a FLUX distilled schedule would improve the requested edits.

BF16 remains the default. The INT8 variant is explicitly lower precision, not a higher-resolution model. Native cache storage remains unquantized. The prompt enhancer stays absent so the user's instruction and reference roles remain under their control.

## Changes driven by the requested use case

| Original behavior | New behavior | Reason |
|---|---|---|
| Whole-frame regeneration | Native Qwen edit of a bounded crop, then protected compositing | Preserve loaded source pixels outside the edit, not merely ask the model to preserve them. |
| Uniform resolution=2048 for every reference | 2048-squared crop ceiling; independent per-reference ceilings | Avoid enlarging all references and avoid a second resize in the native encoder. |
| Resized image 1 defines full output | Original full-resolution source defines final output | A 7K photograph need not become a 2K photograph. |
| No mask selection/review | Manual/SAM 3.1 toggle and preview-only run | Inspect automatic selection before requesting generation. |
| No LoRA controls | Three model-only slots, all None | Permit chosen Qwen 2.1 adapters without forcing an identity-altering adapter on every edit. |
| Two images only | Source plus four visible, nine possible references | Explicit ordered tags and user-written roles. |
| Random seed | Fixed seed default | Useful for comparing a single setting or reference change. |
| Save result only | Original/output comparison plus mask and verification export | Make unintended changes easier to inspect. |

## Why this SAM integration

The current official ComfyUI template uses sam3.1_multiplex_fp16.safetensors through CheckpointLoaderSimple and SAM3_Detect, with two refinement passes. The kit delegates to that native implementation instead of bringing in another SAM Python environment. Meta's SAM 3.1 release primarily documents multi-object video tracking improvements. It does not establish universal superiority on small garment parts in still photographs. The selection should be judged on the user's image.

A painted search ROI is available to improve the scale at which a small object is presented to SAM. Manual intersection, addition, and subtraction permit user corrections. The mask never silently becomes a full-image selection on failure. A preview-only run does not load the Qwen branch.

## Important implementation distinctions

This is crop/edit/stitch inpainting, not a new trained Qwen inpaint model. Native Qwen generates the complete crop from an empty target latent. The effective mask is enforced at the final composite. No unsupported mask channel is invented and no zero-filled context is falsely claimed to be preserved by a noise mask.

The 2K setting is a 2048 x 2048 area ceiling including padding, not a square crop and not a claim that 2048 is the largest possible long dimension. The model's official examples include non-square dimensions above 2048 on one side. Long side is separately capped at 3072 in this kit.

A 1536-squared default per-reference budget is an engineering compromise for multiple references on the user's A40, not a documented optimal resolution. Raise a valuable detail reference to 2048 deliberately; crop it around the information that matters. No A40 speed or VRAM benchmark was performed here.

Preservation is checked on loaded 8-bit-sRGB working pixels. It does not promise lossless RAW processing, original JPEG bytes, original camera metadata, recovery of invisible anatomy, or perfect identity inside generated regions.

## Source links

Exact primary-source URLs, the inspected ComfyUI interface commit, and local baseline hashes are recorded in SOURCES.json. Model and native node documentation supports the compatibility choices; the geometry and CPU tests support the local implementation assertions. Neither substitutes for a GPU quality test.

# Native Qwen edit and full-resolution edit overlay

Two additive Qwen tasks. Existing workflows and custom prompt-enhancer backends are not changed.

## QNATIVE: official ComfyUI edit template, BF16 image models

The generator copies the actual upstream JSON at commit `3e89967349dd95fa68ce3b36966f41d642af8ea7`, file `templates/image_qwen_image_2_1_image_edit.json`.

Only the main diffusion and base image-text encoder selections change:

- `qwen_image_2.1_bf16.safetensors`
- `qwen3vl_8b_bf16.safetensors`
- The existing `qwen_image_2.1_vae_bf16.safetensors` stays unchanged.

The stock prompt enhancer stays `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors`. Its system prompt, internal sampling settings, template flags and branch wiring are not replaced by Workbench/Ollama enhancement. This one intentional INT8 encoder exception is limited to the two new tasks; image-model precision remains BF16.

Stock controls are retained: 25 steps, CFG 1, Euler, simple, resolution 0, custom_size off, lossless/default cache, refine_prompt off and thinking off. Turn on refine_prompt to use the stock enhancer. The template's own notes describe its thinking control. Do not confuse the native template's resolution 0 with a safe size cap: it follows source dimensions rounded for the model and can be expensive for a very large input.

Upload your source and any required reference images into the stock Load Image nodes. Disconnect image_2 when no second reference is needed. This workflow preserves the official template's sample-image wiring, rather than silently changing its defaults.

## QOVERLAY: manual/SAM edit mask, independent SAM protection, transparent export

1. Load the original, full-resolution **sRGB RGB** image. For manual masking, open Mask Editor on that source, paint on the MASK layer and save. Leave auto_mask off. Alternatively enable auto_mask and describe the editable object.
2. The separate protection SAM node examines the **original**, not the generated image. Its default description is `face`. Change this to another object/region, such as `head`, `hair`, or `logo`. Index -1 protects all matching instances. A failed/empty detection stops the run. Disable protection explicitly only when none is needed.
3. Run once with **run_edit off**. The preview shows editable selection in green and protection in red. Inspect the face/element boundary. SAM segmentation is not guaranteed to find every detail.
4. Enter the edit instruction in the native Qwen subgraph and set **run_edit on**. Its prompt enhancer and sampling defaults are identical to QNATIVE. Extra image references can connect to image_2 onward.
5. The edit crop is bounded to 2048 squared pixels, with a 3072-pixel maximum long side, 128 pixels of source context and no automatic enlargement of small crops. The full-resolution original is retained separately. Keep the native subgraph at **resolution 0 / custom_size off**; changing its output dimensions deliberately fails the stitch rather than silently misaligning it.
6. Export produces `*_overlay.png`, `*_composite.png`, and `*_mask.png`. All use the original image dimensions. The overlay is straight-alpha RGBA, not a crop and not a premultiplied image. Place it at x=0, y=0 above the original with Normal blending and 100% opacity.

The final alpha is the edit selection minus the grown protection region, feathered inward. Protection wins after all mask processing. Unchanged pixels are removed from alpha after 8-bit quantization. The overlay contains unblended edit RGB so its edges are not blended twice when placed over the source. PNG alpha is quantized before the included composite is calculated.

Original pixels outside the exported alpha and within protection are copied exactly in the 8-bit sRGB output. This is not preservation of JPEG file bytes, original EXIF metadata, a non-sRGB profile, or 16-bit/HDR precision. Color-managed editors may blend differently depending on their working-space/linear-blending settings. Processing a bounded crop preserves untouched high-resolution source detail; it does not invent native full-resolution detail inside an enlarged generated crop.

## Build and availability

`python scripts/build_site.py` now generates both workflows from the pinned source before generating the catalog. Docker and Pages already call this script. The upstream snapshot and a SHA256 provenance report are saved to `vendor/qwen-stock/`. Subsequent builds reuse that snapshot; `PYTHONPATH=src python -m workbench.qwen_stock_workflows --offline` refuses to fetch a missing snapshot.

Generated workflow paths:

- `workflows/qwen/QNATIVE/Qwen21_Comfy_Template_BF16.json`
- `workflows/qwen/QOVERLAY/Qwen21_Full_Resolution_Overlay_BF16.json`

These JSON files are generated build artifacts, like the existing developer recipes, and CI exports them as `qwen-stock-and-full-resolution-overlay`. The source catalog extension registers QNATIVE and QOVERLAY in the Qwen workspace. Preparing either task installs the stock PE encoder group; the overlay task also requests SAM 3.1. Model weights are not included in the repository or downloaded during image builds. Existing Qwen workspace defaults are not changed, including optional legacy helper installation choices.

A rebuilt Qwen image installs the new `Workbench_qwen_overlay` node package. Merely importing the JSON into an older running Pod does not install those new nodes. Merging a PR does not itself update an already running Pod or publish Pages; use the repository's normal build/deploy steps.

## Validation

The added CPU tests cover pixel preservation, protected holes, padding removal, straight-alpha recomposition, no-op edits, generated transparency, invalid geometry, failed SAM handling, lazy mask review, native-default preservation, graph wiring, catalog isolation and deterministic generation. Where the pinned snapshot exists, the adapter tests use that actual template. Without network/snapshot, a small explicit graph fixture tests the serializer contract instead.

No live GPU generation, SAM accuracy, model download, ComfyUI frontend interaction or Docker build is certified by these tests. Mask preview and a live end-to-end run remain required before production use.

Sources: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1 and https://github.com/Comfy-Org/workflow_templates/tree/3e89967349dd95fa68ce3b36966f41d642af8ea7

# Native Qwen edit and full-resolution watermark removal

Two Qwen tasks. QNATIVE stays a stock-template baseline; QOVERLAY now defaults to watermark removal with full-photo output. Other workflows, shared node defaults, model weights, and custom prompt-enhancer backends are not changed.

## QNATIVE: official ComfyUI edit template, BF16 image models

The generator copies the actual upstream JSON at commit `3e89967349dd95fa68ce3b36966f41d642af8ea7`, file `templates/image_qwen_image_2_1_image_edit.json`.

Only the main diffusion and base image-text encoder selections change:

- `qwen_image_2.1_bf16.safetensors`
- `qwen3vl_8b_bf16.safetensors`
- The existing `qwen_image_2.1_vae_bf16.safetensors` stays unchanged.

The stock prompt enhancer stays `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors`. Its system prompt, internal sampling settings, template flags and branch wiring are not replaced by Workbench/Ollama enhancement. This one intentional INT8 encoder exception is limited to the two new tasks; image-model precision remains BF16.

Stock controls are retained: 25 steps, CFG 1, Euler, simple, resolution 0, custom_size off, lossless/default cache, refine_prompt off and thinking off. Turn on refine_prompt to use the stock enhancer. The template's own notes describe its thinking control. Do not confuse the native template's resolution 0 with a safe size cap: it follows source dimensions rounded for the model and can be expensive for a very large input.

Upload your source and any required reference images into the stock Load Image nodes. Disconnect image_2 when no second reference is needed. This workflow preserves the official template's sample-image wiring, rather than silently changing its defaults.

## QOVERLAY v2: watermark removal and the full original photograph

Defaults: **auto_mask ON**, **sam_prompt = watermark**, **run_edit ON**. The default edit prompt asks Qwen to remove the watermark and reconstruct an opaque photograph, not a cutout or isolated watermark. These are workflow defaults only; dragging a new shared photo node into another graph keeps that node's original defaults.

Load the full original sRGB photograph and queue the workflow. SAM selects all matching watermark instances (`instance_index=-1`). It can miss letters or select too much. To inspect the selection without Qwen inference, turn `run_edit` OFF. For manual masking, paint and save a source mask and set `auto_mask` OFF.

### Mask and image paths

```text
Edit selection -> GrowMask(+16 source pixels) -> InvertMask -> protected area
Edit selection minus protected area -> ONE Q21PhotoCrop
                                      |-> working photograph -> native Qwen edit
                                      |-> source/crop package -> Q21PhotoStitch
Native Qwen result -------------------------------------------> Q21PhotoStitch
Full-photo stitched result ----------------------------------> Q21PhotoReviewSave
```

The final mask is `clamp(edit_mask - (1 - grown_edit_mask), 0, 1)`. For binary SAM/manual masks this keeps the exact selected edit area. Soft manual edges are conservatively restricted. Expansion is applied before inversion. There is no second SAM detection, no protection growth after inversion, and no unused protection branch. The 16-pixel margin does not enlarge the actual paste selection.

One crop package is shared by image generation, stitching, and saving. This removes the duplicated crop path in the earlier live JSON. The processing budget remains 2048 squared pixels, maximum long side 3072, context 128 source pixels, and no upscaling of small crops. Keep native `resolution=0` and `custom_size` OFF so generated geometry stays aligned with that crop.

The **only image-output node is Q21PhotoReviewSave**. Its primary saved/previewed image is a complete, original-size, **opaque RGB PNG**. The generated crop is blended into the original through the final mask, with 12-pixel inward feathering. RGBA generation is flattened against the source crop before stitching, so a transparent generated patch does not turn the complete output into a transparent layer.

Primary output prefix: `Qwen21_Watermark_Removal/FINAL_Full_Photo`. The existing saver also writes a diagnostic `.mask.png` and `.verification.json` alongside it. Those are sidecars, not the displayed result. There is **no transparent-overlay exporter** in this workflow. Legacy overlay node classes remain installed for previously saved workflows.

Pixels outside the final mask are copied from the original. This is 8-bit sRGB pixel preservation, not JPEG byte preservation, EXIF retention, or 16-bit/HDR export. Reconstructing obscured detail is not exact recovery. Correct file compositing does not establish that the image model successfully removed a particular watermark.

## Build and availability

`python scripts/build_site.py` now generates both workflows from the pinned source before generating the catalog. Docker and Pages already call this script. The upstream snapshot and a SHA256 provenance report are saved to `vendor/qwen-stock/`. Subsequent builds reuse that snapshot; `PYTHONPATH=src python -m workbench.qwen_stock_workflows --offline` refuses to fetch a missing snapshot.

Generated workflow paths:

- `workflows/qwen/QNATIVE/Qwen21_Comfy_Template_BF16.json`
- `workflows/qwen/QOVERLAY/Qwen21_Full_Resolution_Overlay_BF16.json`

These JSON files are generated build artifacts, like the existing developer recipes, and CI exports them as `qwen-stock-and-full-resolution-overlay`. The source catalog extension registers QNATIVE and QOVERLAY in the Qwen workspace. Preparing either task installs the stock PE encoder group; QOVERLAY also requests SAM 3.1. The pinned stock PE model and its narrow INT8 download exception are unchanged. Model weights are not included in the repository or downloaded during image builds. Existing Qwen workspace defaults are not changed, including optional legacy helper installation choices.

Rebuild the Qwen image after applying the source update. The new standalone JSON uses the existing Qwen photo nodes and ComfyUI core mask nodes, not a new plugin. The old `Workbench_qwen_overlay` classes remain installed only for compatibility. A source commit does not update an already running Pod or publish Pages. After rebuilding, open QOVERLAY from the Workbench sidebar, which serves the new factory graph directly. Existing saved graphs and copy-once Workbench Factory files on persistent storage are not overwritten.

## Validation

Legacy overlay CPU tests remain. New full-photo tests exercise the actual Q21PhotoCrop, Q21PhotoStitch, and Q21PhotoReviewSave code, including PNG save/read-back, opaque output, exact source dimensions, protected pixels, RGB/RGBA generated crops, resampling, image-edge masks, holes, soft masks, empty masks, invalid geometry, manual mode, SAM routing, and the review switch. Additional graph checks enforce SAM/edit defaults, a watermark prompt in both widget serializations, one shared crop, the connected inverted protection mask, and the absence of a transparent-overlay exporter. Where the pinned snapshot exists, the adapter tests use that actual template. Without network/snapshot, a small explicit graph fixture tests the serializer contract instead.

No live GPU generation, SAM accuracy, model download, ComfyUI frontend interaction or Docker build is certified by these tests. SAM/model calls are mocked in CPU tests. Verify the selection and complete an end-to-end run on representative photos before production use. Run the full repository CI and Docker build after applying this update.

Sources: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1 and https://github.com/Comfy-Org/workflow_templates/tree/3e89967349dd95fa68ce3b36966f41d642af8ea7

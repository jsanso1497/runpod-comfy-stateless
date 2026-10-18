# MOSS text-to-image and image-to-image in the Qwen workspace

This package adds two independent MOSS tasks alongside QKEEP1 and the existing
Qwen tasks. MOSS is not registered as a Qwen checkpoint. Keep WB_WORKSPACE=qwen.

## Tasks

- QMOSS: the pinned ComfyUI MOSS-Image SFT text-to-image template, copied byte for byte.
- QMOSS_I2I: a Workbench standard latent image-to-image adaptation of that template.

QMOSS_I2I is not an official MOSS instruction-edit workflow or a dedicated
reference-conditioned editing checkpoint. It encodes the source image into a
latent, adds partial noise, and resamples with a caption. Faces, text, lighting
and geometry can change. For exact protected regions, use the separate Qwen
workflows, not this basic whole-image MOSS test.

## Image-to-image usage

1. Open QMOSS_I2I and upload a complete image in SOURCE IMAGE.
2. Replace the positive prompt with a description of the complete desired image:
   subject, setting, medium, lighting and style. Do not rely on an instruction
   such as "make this better" to supply the scene's contents.
3. Queue. The starting denoise is 0.35. Lower denoise usually retains more of the
   source; higher denoise permits more change. This is a suggested trial setting,
   not an official or measured optimal MOSS default. At 1.0 the source is no
   longer a useful image constraint. Even 0.0 includes VAE encoding/decoding, so
   it is not a promise of exact source-pixel preservation.
4. Save Image writes the complete generated frame under MOSS_Image/I2I.

The input path is:

    LoadImage -> ImageScaleToTotalPixels -> VAEEncode -> KSampler -> VAEDecode -> SaveImage

Processing starts at 1 MP using the source aspect ratio, rounded to dimensions
that are multiples of 16. ComfyUI uses 1024*1024 pixels for this 1 MP setting. It
can upscale a small source or downscale a large source. This is whole-image
processing, not crop-and-stitch, not a protected overlay, and not automatically
restored to the source's original dimensions. Change the megapixel setting to
change the working/output size. There is no forced square canvas.

## Preserved MOSS settings

Both workflows use the same exact files:

- diffusion_models/moss_image_sft_int8_convrot.safetensors
- text_encoders/qwen3.5_9b_w4a8_gptq.safetensors
- vae/flux2-vae_small.safetensors

The image-to-image adaptation retains the T2I template's 25 steps, CFG 2.5,
er_sde sampler, simple scheduler, disabled SLG switch, and enabled prompt
upsampler. It replaces the empty latent with the encoded input and changes
sampling denoise from 1.0 to 0.35. Its example caption and output folder are
specific to image-to-image. Controls are shown as ordinary editable native
nodes, not a hidden subgraph.

The stock prompt upsampler's system prompt, text-generation settings and model
are preserved. It receives TEXT ONLY, not the source photograph. Its switch can
be turned off to use the exact caption and reduce invented details. No custom
Workbench/Ollama prompt enhancer is added to either MOSS graph.

No extra model file, secret, or custom node is needed for QMOSS_I2I beyond the
preceding QMOSS package. Both tasks share the same moss_image asset group.
The three MOSS files total 13,006,811,386 bytes according to the existing pinned
manifest. Qwen's BF16 weights, private libraries and model-family checks remain
separate. Stock quantization exceptions are limited to the existing two
hash-pinned MOSS files and to these two MOSS task IDs in static graph checks.

## Installation and building

The combined updater supports the reviewed base repository, QKEEP1 already
installed, or the earlier QKEEP1 + QMOSS package already installed. Only exact
recognized previous addon versions can be replaced; unknown local changes stop
preflight before writing anything. Backups stay under local Git metadata.

Review and publish using GitHub Desktop, then run Build HQ workspace > qwen.
The Qwen image keeps the same MOSS-capable ComfyUI pin introduced in the previous
combined package: bab80ef793ac42ad3f25589a6b0507ca537b1dd1. Other workspace pins,
Docker entrypoint, public port and secret bindings are unchanged.

Append QMOSS_I2I to WB_TASKS to register its startup selection; retain QKEEP1,
QMOSS and all other existing selections. The workflow catalog makes all three
available regardless of startup selection. Both MOSS tasks request the same
asset group, so adding I2I does not double model downloads.

On an already MOSS-ready Pod, the included MOSS_Image_I2I_ComfyUI.json can be
imported directly without a restart or installing new custom nodes. Older Qwen
Pods that lack native MOSS support still need the rebuilt image. The included
QKEEP1 live-Pod kit remains QKEEP1-only and does not upgrade ComfyUI for MOSS.

## Source and generated files

The stock source is verified by Git blob ID before the build reads its runtime
nodes. QMOSS remains an exact copy. The second workflow is derived separately
and its provenance explicitly labels it as a non-official img2img adaptation.

- workflows/experimental/moss/QMOSS/MOSS_Image_Stock.json
- workflows/experimental/moss/QMOSS_I2I/MOSS_Image_I2I.json
- vendor/moss-stock/provenance.json

The bundled I2I JSON uses the runtime node fields and connections inspected in
the pinned template. The full official source is still downloaded and verified
at build time. Isolated tests use an explicitly labeled minimal source-contract
fixture; it is never accepted in place of the real source blob in production.
Repository CI compares regeneration against the actual verified snapshot.

## Verification limits

Graph/serialization checks, installer upgrade tests and model-manifest tests
are local checks. They do not prove successful Docker dependency installation,
ComfyUI frontend interaction, model loading, GPU inference or image quality.
Retest an existing Qwen workflow after moving from the old pre-MOSS image.
See the package verification report for executed and deferred checks.

## Sources

- https://github.com/Comfy-Org/workflow_templates/blob/7d8ed32f0bf6b8d52129976535422d6f518fe463/templates/image_moss_image_t2i.json
- https://github.com/Comfy-Org/ComfyUI/blob/bab80ef793ac42ad3f25589a6b0507ca537b1dd1/comfy_extras/nodes_post_processing.py
- https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
- https://docs.comfy.org/tutorials/basic/image-to-image

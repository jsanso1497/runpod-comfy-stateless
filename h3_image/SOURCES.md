# Sources and provenance

## Scope

This release updates the earlier 2026-11-10.1 generated native workspace. It does not modify `h3_still/`, Lab, Eros, Qwen, or existing images. It wraps native model loading, conditioning, Euler sampling and image decoding without changing their algorithms. Runtime/access code is retained from the earlier local package, with isolated backend/supervisor locks and queue-safe controls.

## Source inputs

- User-supplied `README(1).md`, retained as `MODEL_CARD_SUPPLIED.md`.
- User-supplied `pdd_config.json`, retained verbatim. Six evaluations and seven sigma endpoints.
- User-supplied `model_index.json`, retained as `model_index.reference.json`. Image pipeline, not P-H3 or video.

## Primary documentation verified November 10, 2026

- https://huggingface.co/Comfy-Org/MiniMax-H3-Image
  Lists the pruned INT8 model, H3 NVFP4/AWQ text/vision encoder and native image FP16 VAE.
- https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models
  Lists approximately 21 GB for the INT8 diffusion model. Full file sizes, revisions and SHA-256 values are resolved and exported by the build, not invented here.
- https://huggingface.co/Comfy-Org/MiniMax-H3
  Documents that the NVFP4/AWQ encoder does not require Blackwell.
- https://docs.comfy.org/manager/install
  Install the pinned core's manager requirements and launch with --enable-manager.
- https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE
  Standard grant excludes US/UK/EU/South Korea; see additional authorization provisions. The package does not grant rights.

## Exact ComfyUI integration

ComfyUI commit `ede4e5fe8d15445c6fcd6828d70b202c22f9502c` contains native MiniMaxH3ImageConditioning, MiniMaxH3ImageModelSampling and image VAE handling. Its version file still identifies 0.44.0 development code; this is NOT a claim that the older 0.44.0 release tag has the same files. Integration checks assert actual node contracts and the exact sampling grid before publication. The build does not automatically update to current master.

The prior research mentioned a 0.45.0 release requirement. The accessible GitHub latest-release metadata still pointed to 0.44.0 during preparation. We retain the verified newer source commit instead of inventing a 0.45.0 release SHA.

Official template reference: workflow_templates commit `c8e48bdcfc7241d7af588d20d4f924caf9d9d953`, `templates/image_minimax_h3_image_t2i.json`. Enhancer branches are deliberately absent from these manually authored workflows.

## Measurements and limitations

Reference sizes in reports call the native resize function/constants. Timings wrap the native sampling and decode calls. Unit tests use fake model/encoder objects and small tensors where applicable; they do not establish numerical equivalence for real weights. GPU generation and licensed self-host deployment must be validated separately.

## INT8 encoder default update: 2026-11-11

The current source manifest supersedes the earlier NVFP4 default. Only the required text/vision encoder changes.
- Model: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
- Raw pointer: https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
- Reviewed byte size: 27141342152.
- Reviewed SHA-256: bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923.
- Build resolution checks the reviewed fingerprint, records the immutable repository revision, and the existing Pod downloader verifies the full downloaded file.
- Native image model-card lists both encoder formats: https://huggingface.co/Comfy-Org/MiniMax-H3-Image
- No quality, speed or VRAM equivalence is implied. No model weights are included in this update.

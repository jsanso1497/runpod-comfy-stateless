# Source review and provenance

Reviewed November 3, 2026. Sources below are the model/integration developers, the user's repository, FFmpeg and RunPod. No universal model ranking is inferred from these sources.

## User repository, read only

- Current main at review: https://github.com/jsanso1497/runpod-comfy-stateless/commit/250a33cf28b3ce36b00e496684a7dd819eaf53b7
- Model and source catalog: https://github.com/jsanso1497/runpod-comfy-stateless/blob/250a33cf28b3ce36b00e496684a7dd819eaf53b7/catalog/models.json
- Existing video build: https://github.com/jsanso1497/runpod-comfy-stateless/blob/250a33cf28b3ce36b00e496684a7dd819eaf53b7/.github/workflows/build-qwen-video.yml
- Existing isolated video supervisor and media contracts: `src/workbench/seedvr_job.py`, `video_media.py`, `seedvr_worker.py`, `src/nodes/seedvr_stream/__init__.py` at the reviewed repository commit.
- The immediately preceding commit 1421e14721346299bae5728a885ec96882a5a9ad contains the audited Qwen + video implementation. The subsequent commit adds a separate MOSS workspace. The new package is additive and does not overwrite either.

## Model choice and limitations

- Original authors: https://github.com/ByteDance-Seed/SeedVR
  - SeedVR2 is a generative restoration model, not evidence recovery.
  - Authors warn of excessive detail in lightly degraded content and failures with severe degradation/large motion.
- Integration maintainer: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
  - FP16 and LAB are recommended for fidelity.
  - Temporal batch follows 4n+1, at least 5 for video temporal consistency.
  - More context is recommended when memory permits; tiling/offload are documented memory controls.
  - Sharp variants specifically enhance detail; this package selects standard instead for its conservative baseline.
- The 0.50 source blend and small default preview are package design decisions, not developer-prescribed magic settings or proven optima.
- A recent alternative was checked: https://github.com/OpenImagingLab/FlashVSR_Plus . No second engine was included because this request prioritizes a minimal, source-faithful workflow. There is no claim that one model is universally best.

## Exact source contracts reused

- SeedVR2 CLI commit: 4490bd1f482e026674543386bb2a4d176da245b9.
  https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/inference_cli.py
- CLI Git blob: 2d4fff18667e12a7d35ec8a43b952820a6b73ee9. The Docker smoke gate refuses an unexpected blob.
- `_process_frames_core` is reused for model inference, with all namespace fields supplied. Its default streaming wrapper is not reused; the outer stream is frame-indexed and retains generated overlap.
- ComfyUI: 52f98af2e2e42c421070a3e147c161c47cdeaf22.
  https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy_extras/nodes_video.py
  https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy_api/latest/_input_impl/video_types.py
- These are the relevant reviewed repository pins, not a request to update existing workspaces to an unreviewed latest release.

## Exact weights

- https://huggingface.co/numz/SeedVR2_comfyUI/blob/fc8e71533b1b2a07f149b9b090c72a39e05f64ec/seedvr2_ema_7b_fp16.safetensors
  SHA-256: 7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a
- https://huggingface.co/numz/SeedVR2_comfyUI/blob/0de17b6c1cf41a234f525775f77df24b992c4246/ema_vae_fp16.safetensors
  SHA-256: 20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1
- Published file hashes were checked against the provider's file pages. Weights were not downloaded by the assistant. Runtime performs full verification.

## Media and storage

- https://ffmpeg.org/ffmpeg-filters.html#scdet
  - The developer documents 8-14 as useful scene-change thresholds; this implementation uses 10.
- https://ffmpeg.org/ffmpeg-filters.html
  - trim, atrim, setpts, scale, setsar and color-range/matrix handling.
- https://ffmpeg.org/ffmpeg-codecs.html
  - FFV1 preparation, libx264 delivery and audio encoding.
- https://docs.runpod.io/pods/storage/types
  - Persistent volumes versus container disk and transfer/redeployment considerations.

## Test boundaries

No source video was supplied for this new build request, only a JPEG crop. No generated enhanced preview is presented. Local CPU/media tests and mock inference cannot establish perceptual sharpness, GPU speed, memory fit, or absence of hallucinations. The optional 2x path preserves exact integer canvas dimensions but can still produce generative content drift.

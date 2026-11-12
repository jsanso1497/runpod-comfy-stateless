# Validation record

Package: 2026-11-11.1
Target repository revision inspected: ca5ce71174f7676a4fc67e1606d87222c54cc60d
Pinned ComfyUI revision: a592c0a654af22f75d97f214d6e3e34eb13cd9db

## Run during preparation

- 24 local unit/integration checks passed in a CPU-only Python environment.
- Actual FFmpeg/ffprobe synthetic-video decoding, source-resolution keyframes, original-file crops, frame-grid alignment and path checks passed.
- Model-family recipes, precision filename routing, eight/nine-image reference budgets, prompt tag translation and mode validation passed.
- Ten GUI graphs and their API counterparts passed node/link, graph ordering and shared-selector checks.
- Exact zero-mask original-pixel preservation passed on synthetic image tensors.
- Python files compiled successfully.
- The installer was checked separately for a clean additive install, repeat/idempotent install, conflict refusal, rollback and preservation of unrelated files. See the delivered validation JSON for results.

## Not run in this environment

No CUDA device was available. The repository could be read through the GitHub connector, but the execution container could not fetch external source repositories. No Docker image, GitHub Action, native assembled ComfyUI run, browser regression, full weight download or real-model generation was performed here. No GPU memory ceiling or identity accuracy result is claimed.

## Gates configured in the manual GitHub Action

The action must successfully build the isolated image, validate pinned model-file metadata, run actual native ComfyUI schemas and three CPU preprocessing graphs, check H3/P-H3/Fizgig API contracts with synthetic encoder/VAE fixtures, run retained Lab pixel-preservation checks, and run the inherited HTTPS/login/Files/Jupyter browser regression. Only then does it publish the new image tag and actual digest. CPU fixture tests are not equivalent to real-model inference.

One optional asset, P-H3's BF16 diffusion model, has an immutable revision but its pre-recorded LFS checksum was unavailable during preparation because the file page could not be opened. The action verifies that asset's publisher metadata before publication. Runtime independently resolves the SHA-256 from the same pinned revision before downloading and verifies the downloaded bytes. All other new assets have publisher SHA-256 values already included in the lock.

## Post-deployment verification

Run `python3 /opt/h3-still/video_studio/gpu_smoke.py` after model initialization. It submits three low-resolution generations using a synthetic reference video and writes completed-run metadata under `output/H3_Video_Studio/gpu-smoke-summary.json`. It is not a likeness benchmark.

Then test your actual reference video and photographs at 2.5 MP with LoRAs off. Evaluate one engine at a time, then full precision, then higher resolution, then the desired LoRA. An 80 GB card does not establish a universal safe 8 MP setting across arbitrary frame counts, reference counts, precisions and adapters.

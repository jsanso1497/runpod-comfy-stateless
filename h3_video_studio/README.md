# H3 Video Still Studio

Read `../H3_VIDEO_STUDIO_START_HERE.md` for setup and workflow use, `RUNPOD_SETTINGS.md` for the template, and `LICENSE_NOTES.md` before deployment.

This additive workspace keeps your existing Studio foundation and uses native ComfyUI H3/P-H3 conditioning, Fizgig's single-frame latent/decode for video-model stills, and native model-only bypass LoRA application. It creates a new image tag, not an in-place update to a running Pod.

No prompt enhancer, audio model, Spark, sparse-attention accelerator or automatic upscaler is added to the unified workflow. Older Lab tools remain available separately.

Files:
- `nodepack/`: settings, video decoding, reference mapping, single-backend generation and audit UI.
- `manifest.json`: immutable core/model revisions and checksums.
- `build_workflows.py`, `workflows/`, `api_workflows/`: ten matching presets.
- `manage.py`: verified downloads and metadata-only verification.
- `launch.py`, `backend.py`, `assemble.py`, `Dockerfile`: independent deployment on the retained authenticated gateway.
- `tests/`, `cpu_smoke.py`, `native_smoke.py`, `gpu_smoke.py`: checks with clearly separated CPU/GPU scope.

The package has not been built or GPU-tested in the preparation environment. See `VALIDATION.md`.

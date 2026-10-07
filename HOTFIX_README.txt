# Qwen 2.1 Identity Kit v1.0.2

Build reliability hotfix.

- Removes the full ComfyUI CPU server smoke test from the Docker build gate. Docker BuildKit has no GPU and is not a representative environment for GPU-only/custom-node registration.
- Keeps compile checks and static workflow/API graph validation in the Docker build.
- Retains the v1.0.1 SeedVR2 CPU-schema fallback for the optional upscale profile.
- `scripts/build_smoke.sh` remains available as an optional diagnostic and is no longer invoked by the Dockerfile.
- No model list, model precision, workflow quality setting, or RunPod runtime sampling setting was changed.

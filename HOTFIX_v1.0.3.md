# Qwen 2.1 Identity Kit v1.0.3 hotfix

## What failed

The GitHub Actions `validate` job used a clean Python 3.12 runner and executed the entire test suite. `test_helpers_and_graphs.py` imports PyTorch because it tests the actual ComfyUI helper nodes, but the validation runner intentionally does not install the multi-gigabyte PyTorch/CUDA runtime. Test collection therefore stopped with `ModuleNotFoundError: No module named 'torch'` before either Docker image could build.

## Fix

Validation is now split by environment:

1. **GitHub Actions host** performs compilation, all seven workflow/API graph checks, shell syntax checks, downloader tests, and SeedVR2 patch tests. It does not install PyTorch.
2. **Docker build** runs the complete test suite, including all Torch-dependent helper-node tests, using the PyTorch already supplied by the production CUDA base image. A temporary `venv --system-site-packages` installs only pytest for the test run and is deleted afterward.

This avoids downloading PyTorch twice while preserving the full test coverage on the same Python/Torch environment that ships in the image.

## Files changed

- `.github/workflows/build-qwen21-identity.yml`
- `qwen21_identity/Dockerfile`

No models, prompts, workflows, sampler settings, LoRAs, crop/stitch settings, or runtime generation settings changed.

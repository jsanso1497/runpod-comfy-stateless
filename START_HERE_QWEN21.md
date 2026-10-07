# Qwen 2.1 Identity Kit v1.0.0

Prepared October 7, 2026 for a separate Runpod build in `jsanso1497/runpod-comfy-stateless`.

## What this package is

An upload-ready repository overlay, not an already-published GitHub release or Docker image. It contains a dedicated Docker build, GitHub Actions workflow, Runpod template API payloads, seven editable ComfyUI workflows, matching API graphs, exact model allowlists and source-image prompts.

Copy **`qwen21_identity/`** and **`.github/workflows/build-qwen21-identity.yml`** into your existing repository. Keep its existing root Dockerfile and other workflows. The new action does not publish or replace `latest`, KREA, Flux, H3 or other tags.

Read `qwen21_identity/README.md` for setup and the model profiles. Start with `01_Body_Swap_Masked`, inspect the result, then use `02_Head_Refinement_Masked`. A combined three-image workflow is also included.

## Delivery status

Locally completed: Python compilation, shell syntax checks, all seven UI/API graph pairs, model manifest/profile checks, and 31 offline unit tests covering masks, reference encoding delegation, protected-pixel auditing, handoff files, download resume and checksums.

Not performed here: Docker build, real ComfyUI import/registration check, model downloads, GPU image generation, image-quality comparison, GitHub commit/push, GHCR publishing, or Runpod deployment. The Docker build includes a CPU-only real-node schema check before its image can be published. This is a build safeguard, not a completed GPU test.

A GitHub read attempt to the existing repository returned 404. These files were prepared separately and did not change that repository.

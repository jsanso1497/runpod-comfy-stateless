# Validation record

Date: October 7, 2026.

## Completed in the preparation environment

- 31 offline unit tests passed in 2.20 seconds.
- All seven ComfyUI UI JSON files parsed and matched their corresponding API graphs.
- Graph checks passed: node IDs, links in both directions, socket type consistency, cycle detection, native-latent connections, separate adapter branches and sampling baseline.
- Both download profile dry runs passed: exactly five default model files and exactly seven in the optional upscale profile.
- Every model manifest hash is a 64-character SHA-256 value. Hash values were copied from the publishers' Hugging Face file pages; actual multi-gigabyte weight bytes were not downloaded here.
- Downloader tests passed: fresh transfer, correct range resume, server ignoring Range, invalid range rejection, content hash failure, pre-existing bad-file preservation, complete partial promotion, existing valid-file reuse, auth failure and unsafe-path rejection.
- CPU tensor tests passed: empty/wrong-sized/invalid masks, combined scene RGB alignment, alpha compositing, protected-pixel audit pass/fail, mask union, /32 dimension calculations, native encoder delegation and atomic PNG handoff.
- Python compilation passed. Bash syntax checks passed.
- Upstream node definitions and exact source commits were inspected. The custom encoder calls the native ComfyUI implementation rather than reimplementing Qwen conditioning.

## v1.0.1 build fix

The first upscale Docker build exposed an upstream SeedVR2 registration assumption: its DiT and VAE loader schemas call `get_device_list()` and index the first element even on a GPU-less Docker builder. The kit now patches only that empty-list case to expose `cpu` during schema registration. The patch is idempotent and fails closed if the pinned upstream source shape changes. Runtime on a CUDA-equipped Runpod is unaffected because the upstream device list is non-empty.

## Included automated checks not executed here

The Docker build runs the actual pinned ComfyUI server on CPU, loads installed custom nodes, and verifies real `/object_info` schemas against every selected workflow. It fails the image build on missing nodes, incompatible connections or saved widget-order mismatches. The lightweight GitHub validation job runs static checks and the downloader tests.

These future checks have been written, not run in this environment.

## Not verified here

- Docker image build or startup in Runpod.
- Full live ComfyUI registration or frontend visual behavior.
- CUDA/GPU sampling or VRAM/RAM fit on an A40 or other card.
- Model download availability from the user's Pod or registry access settings.
- Actual body/face likeness, realism, prompt adherence, frame preservation in a generated example, or an upscale quality comparison.
- Github commit/push, GHCR publishing or Runpod template creation/deployment.

No claim of production validation or superior measured image quality is made. The first generated body and head on the deployed Pod should be inspected before proceeding to larger crops or optional upscaling.

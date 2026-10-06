# Krea Identity 1.0.1 validation boundary

Prepared from the attached repository ZIP, the prior Krea 1.0 update, and
`logs_101498979872.zip`. No live GitHub repository was modified.

## Failure observed in the uploaded build log

The real CPU ComfyUI smoke test launched successfully, but SeedVR2 failed to
register: `list index out of range`. The next fatal validation error was
`Missing installed node: SeedVR2LoadDiTModel`. Krea2Edit, Rebalance-Pack and the
local Krea helper pack were reported loaded. Disk diagnostics showed 8.3G
available at exit; the fatal error was not a disk-full error.

The upstream DiT and VAE schema definitions use `devices[0]` after a GPU-only
`get_device_list()` call. The installer now changes that assignment to
`get_device_list() or ["cpu"]` in the bundled copies of those two files. This
permits CPU-only schema inspection, not CPU upscaling. Nonempty GPU lists and
inference methods are untouched. Unrecognized source layouts fail explicitly.

## Executed locally after the fix

- 573 Python unit tests: 230 general, 171 portrait, 69 shared LoRAs, 19 file
  manager, 62 Krea and 22 source-snapshot tests.
- The 62 Krea tests include the prior 47 and 15 new CPU-schema regression tests.
  The new tests reproduce the original empty-list failure in minimal schema
  fixtures, then verify the fallback, unchanged CUDA/MPS device lists,
  idempotence, source-drift detection and installer ordering. Fixtures do not
  stand in for a real ComfyUI installation.
- 23 static workflow graphs, 2 frontend serialization checks, 3 repository
  preflights, Python 3.10-compatible syntax, JSON, shell, Dockerfile structure
  and all 3 Actions definitions.
- Source-snapshot verification, archive integrity, and exact reconstruction
  from both the original attached repository and the previous Krea update.
- All workflow files, model manifests, custom-node pins, sampler settings,
  rebalancer settings and inference helper code are byte-identical to 1.0.

## Required live build check, retained without a bypass

`python /opt/krea-identity/check.py --build-smoke --comfy-home /opt/comfy-bundle`
starts real CPU ComfyUI and validates every shipped Krea graph against its
actual node registry, including SeedVR2. Missing nodes still fail the build.
It does not download model weights or queue image inference.

## Not executed locally

The corrected Docker build/push, a real ComfyUI registry run, RunPod deployment,
model-weight downloads, CUDA inference, VRAM/timing benchmarks or visual
identity comparisons. This environment has no Docker executable and direct
upstream source downloads failed DNS resolution; the runtime was not recreated.
The previous user-run Docker smoke failure is the supplied evidence, not a
successful validation of this corrected release.

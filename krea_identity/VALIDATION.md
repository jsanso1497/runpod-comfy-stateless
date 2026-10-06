# Krea Identity 1.0 validation boundary

Prepared from `New Compressed (zipped) Folder(1).zip`.

## Executed locally

- 47 new Python tests, including real CPU Torch image tensors, reference-sheet
  layout, no-stretch fitting, canvas caps, crop/stitch behavior, exact outside-crop
  preservation, true rebalancer bypass, fixed-seed serialization, graph wiring,
  versioned-workflow preservation and opt-in asset selection.
- All 511 existing repository Python tests, including source-snapshot tests.
- 2 existing frontend widget-migration checks.
- 23 static workflow graphs: 19 retained plus 4 new.
- Python 3.10-compatible syntax checks, shell parsing, both Dockerfile command
  structures, all three Actions definitions, dependency pins and model catalogs.

Total Python tests: 558. These include mocks for network/service boundaries and
do not substitute for model inference.

## Added to Docker builds, not run locally

`python /opt/krea-identity/check.py --build-smoke --comfy-home /opt/comfy-bundle`
starts a real CPU ComfyUI server and checks the new graphs against its actual node
registry. It does not download weights or generate images. Dependency or schema
mismatches fail the build rather than being silently ignored.

## Not executed

Docker build/push, RunPod deployment, actual weight downloads, CUDA inference,
peak-VRAM/timing measurements, identity scoring, photographic comparison,
reference-sheet versus single-reference comparison, rebalancer on/off comparison,
or SeedVR2 facial-detail comparison. The local environment has no Docker
executable and reports CUDA unavailable.

No quality ranking, guaranteed exact identity, or successful live deployment is
claimed by these checks.

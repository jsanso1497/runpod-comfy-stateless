# Architecture and operational contract

## Source of truth

`catalog/tasks.json` maps task IDs to workflows, dependency asset groups and compatible workspaces. `catalog/models.json` pins ComfyUI and external node source commits. `catalog/assets.json` contains public model origins and expected source hashes where supplied. `catalog/migration.json` records all 82 source workflows. The HTML page is generated from these same records.

## Separate environments, shared interface

Each model workspace gets a separate image from one Dockerfile and one build action. A single fresh upstream PyTorch runtime replaces the historical project images. Those old images could contain private text in lower layers; deleting it in a newer layer would not remove it. The new build never inherits them.

ComfyUI code lives at `/opt/ComfyUI`; the workbench source is at `/opt/workbench`. User input, output, workflows, model cache, reference packages and helper assets are stored under `WB_DATA_ROOT`. Model directories are linked explicitly to those paths. No installer searches several possible active ComfyUI locations.

Source commits, a base image tag and the Ollama executable checksum are supplied. Python package dependency resolution still occurs during the build; it is not a fully offline reproducible lock. The successful build records `pip freeze` and source provenance. Promote a tested image by digest, not only by its moving `*-hq` alias.

## Startup and on-demand preparation

The authenticated gateway starts the selected ComfyUI installation, copies default config only when missing, and prepares the selected task asset groups. ComfyUI can start before large downloads finish, but a task is not shown as available until its required file records and node types exist.

The runtime exposes a credential-free status endpoint, fixed-catalog task preparation, immutable workflow downloads and input/output media listing behind authentication. Downloads are serialized with a filesystem lock. Runtime-only private entries use the same restricted provider transport. A failed optional asset does not remove unrelated working tasks.

The sidebar's availability check is a readiness check, not certification. `scripts/validate_live.py` additionally compares graph ports with actual installed node schemas. GPU execution and a human visual review remain separate release gates.

## Quality policy

No INT8/FP8/Lite variants, no BFS methods, no torso-specific code and no KREA workspace are shipped. Higher precision does not justify blindly increasing every sampler or output dimension. Source model-specific methods, crop sizes, image ordering, exact-pixel protection and offload handoffs are retained unless explicitly migrated.

H3 still generation has been ported from a Lite-only installation gate to the Full/BF16 configuration. It remains experimental until GPU-tested. The helper models are separate 32B F16 Instruct and Thinking tags. They are prepared independently, loaded sequentially and unloaded before H3 generation through the source helper handoff logic.

## Storage

Stateless mode budgets container disk for models and working data. Export everything important before stopping or terminating a Pod. A `/workspace` path by itself does not prove persistence.

Persistent mode reserves a separate `/workspace` volume and keeps container capacity for installed software. A Pod volume supports stop/start reuse but is not a network volume and does not promise survival after Pod deletion. Choose and manage network storage explicitly when deletion durability is required. Budget estimates exclude your private libraries, long video outputs, and optional assets whose original sizes were not specified.

## Hosted configurator

The page works independently of a deployed Pod. To host it, enable GitHub Pages using GitHub Actions and manually run `Publish configurator to Pages` with the explicit public-catalog confirmation. Only `site/` is uploaded. Pages availability and private-site access depend on the GitHub account plan; a private repository does not automatically imply a private website. The same page is also served by a running Pod at `/_workbench/` behind its password.

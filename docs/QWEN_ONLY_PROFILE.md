# Qwen-only profile and watermark CI correction

Base inspected: `a1732f3024a8c24ed2077efae08294a0fdf8874a`.
This is an incremental update on top of the consolidated repair v2, not a replacement
for its QKEEP1, CUDA-compatibility, progress-reporting, or privacy changes.

## What caused the attached CI failure

The attached GitHub Actions log reports one failure and 683 passes. The failing
watermark regression expected `watermark` at the SAM text encoder even though
`detection_prompt(text, -1)` now intentionally requests `watermark:64`. The fix
updates the expectation and strengthens the fixture to union two nonempty masks.
It does not undo the all-instances implementation or reduce test coverage.

## Qwen-only launch

The new `WB_QWEN_ONLY=1` mode applies only when `WB_WORKSPACE=qwen`.
Its default is OFF, so other existing templates and workspace families are unchanged.
Rebuild the image before using the new flag. Old images do not implement it.
Use the complete settings in `RUNPOD_QWEN_ONLY.env.example` as RunPod key/value rows,
not shell commands. RunPod resolves the secret placeholders.

The profile:

- Removes `ollama`, `qwen_pe_i2i`, `qwen_stock_pe`, and `moss_image` from the Qwen
  runtime catalog's download groups and toolbox.
- Hides QMOSS, QMOSS_I2I, U10, and U11 from the Qwen sidebar. Shared U11 remains
  available in the Krea workspace; other workspace sources are unchanged.
- Leaves Q01-Q28, SQ, QNATIVE, QOVERLAY, QDEV, QGS1, QKEEP1 and Qwen utilities
  available. Only asset preparation, not workflow availability, is selected by WB_TASKS.
- Preserves the main Qwen BF16 diffusion model, its required Qwen3-VL 8B text
  encoder, and its VAE. The base text encoder is NOT a prompt enhancer.
- Preserves SAM, SeedVR2, Qwen consistency adapters, and private library bindings.
  SeedVR2 remains because existing Qwen/restoration recipes use it; it is an image
  restoration model, not a prompt-writing model.
- Does not launch/pull Ollama for this selection. Legacy Workbench enhancer
  calls pass the original prompt through unchanged, even if an old toggle is ON.

### Native workflows need more than an OFF switch

A connected optional CLIP loader can refer to a missing enhancer filename even
when a downstream switch is OFF. Qwen-only mode therefore projects native Qwen
workflows at load time: the direct user prompt is wired into TextEncodeQwenImage21,
the enhancer branch and its model loader are removed, and the subgraph/outer
ports and widgets are reindexed. The base image model, encoder, VAE, sampling,
references and protection geometry remain unchanged.

The Workbench workflow endpoint, readiness checks and versioned factory files
all use the same projected graph. Source JSONs remain unchanged on disk. Reopen
workflows from the Workbench sidebar after deploying; old browser-saved native
JSONs are not retroactively edited. Manually importing a different workflow is
outside this profile's graph projection.

## Models and storage

Expected standard preparation groups for the supplied environment:
`qwen,qwen_identity,sam,seedvr2` (order may vary).
No enhancer or MOSS model group should appear. Preparing a disabled group through
the Workbench API is rejected; it is not silently downloaded.

Start with the existing A40 48 GB configuration, at least 64 GB container RAM,
80 GB container disk and a 200 GB volume/network volume mounted at `/workspace`.
These are planning sizes, not a newly benchmarked minimum. Increase persistent
storage for large private checkpoints or extensive output. A network volume is
preferable when keeping files across Pod deletion. Verify the mount before
replacing any existing Pod; do not assume a directory named `/workspace` is persistent.

Model files already on a reused volume are not deleted. This mode prevents the
profile's downloads and use of prompt enhancers; it does not reclaim old storage.

## Deployment

1. Run the supplied updater with `--repo . --check`, then `--repo .` from the local
   repository. All target paths are checked before any write. Backups and rollback
   records are stored beneath Git's private directory.
2. Review, commit and push using GitHub Desktop.
3. Run Validate repository, then Build HQ workspace with workspace `qwen`.
4. Deploy the exact digest from the successful build, or the newly built
   `ghcr.io/jsanso1497/runpod-comfy-stateless:qwen-hq` tag.
5. Replace the old environment-variable block with the Qwen-only block. Keep the
   existing secrets; remove either private-library binding if that secret does
   not exist. No WB_QWEN_PROMPT_MODEL/WB_QWEN_PROMPT_FAILURE setting is needed.
6. Open Qwen workflows from Workbench, not an old browser copy.

For a read-only runtime check use the image's `python /opt/workbench/scripts/doctor.py`.
The configuration should show only the four standard preparation groups above.

## Verification scope

The attached verification JSON records the tests actually executed here. These
include native workflow projection, mask-instance behavior contracts, original
prompt preservation, blocked preparation requests, skipped Ollama startup,
factory/source consistency, installer safeguards and existing CPU repair,
privacy and compatibility tests. Provider downloads, the full GitHub test suite,
Docker building, browser import and live GPU image quality must still be checked
on the rebuilt image. No remote changes are performed by this package.

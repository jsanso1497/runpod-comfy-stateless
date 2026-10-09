# Qwen I2I prompt-enhancer update: validation

Version: 0.1.2. Source: the user-uploaded `runpod-comfy-stateless-main (1).zip` (208 files), not an assumed live checkout.

## Implemented

- Official Qwen/Qwen-Image-2.1-PE-I2I with original BF16 shards and pinned source revision.
- One OFF-by-default control in all 35 Qwen generation workflows, connected to all 40 Qwen edit passes.
- Updated all 34 existing matching API exports.
- Added standalone U10 prompt preview. Total catalog: 58 entries and 70 editable workflows.
- Kept all existing sampling, crop, mask, reference, output and negative-prompt connections unchanged.
- Added original/final prompt UI, exact-literal checks, response-schema validation, per-pass routing and content-keyed rewrite caching.
- Added approximately 19 GB to the Qwen default asset preparation and configurator estimate, with no new environment variables or secrets.
- Kept existing ComfyUI revision, Dockerfile, requirements, GitHub Actions, provider downloader and other model families unchanged.
- No private LoRA text file was opened or packaged.

## Checks actually run

| Check | Result |
|---|---|
| Python tests, including mocked inference | 127 passed |
| Existing warnings | 2 non-failing aiohttp warnings |
| Configurator assertions | 112 passed |
| Prompt-preview JavaScript assertions with mocked ComfyUI hooks | 28 passed |
| Static graph/catalog validation | 70 workflows, 58 tasks, 82 original source dispositions; no errors |
| Original Qwen node-content regression | 35 workflows, 40 encoder passes; only optional control connections added |
| Other existing workflow files | 34 unchanged byte-for-byte |
| Existing API graph invariants | All 42 checked; only the relevant controls/connections added |
| GitHub Actions and secret mappings | Unchanged |
| Generated HTML/catalog | Deterministic regeneration |
| Desktop/mobile configurator smoke test | Passed in headless Chromium; no JS errors, no horizontal mobile overflow, no external requests |
| Python compilation | Passed |

The browser smoke test loads standalone HTML inline because file navigation is restricted in the testing environment. Prompt-preview UI assertions use mocked ComfyUI frontend hooks, not a running ComfyUI server.

## Not performed

No remote repository was changed. Live access to the private repository was unavailable. No Docker build, multi-gigabyte model download, GPU inference, memory benchmark, latency benchmark, or visual-quality comparison was performed. The original official BF16 model and current pinned native loader APIs were inspected, but their combined execution remains a GPU smoke-test requirement.

An OFF toggle means exact original prompt routing relative to the existing encoder, not a guarantee of identical image pixels across different software/hardware. ON can clarify a prompt but is not guaranteed to improve every result; use U10 to review wording first.

## Install

This is an overlay, not a full repository replacement. Upload only the contents of `UPLOAD_TO_REPO` at the existing repository root. Keep `.github` and all other unlisted files. Validate, rebuild the Qwen workspace, and deploy the newly reported image digest. Review the updated storage estimate before launching. Fresh catalog workflows include the control; saved personal copies are intentionally preserved.

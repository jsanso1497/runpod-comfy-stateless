# Consolidated Qwen + MOSS repair

Source audit: `jsanso1497/runpod-comfy-stateless` at `e84d27ccaf1d3184bd3b8122207930402dcd7680`.
Package version: `audit-20261018.1`.

This is a source repair, not a certificate of GPU compatibility or of successful background reconstruction. It supersedes the individual QKEEP1 alpha-stripping, green-fill and opaque-prefill hotfixes supplied earlier. Do not reapply those older patches over it.

## What changes

### CUDA-12 compatibility

A local custom-node shim activates before workflow model loading. On the reported CUDA 12.8/PyTorch build, it disables incompatible Comfy-Kitchen attention hints and uses standard PyTorch attention. It also replaces the direct fixed-cache attention call used by Qwen3.5 token generation, including multi-token prediction. This addresses both observed MOSS errors, not just the image-generation error.

`WB_CUDA_COMPAT=auto` is the default. `safe` forces the fallback and `off` disables this shim for a controlled test. The shim does not change model weights, model precision, sampling settings, PyTorch, CUDA, or the host driver. Supported CUDA-13 runtime/driver combinations retain their native path in auto mode. This is a Workbench workaround for the pinned ComfyUI APIs, not an upstream guarantee. Recheck it before any future ComfyUI upgrade.

The earlier MOSS-only bypass worked on the user's Pod with prompt upsampling disabled. The broader shim in this package has CPU dispatch and attention-math tests, but has not been executed on that GPU. Its first MOSS run should use the prompt upsampler OFF; test ON separately after image generation works.

### QKEEP1: reconstruction and compositing

The original problem was not solved by deleting alpha: that exposed hidden purple RGB. The green-conditioning test also returned transparency. This repair does not simply remove the alpha check or label transparent pixels as repaired scenery.

The new path constructs a rough opaque fill from visible background pixels, excluding every detected person and protected region from its donor set. It asks Qwen to refine that complete scene. The factory graph no longer feeds the isolated person against a gray backdrop as image2. Existing saved graphs remain compatible with the runtime node ports, but the freshly generated QKEEP1 graph is preferred.

The final paste mask remains defined by SAM and the user protection masks. The original keeper's visible pixels and every painted protection pixel remain unchanged. A straight-alpha overlay is exported alongside the opaque full-resolution composite. Normal compositing of that overlay over the same 8-bit source reproduces the saved composite.

If the generated crop still contains low-alpha RGB, that RGB is not exposed. A coarse background fill covers those pixels. Such output is labeled `DRAFT_FALLBACK_REVIEW_REQUIRED`, gets `_DRAFT_FALLBACK` in the filename, displays a warning, and saves the prefill and raw-model diagnostic PNGs. This is not a claim of successful semantic repair. Large gaps can look smeared or wrong. `transparency_policy=stop on transparency` rejects substantial fallback instead. Even fully opaque model output requires visual review for anatomy, background continuity and identity.

Mask preview remains lazy: `run_edit=OFF` does not run Qwen. One-person/no-repair input remains an exact no-op. QOVERLAY's watermark defaults remain SAM ON, edit ON and target watermark.

### Observable preparation

Standard assets are recorded as ready after each asset succeeds, not only after the entire group finishes. A failed recheck clears that asset's stale ready record. The progress state distinguishes metadata resolution, transfer, checksum verification, private libraries and Ollama. Ollama downloads use streaming progress instead of a silent long request. Heartbeats and a final summary appear in the log, and progress is also visible in the sidebar/status API.

Checksums and original precision/source rules remain mandatory. Within a preparation job, unchanged files can reuse a checksum result keyed to their file metadata; changed files are rehashed. Private URLs, filenames, provider error bodies and credentials are not printed as progress. Concurrent preparation clicks are rejected, and cancellation waits for an active file-writing worker before releasing the preparation lock.

### Automatic MOSS preparation

For the already-installed Qwen + MOSS addon, `WB_MOSS_AUTO_PREPARE=1` is the default. It adds `moss_image` to startup preparation even when an old `WB_TASKS` value omits QMOSS. Both MOSS workflows still share the same three source-pinned files. No new model is introduced.

To opt out, set the variable to 0 AND remove MOSS tasks/features from the explicit startup selection. The global workflow catalog remains unchanged.

### Other audit fixes

- Factory workflows use a content-versioned directory. New source versions no longer silently reuse stale factory copies. User-edited copies and old versions are preserved, not overwritten.
- SAM `instance_index=-1` now requests multiple detections when a plain object label is supplied. Explicit counts such as `person:6` or `shirt sleeves:2` remain unchanged. SAM loading/encoding failures are also covered by model cleanup.
- The legacy overlay exporter respects `WB_SAVE_METADATA=0` and previews the full composite first while preserving the separate overlay file. Q21PhotoReviewSave also strips nested effective prompts and reference filenames from saved verification metadata/sidecars when metadata is disabled; its local UI report is unchanged.
- Validation descends into native subgraphs and checks reciprocal connections, endpoints and execution cycles. QKEEP1 is included in generated-file reproducibility checks.
- Docker health checks query ComfyUI readiness rather than only gateway liveness. The public readiness response contains no private information; workflow, files and preparation endpoints remain authenticated.
- `scripts/doctor.py` prints a read-only diagnostic with actual cgroup memory, data mount, free space, driver and preparation state. It is explicitly included in Docker's copy and ignore allowlists.

## Installation and recovery

Use the supplied standalone updater from Mac Terminal inside the local repository:

```bash
python3 "$HOME/Downloads/apply_workbench_consolidated_repair.py" --repo . --check &&
python3 "$HOME/Downloads/apply_workbench_consolidated_repair.py" --repo . &&
git diff --check &&
git status --short
```

All files are preflighted before any write. Exact-hash guards protect replaced files; syntax-aware or exact-context edits protect shared files. Unrelated edits outside those changed regions are preserved. Known prior QKEEP1 patches are supported; unknown local modifications cause a stop, not a forced overwrite.

The updater prints its backup path under Git's private directory. Roll back before publishing with:

```bash
python3 "$HOME/Downloads/apply_workbench_consolidated_repair.py" --repo . --rollback "/exact/backup/path/printed/by/updater" --check
python3 "$HOME/Downloads/apply_workbench_consolidated_repair.py" --repo . --rollback "/exact/backup/path/printed/by/updater"
```

Rollback refuses to overwrite files edited after installation. The updater performs no network requests, commits, pushes, model downloads or live-Pod changes.

Review and publish through GitHub Desktop. Run Validate repository and Build HQ workspace with `qwen`. Use the exact image digest from the successful new build. The old main build was successful, but the new repair must pass its own full CI and Docker build.

After deploying, open QKEEP1 from the Workbench sidebar or the new `Workbench Factory/source-...` directory, not an earlier saved factory copy. Keep your provider/private-library secret bindings. See `RUNPOD_QWEN_AUDIT.env.example` for the complete environment.

## Existing Pod: storage and test plan

The reported `df` output showed `/workspace` on the root `overlay` filesystem. Treat those models, uploads and outputs as potentially disposable container data. Back them up or copy them to a verified persistent/network volume BEFORE stopping/recreating/deleting that Pod. A local repository update cannot move or protect its data.

This package intentionally does not ask you to paste another long live source replacement. Deploy the rebuilt image separately while retaining/backing up the old Pod's data. A40 was sufficient for the reported Qwen run; its 67 GB cgroup RAM is not the 503 GiB host memory shown by `free`. The large optional local enhancer has additional RAM needs. No specific GPU resolution/throughput guarantee is made here.

Smoke-test in this order:

1. `python /opt/workbench/scripts/doctor.py`. Verify mount, preparation groups and no missing required nodes/assets.
2. MOSS T2I at stock resolution with prompt upsampler OFF, then ON. Verify CUDA-compat log and successful image output in both tests.
3. MOSS I2I at the existing 1 MP / denoise 0.35 defaults.
4. Existing Qwen edit and QOVERLAY, including manual and SAM masks.
5. QKEEP1 mask preview, selection of the keeper, a small removal, painted protection, then a more difficult removal. Inspect actual composite, overlay and any fallback diagnostics. Do not use a DRAFT file as an approved result without inspection.
6. Confirm no-op keeper case and exact overlay recomposition. Check that a new factory directory appears without modifying old user files.

## Verification scope

The accompanying verification report separates numerical/CPU tests, mock-based contracts, installer tests and remaining GPU/build work. The repository was reviewed through its live GitHub connector and compared with the uploaded logs and earlier packages. A network-disabled container prevented cloning the full repository and fetching all upstream snapshots for a complete local CI run. Relevant exact source files from the packages were checked against GitHub blob IDs; partial shared-file edits are guarded by the reviewed source contexts.

Passing local tests does not establish full-repository CI success, Docker build success, GPU kernel compatibility, SAM detection quality or realistic background repair. No remote changes were made.

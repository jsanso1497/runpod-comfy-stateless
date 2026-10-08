# Qwen Photo 2K / SAM 3.1 + Native Suite + Torso Lock

**Integration overlay:** v2.1.0, October 7, 2026. **Status:** code and offline validation complete; **not yet built, pushed, or tested on a live GPU**. This package requires the existing `qwen21_photo_edit/` folder from your current `runpod-comfy-stateless` repository. It does not contain or replace your entire private repository.

## What the one unified image contains

| Area | Present in same ComfyUI? | Default model download |
|---|---|---|
| Qwen Photo 2K / SAM 3.1 (BF16 + INT8 *workflow files*) | YES: 2 workflows | BF16 Qwen + SAM 3.1 |
| Native Qwen 2.1 identity/body/face/repair/reference workflows | YES: 34 workflows | Reuses Photo BF16 weights |
| Torso Lock 36, 36A, 37, 38 | YES: 4 workflows | Reuses Photo BF16 weights |
| Native SeedVR2 finishing or BFS adapter comparison workflows | Source retained but **NOT installed** in this lean unified image | Neither SeedVR2 nor BFS downloaded |

**40 usable workflows** are installed by default, with three separately organized groups under saved workflows. Both Photo workflow JSON files are installed, but INT8 *model weights* are downloaded only when you deliberately select `QWEN_PHOTO_MODEL_PRECISION=both`. Do not select `int8` alone because Native and Torso workflows need BF16 models. No LoRA is enabled by default.

The **sports-bra / navel-position** scenario remains in `36_Sports_Bra_Torso_Landmark_Lock.json`. The unchanged original abdomen is restored after the edit mask is composited. SAM selection is available in the separate Photo 2K/SAM 3.1 workflow; automatic SAM masking is *not* invisibly added to Torso Lock's own mask input.

## Important: the Photo Docker build error

The uploaded GitHub log for the Photo image hit `No space left on device` while installing a large set of optional ComfyUI gallery/demo workflow-template media. The unified Dockerfile is generated from your **actual existing Photo Dockerfile**, preserving its pinned ComfyUI, Torch, SAM and runtime installer; it changes only:

- Pip wheel downloads/install use `--no-cache-dir`.
- The optional **ComfyUI built-in demo workflow-gallery bundle** requirement is removed from the Docker dependency list. Your own workflow JSON files and the ComfyUI frontend remain.
- The GitHub build job reclaims unused runner SDK/toolchains before Docker Buildx.
- Reviewed AusBoss, Native and Torso components are added with live runtime-schema checks.

This is a best-effort fix for the observed BuildKit disk exhaustion, not a guarantee that GitHub's disk allocation will always suffice. No model checkpoints go into Docker layers; weights are downloaded at Pod startup.

## Integrate into existing GitHub repository

**Do not upload this ZIP as a replacement repository.** Extract it locally or to your own checked-out repository.

In a terminal, using your actual cloned GitHub checkout:

```bash
python /path/to/extracted/APPLY_QWEN_UNIFIED.py --repo /path/to/runpod-comfy-stateless
```

On Windows PowerShell, the equivalent is:

```powershell
python C:\path\to\extracted\APPLY_QWEN_UNIFIED.py --repo C:\path\to\runpod-comfy-stateless
```

The script checks for your existing Photo Dockerfile/startup/validator, copies the Native Suite and four Torso workflows, adds the new unified action, and **derives** `qwen21_photo_edit/Dockerfile.unified` from your Photo Dockerfile. It refuses an unrecognized source layout or conflicting existing native files rather than guessing. For native-file conflicts, inspect changes before considering `--force-native`.

It also attempts to update `SOURCE_SNAPSHOT.json` using the actual repository manifest structure and runs the existing `tools/verify_snapshot.py` when available. If that verification fails, it restores the old manifest and rolls back all copied files. Explicit `--force-native` can rehash modified Native files only, and still requires your repository snapshot verifier to pass. Other pre-existing digests are never silently changed. **Do not disable snapshot checks or use `--skip-snapshot` in a snapshot-guarded production checkout.**

Verify before committing:

```bash
python qwen21_unified/scripts/verify_bundle.py --repo .
python qwen21_native/scripts/validate_workflows.py --profile upscale --include-bfs
python qwen21_native/addons/torso_lock/scripts/validate_workflows.py
python qwen21_unified/scripts/generate_dockerfile.py --repo . --check
python tools/verify_snapshot.py
```

GitHub's current checkout should be the source of truth. You may omit the final verifier command only if your repository does not have it. Changes stay inside `qwen21_native/`, `qwen21_unified/`, the newly generated Photo Dockerfile, the new GitHub action, and the source snapshot file. **Your old Photo Dockerfile and its old build workflow remain intact.**

To avoid automatically queuing older Photo and Native actions while you introduce this new package, you may make the integration commit with `[skip ci]` and then **manually** run the unified action. Do not use `[skip ci]` on a PR that requires branch protection checks, since such checks can remain pending.

```bash
git add qwen21_native qwen21_unified qwen21_photo_edit/Dockerfile.unified \
  .github/workflows/build-qwen21-unified.yml SOURCE_SNAPSHOT.json
git commit -m "Unify Qwen Photo, Native and Torso Lock [skip ci]"
git push
```

Open GitHub Actions and choose **Build Qwen Photo + Native + Torso (Unified)** > **Run workflow**. **Do not rerun the older failed Photo build** and do not select an older SHA. The resulting tags, only after a successful build and push, are:

```text
ghcr.io/jsanso1497/runpod-comfy-qwen21-photo:unified
ghcr.io/jsanso1497/runpod-comfy-qwen21-photo:unified-v2.1.0
ghcr.io/jsanso1497/runpod-comfy-qwen21-photo:unified-sha-<commit-sha>
```

`latest` and your existing Photo/Native image tags are **not overwritten** by this new action.

## RunPod template: Qwen unified BF16 (recommended)

Use the `qwen21_unified/config/runpod-template.json` file as your canonical preset. The essential settings are:

| Setting | Value |
|---|---|
| **Image** | `ghcr.io/jsanso1497/runpod-comfy-qwen21-photo:unified-v2.1.0` (after successful publication) |
| GPU | NVIDIA A40 **48 GB** or higher to start; not benchmarked for 4 MP or many references |
| System RAM | Prefer **96 GB+** when available for BF16 offloading |
| Container disk | **160 GB** suggested for stateless Pod |
| Persistent volume | **0 GB** (same stateless behavior), or supply one if you want persistent downloads/outputs |
| Exposed port | `8188/http` |
| Docker startup command | **Blank** |
| Docker entrypoint override | **Blank** |
| Model precision | `QWEN_PHOTO_MODEL_PRECISION=bf16` |
| SAM 3.1 | `QWEN_PHOTO_DOWNLOAD_SAM=1` |
| VRAM setting | `VRAM_MODE=auto` |
| Port env | `COMFY_PORT=8188` |
| BFS | `ENABLE_BFS_COMPARISONS=0` (no BFS adapter in this default image) |
| Hugging Face token | `HF_TOKEN`, **only if required**, supplied as a secret |

Selecting `QWEN_PHOTO_MODEL_PRECISION=both` makes both Photo BF16 and Photo INT8 checkpoints available, but increases runtime download, disk, and cache demand. Selecting `int8` alone is rejected to avoid advertising nonfunctional Native/Torso BF16 workflows.

**Do not set `DATA_ROOT` to the old Native Suite's path.** The existing Photo `start.sh` remains responsible for model download paths. To override a nonstandard ComfyUI path deliberately, the integration accepts `UNIFIED_COMFY_DIR` and `UNIFIED_USER_DIR`, but they should be omitted for the normal template.

If the GHCR image is private, make it accessible to RunPod via your registry configuration. Docker publication and private GHCR authorization are separate steps.

## Runtime proof that all components are present

Open ComfyUI on HTTP 8188. The image verifies live `/object_info` schemas for native Qwen, SAM3, AusBoss, Torso, and Photo. After startup, this file is written:

```bash
cat /workspace/qwen21_unified/logs/runtime_schema_check.json
```

A successful report identifies **Photo 2 + Native 34 + Torso 4**. The startup supervisor also locates the actual running `main.py` process and its `--user-directory`, then ensures all three workflow groups are present there. Missing node schemas or a missing Photo/SAM source workflow cause a clear startup failure, rather than a falsely "ready" template.

This check is **structural**. Live Qwen image generation, actual SAM mask detection, VRAM headroom, and the quality of reconstructed anatomy still require an on-GPU test with your real photographs.

## Notes for future releases

See [`qwen21_unified/config/RELEASE_CONTRACT.json`](qwen21_unified/config/RELEASE_CONTRACT.json). This explicitly records the source/model/RunPod constraints that future upgrades should preserve. Do not replace the unified template image with a Photo-only or Native-only tag unless intentionally changing the feature set.

# Qwen 2.1 Photo Edit + SAM 3.1: GitHub overlay

**Overlay version:** 2.0 GitHub integration | **Reference tree:** the user-uploaded `New Compressed (zipped) Folder(5).zip`, dated October 7, 2026.

This overlay adds an isolated GitHub Actions build and image, while preserving the repository's root Dockerfile, FLUX Photo, KREA, H3, existing `qwen21_identity` and `qwen21_native` paths. No existing application Python files or existing build workflow files are modified. One existing root file, `SOURCE_SNAPSHOT.json`, is updated to cover the complete provided source tree and the new overlay.

## Copy into GitHub

1. Extract the **overlay ZIP**. Its contents start at the repository root, NOT inside another `runpod-comfy-stateless-main` folder.
2. Merge those files into the repository root next to your existing `Dockerfile` and `README.md`. Make sure the hidden `.github/workflows/build-qwen21-photo-edit.yml` file is uploaded as well.
3. Commit the changes. With a local clone:

   ```bash
   git add qwen21_photo_edit .github/workflows/build-qwen21-photo-edit.yml SOURCE_SNAPSHOT.json QWEN21_PHOTO_EDIT_GITHUB_OVERLAY.md
   git commit -m "Add isolated Qwen Image 2.1 photo editor with SAM 3.1"
   git push
   ```
4. Open GitHub Actions. The new action is **Build isolated Qwen 2.1 Photo Edit + SAM 3.1**. It runs automatically on pushes to `main`/`master` that touch its own folder/workflow, or you can start it with **Run workflow**. Docker image publication happens only after validation and the build succeed.

After a successful run the image is:

`ghcr.io/<your-github-owner-lowercase>/runpod-comfy-qwen21-photo:latest`

For the repository owner used in your earlier image names, that would be:

`ghcr.io/jsanso1497/runpod-comfy-qwen21-photo:latest`

The workflow also publishes `:sha-<full-commit-sha>`. No package is published by making or uploading this ZIP. For private GHCR packages, configure registry authorization in RunPod or make the package publicly accessible through GitHub's package settings.

## RunPod settings

Use a **new** RunPod template or Pod for this tag, not your existing FLUX or H3 image tag. The dedicated container exposes ComfyUI on HTTP **8188**, and intentionally does **not** bundle a separate FileBrowser service on 8888. ComfyUI itself supports image upload and saving. No external model API is used.

Recommended initial hardware: **A40 48 GB or larger** for the BF16 preset, with memory-dependent tradeoffs. There is no GPU benchmark or guarantee that a 4 MP crop fits on the A40. Allocate around **150 GB of container disk** for the ComfyUI environment, models and outputs when stateless. Use `VRAM_MODE=low` or switch to INT8 if out of memory.

Optional RunPod environment variables:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
COMFY_PORT=8188
QWEN_PHOTO_MODEL_PRECISION=bf16
QWEN_PHOTO_DOWNLOAD_SAM=1
VRAM_MODE=auto
```

`HF_TOKEN` is optional when the Comfy-Org files are publicly downloadable. A token must be supplied through a secret if needed. The selection `bf16` downloads only the BF16 Qwen model and BF16 Qwen3-VL encoder, plus the 2.1 VAE. `int8` downloads the matching INT8 ConvRot model/encoder instead, with the same VAE. `both` downloads both sets. `none` disables automatic Qwen model downloads. `QWEN_PHOTO_DOWNLOAD_SAM=1` downloads the SAM 3.1 FP16 checkpoint. Downloads occur **at Pod startup**, not during the GitHub Docker build.

The container uses a verified, pinned ComfyUI source commit rather than updating an unknown installation at runtime. The Dockerfile preserves the PyTorch/CUDA versions from its PyTorch base image, and runs no-weights tests before publication.

## Editing workflow

After opening ComfyUI, select the **Qwen21_Photo_Edit** folder and start with:

- `Qwen21_Photo_2K_SAM3_BF16.json` (higher-precision edit path)
- `Qwen21_Photo_2K_SAM3_INT8.json` (lower-memory alternative)

The 2K crop workflow provides one source photo, manual or SAM 3.1 mask selection, mask preview with generation disabled, four visible reference/role loaders with up to nine references supported, three opt-in Qwen 2.1 LoRA slots, native Qwen 2.1 conditioning, a 40-step Euler sampler with CFG 1, and protected crop/stitch/compare/export. Its default SAM example is `shirt sleeves:2`, using SAM's native `:N` detections syntax. The mask boundary remains editable; check the actual preview before generation.

**Manual masking:** In Source `Load Image`, right-click > MaskEditor, paint your target area and save. Set `auto_mask=OFF`. No SAM weights are loaded.

**Automatic masking:** Set `auto_mask=ON`, describe the object(s) briefly in `sam_prompt` (e.g. `shirt sleeves:2`), optionally restrict it to a painted search region and apply manual corrections. Use `instance_index=-1` to combine detections. SAM can be inaccurate at boundaries; the mask preview exists to catch that.

**Two-stage execution:** Keep `run_edit=OFF` until the preview is acceptable. Switch it ON to run Qwen and export. The final photo remains the original pixel dimensions, and the pixels outside the effective mask remain source copies. Only the crop is generated; this is not a separate trained masked-inpaint network.

**Reference numbering:** `<image1>` is always the source *crop*. Reference 1 is `<image2>`, Reference 2 is `<image3>`, etc. Set unused references to None and fill slots consecutively. Reference-role text is appended without an LLM prompt rewrite. Each reference has its own size ceiling, not a forced rescale of the entire photo.

**LoRAs:** Install Qwen-Image-2.1-compatible LoRAs into `/workspace/ComfyUI/models/loras`. All three slots default to None, and LoRAs from older Qwen Edit releases or FLUX should not be used. Unlike your older multipurpose template, this new isolated image does not automatically sync arbitrary shared LoRA URLs.

## Source-snapshot compatibility

**Important:** The uploaded ZIP already failed its own `tools/verify_snapshot.py` before this overlay, because it contained 67 existing paths omitted from its 210-entry `SOURCE_SNAPSHOT.json` manifest. The file names include the previously added `qwen21_identity` and `qwen21_native` trees. All 210 previously hashed files matched their expected digests. This overlay reconciles the manifest with **those exact uploaded files** and adds the new package/action. It does **not** disable or relax the repository's snapshot verification code.

This manifest is tied to the precise ZIP you uploaded. If the actual live GitHub repository differs, merge carefully and regenerate an audited manifest for the actual commit rather than uploading a mismatched manifest. Because existing GitHub Actions watch `SOURCE_SNAPSHOT.json`, merging this overlay may also trigger unrelated older builds. Their logic and images have not been changed.

## Install on an already running Pod (alternative)

You can also copy just the `qwen21_photo_edit` directory from the overlay onto a currently running pod, then use:

```bash
bash /workspace/qwen21_photo_edit/install.sh --comfy-home /workspace/ComfyUI --models bf16 --sam
```

This requires a sufficiently new native ComfyUI installation that supports **Qwen-Image 2.1**, **QwenImage21Cache**, and **SAM3_Detect**. The preflight checks these and stops without automatically replacing the installed ComfyUI/Torch stack if incompatible. The dedicated GitHub image avoids that version uncertainty.

## Validation scope

Python compile checks, graph/UI/API consistency checks, shell syntax checks, CPU unit tests and source-snapshot verification are included. Neither the isolated Docker image nor real GPU Qwen/SAM inference has been built/run as part of packaging this overlay. Use the GitHub build log and a real photo/segmentation run before relying on it for production output.

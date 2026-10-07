# FLUX Photo 1.0: 9B distilled, protected original-size inpainting

This is a new, separate RunPod image, built from the repository ZIP supplied for this update. The existing General, H3 Full, H3 Lite and Krea recipes and inference code are retained. Do not select the old General or H3 image for these new workflows.

**New build:** `Build FLUX Photo 9B Distilled image` in GitHub Actions.  
**New image after that build succeeds:** `ghcr.io/jsanso1497/runpod-comfy-flux-photo:latest`  
**New workflows:** `FLUX_Photo` in the ComfyUI workflow browser.  
**Model:** original non-quantized BF16 FLUX.2 [klein] 9B distilled, not base, 4B, FLUX.1, Krea or the separate KV model.

The image has NOT been built, published or run on a GPU during preparation of this ZIP. The new Action runs additional real ComfyUI CPU schema checks before publishing. Local validation and its limits are recorded in `flux_photo/VALIDATION.md`.

## 1. Put this complete source snapshot in GitHub

Extract the ZIP to a normal folder first. Preserve the folder structure and the hidden `.github` folder. Do not upload a ZIP as the source code and do not put everything inside an extra parent folder.

**For the current repository matching your supplied ZIP:** the separate `GitHub_Update_Only` archive contains every new or modified file and no deletions. Upload its extracted contents into the existing repository root, overwriting matching files and keeping all other existing files. This smaller upload is the most direct browser-only route. It was assembled by comparison against the exact supplied ZIP.

**For a fresh repository or a complete replacement:** use `Full_Repository`. It contains the entire updated source snapshot. Do not combine it with an unrelated or older source tree. The snapshot verifier detects missing, extra or changed files.

The only old files intentionally updated are README, BUILD_INFO, PACKAGE_FILE_LIST, SOURCE_SNAPSHOT and the snapshot verifier's version identifier. Existing model workflows and build recipes are retained. The new files live in `flux_photo/`, `.github/workflows/build-flux-photo.yml` and this guide.

Your supplied `config/lora_links.txt` is preserved byte-for-byte. It remains intentionally editable without breaking the snapshot verifier. Do not commit tokens or signed private URLs to a public repository.

Open **Actions > Build FLUX Photo 9B Distilled image** and run it on `main`. Its published-image summary provides the actual tag and digest. Under a different GitHub account, use the lower-case owner shown in that summary rather than `jsanso1497`.

Other existing Actions can also trigger because the shared snapshot manifest changes. They are independent builds, not the FLUX build. No new image tag exists until the new Action succeeds. For reproducible deployment, use the digest from its summary; `:latest` is the convenient moving tag.

## 2. Accept access to the original model

Visit the official BFL model repository and accept its access agreement while signed in to Hugging Face:

https://huggingface.co/black-forest-labs/FLUX.2-klein-9B

Create or use an HF read token authorized for that gated repository. Enter it in RunPod as the `HF_TOKEN` secret. The original 9B model uses BFL's non-commercial license. For commercial/client work, obtain appropriate commercial rights before use. This package does not grant additional model or LoRA rights.

The downloader uses only official BFL/Comfy repositories, immutable revision pins and SHA256 verification. It never switches to a quantized model, community mirror, 4B model or different editor to recover from an error. The gated diffusion file's SHA256 is obtained from authenticated Hugging Face LFS metadata at its pinned commit and verified at download. It was not downloaded during authoring.

## 3. Make a new RunPod template

| Setting | Value |
| --- | --- |
| Template name | `FLUX Photo 9B Distilled` |
| Container image | `ghcr.io/jsanso1497/runpod-comfy-flux-photo:latest`, after successful publication |
| Container disk | **200 GB recommended starting allocation** |
| Volume disk | **0 GB** |
| Network volume | None |
| HTTP ports | **8188, 8888** |
| TCP ports | None required |
| Docker command / entrypoint override | Leave blank |
| GPU starting point | 48 GB VRAM for the fit preset; 80 GB preferred for native large-crop experiments |
| Host RAM starting point | 64 GB; prefer 128 GB for the largest native edits |

The GPU/RAM figures are planning recommendations, not measured requirements or a guarantee that a particular crop fits. The model weights alone are roughly 35 GB on disk. Image layers, uploads, LoRAs, outputs and temporary files need additional room. More VRAM does not guarantee a better generated face or fix out-of-distribution resolution.

This is stateless. A new Pod downloads the models again. Export photos, masks, workflows and wanted outputs before terminating the Pod. No network-volume setup is required.

### Environment variables

Required secrets:

```text
HF_TOKEN=your_Hugging_Face_read_token
FILEBROWSER_PASSWORD=your_unique_password_at_least_12_characters
```

Recommended explicit defaults, already built into the image:

```text
FLUX_PHOTO_DOWNLOAD_MODELS=1
SYNC_SHARED_LORAS=1
VRAM_MODE=auto
COMFY_REF=pinned
COMFY_HOME=/workspace/ComfyUI
COMFY_PORT=8188
FILEBROWSER_PORT=8888
COMFY_MAX_UPLOAD_MB=1024
```

No Ollama, Krea, H3, SeedVR, Gemini or other API-model key is required. No source or reference image is sent to an external inference API by either workflow. The container still makes network requests for model/configuration downloads and runs the normal ComfyUI distribution.

For live LoRA link updates without rebuilding, optionally add:

```text
CONFIG_REPO=https://github.com/YOUR_OWNER/YOUR_REPOSITORY
CONFIG_REF=main
GITHUB_TOKEN=your_repo_read_token_if_the_repository_is_private
```

Without `CONFIG_REPO`, it uses the LoRA links bundled from your source snapshot. Live configuration supplies only `config/lora_links.txt`; it does not replace the packaged executable code. Store tokens as secrets, not inside URLs.

Open port **8188** for ComfyUI and **8888** for the password-protected JupyterLab file browser. The file browser can become available before model downloads finish. Use RunPod's private/authenticated access controls for ComfyUI itself; the file-browser password does not independently protect port 8188. Do not expose sensitive source/reference photos through an unprotected public ComfyUI endpoint.

## 4. Choose the workflow

| Preset | Generation crop | Best use | What happens to the full photograph |
| --- | --- | --- | --- |
| `FLUX_Photo_01_Quality_Fit_2048_v1_0` | Aspect-preserving fit inside 2048 px; smaller crops are not enlarged | Starting point, lower memory, general editing | Original canvas is untouched except for masked compositing |
| `FLUX_Photo_02_Native_3072_v1_0` | No crop resampling; pads to a 16 px multiple; default maximum 3072 px | Large native-resolution experiments with more VRAM | Same exact output dimensions, with native crop pixels stitched back |

For a 2500 x 2500 mask with 128 px of context on each side, the crop is 2756 x 2756 and the model receives **2768 x 2768** after right/bottom alignment padding. The native workflow therefore accommodates that crop without reducing it. Padding is removed before stitching, so the output photograph does not gain a border or change dimensions.

Native mode throws an explicit error above its selected size cap. It does not silently turn into the fit workflow. Increase the cap manually only with sufficient resources and a deliberate experiment. Masks near the photograph's edges may have less context because the source ends there.

**Important quality limit:** BFL describes FLUX.2 at up to roughly 4 MP. A 2768 x 2768 generation is about 7.66 MP and is beyond that published range. It may run but give worse texture, structure or identity than a smaller crop. Native size is not proof of native photographic detail. Tiled VAE encoding/decoding reduces VAE memory; it does NOT tile the diffusion transformer or make its attention cost disappear. There is no concealed tiled-diffusion refinement pass or whole-photo upscaler.

## 5. Edit a photograph

Open one preset from `FLUX_Photo`, or drag its JSON from `flux_photo/workflows` into ComfyUI. The JSON requires the bundled FLUX Photo custom nodes; it is not a standalone graph for an unmodified ComfyUI installation.

1. In **SOURCE PHOTO + MANUAL MASK**, upload the original photograph. Right-click it, open the Mask Editor, paint the region that may change and save the mask back to the node. White mask means editable. An empty/mismatched mask stops with a clear error. For a replacement, include the entire old subject/object and unwanted boundary, not just its interior. Anything you leave unmasked is deliberately protected.
2. In **REPLACEMENT REFERENCE**, upload the replacement reference. A clear, tightly framed photo of the intended person/object is more useful than a 7K image where the subject occupies a tiny area. This graph accepts exactly one replacement image plus the source. The reference is fitted, never stretched; its default long-side cap is 2048.
3. In **YOUR PROMPT**, write the actual edit instruction. In the model's reference order, **image 1 is the source crop** and **image 2 is the replacement reference**. The software never rewrites your prompt. Specify what transfers and what must remain from the scene, for example facial identity versus clothing, pose, lighting or perspective.
4. Leave all three LoRA nodes at `None` initially, keep four steps, and run. Choose a compatible LoRA only when needed. Examine the comparison, then inspect the saved full-resolution PNG at 100% for fine detail.

Example prompt structure, not a required magic phrase:

```text
In image 1, replace the masked person's facial identity with the person in image 2.
Preserve the head position, expression, camera perspective and lighting from image 1.
Use image 2 for the person's facial proportions, eye shape, nose, lips, complexion
and identifying features. Keep the surrounding scene and all other people unchanged.
Render a natural photographic result with consistent local light and skin texture.
```

Adapt this for an object or a full-body replacement. Crop context matters: a very tight head crop cannot supply a whole-body pose. A single flat reference cannot reveal unseen facial/body detail. The workflow supports identity reference editing, not a guaranteed pixel-exact identity transplant.

## Controls worth changing

| Control | Default | Meaning |
| --- | --- | --- |
| `processing` | `fit` or `native` | Whether only the generation crop may be resized |
| `max_crop_side` | 2048 / 3072 | Work-crop size limit; not final-photo size |
| `context_pixels` | 128 | Real source context around the mask |
| `sampling_mask_grow` | 16 | Extra working-mask support for the sampler, not permission to change more final pixels |
| `scene_reference_long_side` | 1024 | Source crop supplied as reference image 1 |
| `replacement_reference_long_side` | 2048 | Detail cap for replacement reference image 2 |
| `text_encoder_device` | `cpu` | Keeps the full Qwen3 encoder out of GPU memory; slower than GPU encoding |
| `steps` | 4 | Intended distilled-step count; larger numbers are not automatically better |
| `seed` | 12345, fixed | Repeatable starting point; change seed to compare alternative generations |
| `feather_pixels` | 16 | Feathering inside your mask, measured at final source resolution |
| `boundary_color_strength` | 0 | Color matching off; optional conservative local boundary RGB shift |
| `preview_long_side` | 2048 | Comparison preview only; has no effect on saved output |

The sampler intentionally fixes Euler, CFG 1 and full masked replacement with the native FLUX.2 resolution-aware schedule. There is no conventional low-denoise slider in this version. Refine edit scope with the mask and prompt. Do not substitute FLUX.1 CLIP/VAE files or load the Qwen encoder as Krea.

## LoRAs

Three optional model-only LoRA slots are already connected in sequence. `None` is a valid value even when the LoRA folder is empty. Use adapters expressly compatible with **FLUX.2 [klein] 9B** and the distilled inference setup, not generic FLUX.1, 4B, H3 or Krea adapters. The core loader handles compatible weights; this template cannot infer an adapter's quality or guarantee every third-party LoRA is effective. Start with one LoRA at a moderate strength and compare against `None` using the same seed.

Either upload `.safetensors` files to `/workspace/ComfyUI/models/loras` using the file browser, or keep direct download links in `config/lora_links.txt`. The existing shared downloader is reused unchanged. After a manual upload, refresh the ComfyUI model list or reload the browser. No Civitai identifiers or checksum entry is required by the new workflow itself.

## What is preserved, and what is not

The stitcher copies the loaded source canvas, changes only pixels allowed by your saved mask, and saves a full-size lossless PNG. Zero-mask pixels remain identical to the decoded working source. Mask growth is only for sampling; inward-only feathering cannot leak the final edit outside your mask. Local color adjustment is off by default. No full-image VAE roundtrip, face restoration, sharpening, automatic grading or whole-image enlargement is applied.

**Use 8-bit sRGB RGB exports for both inputs.** The standard ComfyUI image loader and this exporter are not an archival RAW, 16-bit TIFF, Adobe RGB, Display P3, CMYK, HDR, layered-file or original-EXIF roundtrip. The PNG has an sRGB ICC profile and preserves 8-bit RGB values outside the mask. It does not preserve the original JPEG's compressed bytes, camera metadata or an arbitrary source color profile. Convert wider-gamut/high-bit-depth masters to sRGB copies first, keeping the masters separately. Canvas dimensions refer to the loaded, EXIF-oriented image.

The newly generated region can still have identity drift, invented detail, different local color or a visible join. Pixel preservation outside the mask is enforceable; perfect generated identity, color, optics and texture are not guaranteed by a generative model.

Output folder: `/workspace/ComfyUI/output/FLUX_Photo/`. Each output includes a `.verification.json` sidecar and embedded workflow/report metadata. The report records original/output dimensions, generation crop size, whether it was resized and zero-mask change verification. The interactive comparison uses reduced previews by default; it is not a substitute for inspecting the saved PNG at 100%.

## Failure handling

An out-of-memory error never silently reduces precision, changes model or resizes native mode. First use the fit preset; then reduce replacement reference cap to 1536 or 1024, reduce unnecessary context, split distinct subjects into separate edits, or choose more VRAM. Do not assume increasing steps will fix identity or seams. For a complex replacement that struggles as one huge edit, establish the subject first and do a separate, tighter face/detail edit afterward, reviewing both rather than automatically overwriting the first result.

A 401/403 during model download means access/token approval needs attention, not that a different model should be downloaded. A file-browser startup error can indicate a missing/short password. A source-snapshot failure means the GitHub upload is incomplete or mixed with older files; restore the complete ZIP instead of bypassing the verifier. A runtime node-schema failure during Docker build means the new image was not published, preventing deployment of that unvalidated graph.

# Qwen 2.1 Identity Kit

**Version 1.0.0 | Prepared October 7, 2026 | Personal image workflow**

Dedicated pipeline: genuine body reference + face reference + existing scene, with a separate body pass and head pass. Each pass uses AusBoss crop-and-stitch. The original scene is retained at its original dimensions; only the actual blend region is replaced. This is an original integration of native ComfyUI, BFS adapters and AusBoss nodes, not a byte-for-byte copy of the AusBoss Person Swap graph.

## Choose the image profile

| Profile / Docker tag | Weights downloaded on first startup | Intended use |
|---|---|---|
| `qwen21-identity-bf16` | Five files, approximately 32.85 GB decimal | Body swap, head refinement, source preparation, local repair, optional Lanczos resize |
| `qwen21-identity-upscale` | Those five plus two SeedVR2 files, approximately 49.85 GB total | Same pipeline plus optional generative still-image finishing |

For the leanest dedicated setup, use the BF16 tag. Use the upscale tag when SeedVR2 finishing is wanted. Neither profile downloads unrelated models. The upscale tag downloads its seven selected files at startup, but SeedVR2 is only loaded for inference when its workflow runs.

### Default five model files

```text
diffusion_models/qwen_image_2.1_bf16.safetensors
text_encoders/qwen3vl_8b_bf16.safetensors
vae/qwen_image_2.1_vae_bf16.safetensors
loras/bfs_body_swap_v1.0_qwen_2.1.safetensors
loras/bfs_head_v1.1_qwen_2.1.safetensors
```

Optional upscale profile adds only:

```text
SEEDVR2/seedvr2_ema_7b_fp16.safetensors
SEEDVR2/ema_vae_fp16.safetensors
```

No Ollama, prompt enhancer, KREA, Flux, Minimax, ControlNet, face detector, SAM, Lightning adapter, alternate head adapter, alternate precision set, or unrelated upscaler weights are downloaded. Only one Qwen text encoder is needed. The two BFS adapters are applied to separate branches from the base model, not stacked together.

Source-prep and local repair use the same Qwen base files without either BFS adapter. Lanczos resizing needs no model. SeedVR2 is optional because its synthesis can change likeness, even when the image looks sharper.

## Install into your existing GitHub repository

1. Extract the outer ZIP completely. Copy its `qwen21_identity/` directory into the repository root, and copy the included `build-qwen21-identity.yml` into the repository's `.github/workflows/` directory. Do not replace the existing root Dockerfile or other workflows. The `.github` folder can be hidden in file browsers.
2. Commit and push those two paths. In GitHub Actions, run **Build isolated Qwen 2.1 Identity**, or let the push to main/master trigger it.
3. The action builds two separate tags. It performs graph validation and a CPU-only live ComfyUI node/schema check during each Docker build. If that registration check fails, the image is not published. Check the action log rather than assuming a tag exists.

From an existing local clone, after copying the files:

```bash
git add qwen21_identity .github/workflows/build-qwen21-identity.yml
git commit -m "Add isolated Qwen 2.1 BF16 identity workflows"
git push
```

Expected image names **after the action succeeds**:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:qwen21-identity-bf16
ghcr.io/jsanso1497/runpod-comfy-stateless:qwen21-identity-upscale
```

The action also publishes commit-suffixed tags. Use one of those to retain a particular build. If placing the package in a different repository, the action automatically uses that repository's name; update `imageName` in the two Runpod JSON files accordingly. A private GHCR package needs registry credentials in Runpod, or make that image package public. Never put registry or Hugging Face tokens in GitHub files.

### Local Docker alternative

```bash
docker build --build-arg PROFILE=identity -t qwen21-identity:bf16 qwen21_identity
# Or build the optional finishing profile:
docker build --build-arg PROFILE=upscale -t qwen21-identity:upscale qwen21_identity
```

This package has not been Docker-built or GPU-benchmarked in the preparation environment. The commands are provided for execution in a Docker/GPU-capable environment.

## Runpod template settings

| Setting | Value |
|---|---|
| GPU starting target | A40 48 GB or another 48 GB-plus NVIDIA GPU |
| System RAM planning target | Prefer 96 GB or more for BF16 loading/offload and large crops; actual use is not benchmarked here |
| Container image | One of the published tags above |
| Container disk | 150 GB |
| Volume disk | 0 GB for the requested stateless configuration |
| Network volume | None |
| HTTP port | 8188 |
| TCP ports | None required |
| Docker start command | Leave blank |
| Docker entrypoint override | Leave blank |

The GPU/RAM guidance is a planning target, not a tested minimum or fit guarantee. A 48 GB card is not a promise that every 4 MP crop or SeedVR2 target fits. Start at the included 2 MP edit crop.

Environment variables:

```text
DATA_ROOT=/workspace/qwen21_identity
COMFY_PORT=8188
ASSET_DOWNLOAD_WORKERS=2
```

`HF_TOKEN` is optional and should be supplied through Runpod secrets when upstream access requires it. Public accessible files do not need it. No other environment variables from your existing multipurpose template are needed. The image profile is selected at build time, not by copying old `ENABLE_*` switches.

Two valid Runpod REST template request bodies are included:

```text
config/runpod-template-identity.json
config/runpod-template-upscale.json
```

They are **API request payloads**, not ComfyUI workflows. Use the console settings above or, after the image exists, post the selected payload yourself:

```bash
# Run from qwen21_identity/. Set RUNPOD_API_KEY privately in your own shell first.
curl --fail-with-body --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data-binary @config/runpod-template-identity.json
```

This creates a template, not a running Pod. The package does not execute this API call. A private registry setup additionally needs a valid `containerRegistryAuthId` in that request.

### First startup

The image verifies or downloads its allowlisted weights before launching ComfyUI. Port 8188 is not ready while this happens. Read the Runpod container log for download/verification progress. The downloader supports safe resumption after interrupted transfers, checks every file's SHA-256, and fails instead of quietly loading a different precision or model. It does not use repository snapshots or wildcard downloads.

On a stateless Pod, terminating or recreating the container loses local models, source uploads and outputs. Download accepted outputs before stopping or terminating the Pod. A fresh container downloads the selected assets again.

The template does not add Jupyter, SSH or a file browser. Upload sources in ComfyUI and download results from its image output. ComfyUI is not given an application login by this package. Treat the Pod's access URL as private and use your provider's access controls for sensitive photos.

## Workflows installed in ComfyUI

Open the workflow browser folder **Qwen21 Identity v1.0**. Alternatively, drag a JSON from `workflows/` into ComfyUI. These workflows need the included `qwen21_identity_tools` nodes and the pinned AusBoss version; simply dragging them into an unrelated older installation is not sufficient.

| File | Function |
|---|---|
| `00_Source_Prep_2_References.json` | Optional body/face reference normalization with the same base model. Use originals when they are already suitable. |
| `01_Body_Swap_Masked.json` | Replace physique and clothing inside the selected scene region. |
| `02_Head_Refinement_Masked.json` | Resolve face/head likeness on an accepted body result. |
| `03_Combined_Body_Then_Head.json` | One queued job with three unique source images and two separately drawn scene masks. |
| `04_Local_Repair_No_LoRA.json` | Local hands, seams, anatomy or lighting repairs without a swap adapter. |
| `05_Final_Resize_No_Extra_Model.json` | Optional 2x Lanczos enlargement, not generated detail. |
| `06_Optional_SeedVR2_FP16_7B.json` | Optional generative finish. Installed in the upscale image only. |

Matching `api_workflows/*.api.json` files are for ComfyUI API automation. They still require actual uploaded images and masks. They are not a replacement for painting a valid mask and are not submitted automatically.

## Recommended first run: body, then head

### 1. Prepare the source roles

Use a real full-body photograph for `body.png`, the clearest suitable face photograph for `face.png`, and the original high-resolution scene for `scene.png`. File names in the templates are placeholders; upload/select your own files in the Load Image nodes.

The BFS body reference includes wardrobe. Default body instructions transfer the reference clothing as well as physique. When retaining the scene's outfit is essential, change that clause explicitly and inspect the result; this is not a guaranteed wardrobe-lock adapter. Do not ask for a different body shape while simultaneously requiring the old pixel-exact silhouette.

A standing, front-facing full-body reference on a plain light square canvas is the adapter's documented preference. Keep the person large within that square with feet and hands visible. Do not stretch a portrait photo to a square. The included source-prep workflow creates a square canvas by fitting and padding rather than distorting.

The body reference is resized independently toward 0.59 MP; the face reference toward 1 MP. Multiples of 32 can move the exact pixel budget slightly. Upsizing a small source is disabled except for the small alignment rounding. These sizes describe model conditioning, not the final scene dimensions.

### 2. Run the body workflow

Open `01_Body_Swap_Masked`. Load your scene and body reference. Right-click the scene Load Image node and choose **Open in Mask Editor**. Paint on the **mask layer**, not a colored RGB drawing layer, then save/apply the mask.

White selects what may be replaced. Cover the entire original person's silhouette, plus sufficient room for the replacement body/hair, clothing and affected contact shadows. Exclude foreground objects and other people that must stay in front. A replacement wider than the old person needs extra mask room. Leave practical seam space around the subject.

The mask checker deliberately rejects an empty or wrong-sized mask. Without it, an unpainted image can appear to run but leave you with a no-op or wrongly localized result.

Edit the instruction to identify the intended person clearly, especially if more than one person appears in the context crop. The default says the person centered in the crop. Then queue one image and inspect the full-size composite and generated crop.

Two results are saved:

```text
output/Qwen21/body/...                         # historical saved output
input/qwen21_handoff/body_latest.png           # overwritten latest handoff
```

### 3. Refine the accepted head

Open `02_Head_Refinement_Masked`. Its scene loader defaults to `qwen21_handoff/body_latest.png`; refresh the image selection/browser if needed after creating the first handoff. To refine an earlier accepted body result rather than the newest one, select/upload that specific image.

Paint a NEW mask around the head, hair, ears and necessary neck transition on the accepted body result. Include enough room for the reference hairstyle. The crop contains surrounding shoulder/lighting context, but the final feathered mask determines what is actually pasted back.

Load the face reference, check the prompt's target description and queue. This pass uses only the head adapter and does not rerun the body graph.

```text
output/Qwen21/head/...
input/qwen21_handoff/head_latest.png
```

Both handoff files are normal 8-bit PNGs. The combined graph passes tensors directly between stages and avoids this intermediate file round-trip. Historical Save Image outputs remain separate; the four `*_latest.png` handoff names are intentionally replaced.

### 4. Optional repairs

Use `04_Local_Repair_No_LoRA` for a small defect. Its default reference can be replaced with whichever genuine reference shows the relevant anatomy. Write a narrow instruction naming the actual error. It saves `repair_latest.png`; explicitly select that accepted repair in a finishing workflow. The finishing workflows otherwise default to `head_latest.png`.

## Combined workflow

`03_Combined_Body_Then_Head` uses **three unique image files but four image loaders**:

- Scene A: the original scene with a full-person mask.
- Body: the physique/wardrobe reference.
- Scene B: the SAME original scene with a separate head/hair/neck mask.
- Face: the identity reference.

The head pass receives the generated body result, not Scene B's RGB image. Scene B only supplies the second mask. A consistency check rejects different scene sizes or RGB content. Memory cleanup is requested between heavy stages; it is best-effort rather than a VRAM-fit guarantee.

The final result is checked against the original scene outside the union of both actual blend footprints. If the body pass shifts the head beyond the prepainted head mask, use the separate head workflow and draw the mask on the accepted result instead. This is why the staged approach is the recommended first run.

## Quality controls and resolution

The included baseline is BF16 Qwen weights, lossless default KV cache, 40 steps, Euler/simple, CFG 1, denoise 1, fixed seeds, and adapter strength 1. Both stages start at a 2 MP crop. No acceleration/distillation adapter or quantized cache is enabled.

`Q21 Native Encode` delegates to ComfyUI's `TextEncodeQwenImage21` with `resolution=0`. References keep their independent sizes, and the sampler uses that node's matching empty latent. Do not substitute an SDXL empty latent, an arbitrary output canvas or a noise mask on an empty latent. Qwen redraws the context crop; localization here is enforced by crop-and-stitch, not native masked-latent inpainting.

The full original scene is never resized in the swap workflows. A large original remains that size. Only the edit crop is generated at approximately 2 MP. This preserves existing background detail but does not make the generated subject native 20-plus-megapixel detail. Compare a 4 MP crop only after the identity is correct; higher resolution is not automatically better, and the helper refuses crops above 4.5 MP.

AusBoss `blend_pixels` affects the compositing seam, not just the visible painted mask. The preservation audit protects pixels **outside the actual feathered blend**, not every pixel outside the hard painted outline. It checks decoded image pixels, not original JPEG bytes, EXIF, embedded profiles or archival metadata.

Useful adjustments, one at a time with fixed seeds:

| Problem | First adjustment |
|---|---|
| Different physique but limbs appear wrong | State that reference anatomy is projected into the scene's camera perspective; preserve scene contact points, not reference apparent limb lengths. |
| Hair/head is clipped by the old silhouette | Enlarge the head mask to cover the intended replacement and blend area. |
| Neck seam is visible | Include the transition in the mask; increase crop context modestly. Compare a little more blending before changing tone matching. |
| Scene/other person changed | Narrow the actual mask, identify the target clearly, inspect the blend-mask footprint. |
| Source identity drifts | Return to the clearest original reference, simplify instructions and compare fixed-seed results. Do not upscale a wrong face. |
| Qwen runs out of VRAM | Use staged workflows, keep 2 MP or reduce crop size; use cache device `cpu` while retaining dtype `default`; do not change latent wiring. |
| Body keeps changing while fixing face | Use the separate head workflow rather than queuing a new randomized body pass. |

## Optional finishing

### Lanczos

Workflow 05 enlarges both dimensions by 2. It does not synthesize new facial detail. It changes pixel sampling across the image and is unnecessary when the original-size composite is large enough.

### SeedVR2

Workflow 06 uses the optional full-precision 7B FP16 model, not the smaller or quantized editions. It is a **single-image** run with batch 1, no input or latent noise injection, no compilation, SDPA, and CPU offload. The baseline uses untiled VAE processing. If it runs out of memory, enable both VAE tiling options at 1024 pixels with 128 overlap, then consider DiT block swapping rather than replacing the model with a smaller one.

Default target: **2048 pixels on the shorter edge, capped at 4096 on the longer edge**. It is not a universal 2x switch. Skip this workflow for an already larger scene unless intentionally resizing. Keep the native result, compare the enhanced candidate against the original references, and reject changed eyelids, lip shape, jaw, hairline or distinctive marks. This optional full-frame enhancement does not preserve untouched background pixels bit-for-bit.

## Validation and maintenance

Local preparation checks passed: 31 offline tests, Python compilation, shell syntax, and seven UI/API graph pairs. No actual image was generated and no Docker image was built in the preparation environment.

The included Docker build starts ComfyUI on CPU and checks real node registration, required inputs, connection types and saved widget order. It does not download model weights or validate GPU inference/likeness. After the image is built, the first real generation on your GPU remains the acceptance test.

```bash
python -m pytest tests -q
python scripts/validate_workflows.py --profile upscale
python scripts/download_models.py --models-dir /tmp/not-used --dry-run
# Against your running Pod's local ComfyUI service:
python scripts/validate_workflows.py --profile identity --server http://127.0.0.1:8188
```

`test_helpers_and_graphs.py` requires PyTorch, NumPy, Pillow and pytest. Downloader tests need requests and pytest. GitHub's lightweight validation job runs the downloader tests and static checks; the Docker build performs the live schema check.

Code source commits are pinned in `config/upstream.json`. Model **content hashes** are pinned in `config/models.json`; download URLs resolve `main`, but altered bytes are rejected. This intentionally fails rather than silently upgrading weights. Non-Torch Python dependencies are resolved from the pinned upstream requirements at build time, so this is not a fully hermetic dependency lock. Each built image records its resolved versions in `config/built_python_packages.txt`.

See `docs/SOURCE_IMAGE_PROMPTS.md`, `docs/SOURCES.md` and `docs/VALIDATION.md` for prompts, references and verification scope. Preserve upstream licenses; this package does not modify model restrictions or add content filtering/uncensoring patches.

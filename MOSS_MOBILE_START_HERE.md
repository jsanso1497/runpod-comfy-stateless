# MOSS Mobile HQ

Prepared November 3, 2026. Additive workspace for `jsanso1497/runpod-comfy-stateless`.

**Status: source package with passing local CPU/unit/static tests. Not GPU-certified.** The supplied build action adds a real ComfyUI CPU startup/schema gate before publishing. A successful build is not a GPU image-quality test. No image has been built or pushed by the assistant.

## What this adds

Four genuine ComfyUI App Mode workflows, with selected inputs/outputs and App Mode as their initial view. This is not a third-party mobile UI and does not need a separate client app.

| App | Inputs on your phone | Result |
| --- | --- | --- |
| 01 Generate HQ | Prompt, shape, native detail size, prompt expansion toggle | Native generated PNG |
| 02 Edit HQ | Photo, working detail size, exact change instruction | Edited PNG at the original photo's dimensions |
| 03 Reference Edit HQ | Photo, second reference, working detail size, instruction | Reference-guided edit; image 1 controls canvas |
| 04 Protected Mask Edit HQ | Photo, mask, working detail size, instruction | Original decoded pixels retained wherever mask is pure black |

The existing QMOSS and QMOSS_I2I workflows are left untouched. The old img2img graph remains available for comparison; the new editing apps use a trained reference-conditioned edit adapter instead of conventional source-latent img2img. The approved Qwen/SAM workflows, main Dockerfile, task catalog and existing build actions are not modified.

## Install locally, then build

Apply the companion `apply_moss_mobile.py` from your local repository root. It verifies its payload, refuses conflicting existing files, and adds only the isolated MOSS workspace, its guide, and its new build action. It does not commit, push, or access the network.

Review changes in GitHub Desktop, commit, then click **Push origin** yourself. In GitHub, open **Actions > Build MOSS Mobile HQ > Run workflow**. This builds only the MOSS image. After it succeeds, copy the exact digest reference shown in the action summary into RunPod. Do not deploy an assumed image tag before the build has succeeded. The moving tag will be `ghcr.io/jsanso1497/runpod-comfy-stateless:moss-mobile-hq`; use the successful build's digest for reproducibility.

A private GHCR package requires RunPod registry credentials. Repository visibility does not automatically establish package visibility. Do not put GitHub tokens in workflow JSONs or prompts.

## RunPod setup

| Setting | Value |
| --- | --- |
| Suggested first validation GPU | **A40 48 GB** or **RTX A6000 48 GB** |
| GPU count | 1 |
| Host CUDA compatibility | Select a machine compatible with CUDA 12.8 |
| Host RAM | Prefer at least 48 GB |
| Container image | Exact successful build digest from the action summary |
| Container disk | 20 GB |
| Persistent storage | 60 GB to start, mounted at `/workspace` |
| Storage preference | Standard network volume for portable persistent models and images |
| HTTP ports | **8188 only** |
| TCP ports | None needed for this package |
| Container start command | Leave blank |
| Login username | `workbench` |

Required environment variable:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

Create the RunPod secret `comfy_password` as a unique password of 16-72 UTF-8 bytes. An unresolved secret or weak password stops startup. Use RunPod's **HTTPS** connection for port 8188. Do not expose the internal ComfyUI port 8189.

Optional environment variable, using your existing secret:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

The model files are public; this token is optional, not a separate inference API subscription. No CivitAI or RunPod API key is needed by the app runtime.

The first start downloads roughly **30.3 GB** of locked weights and verifies SHA-256 hashes. Models, inputs, output images and saved workflows are written under `/workspace/moss-mobile`. Later starts reuse the same files. Do not attach the same writable workspace to multiple active pods simultaneously.

At the prices checked November 3, the public pricing page lists A40 and RTX A6000 at $0.59/hour. Actual region/cloud availability and console prices can differ. A standard 60 GB network volume at $0.07/GB/month is approximately $4.20/month, excluding compute, container storage, taxes and any other charges. Network-volume availability limits GPU/datacenter choices.

A 24 GB GPU with adequate host RAM may work using native offload, but is not the default recommendation and has not been benchmarked for this all-BF16 stack. A cheaper hourly GPU is not necessarily cheaper per finished image. Do not buy an A100 80 GB solely for this package before testing the less expensive 48 GB option. No guaranteed VRAM peak, image runtime or cost per image is claimed.

## Use it on your phone

Start or deploy the Pod from the RunPod console. Open its HTTPS 8188 connection and sign in. In ComfyUI, open **Workflows > 00_MOSS_Mobile_HQ_v1**, then select one of the four apps. The files contain App Mode configuration already; there should be no need to rebuild the app UI. If ComfyUI opens the graph, choose **Enter app mode** from its workflow menu.

Enter a prompt, select/upload the required image inputs and press Run. Open the finished PNG from the result panel and use your browser's image download/share controls. Save a personal copy of a workflow to retain preferred inputs. The factory folder is copied once; your existing personal saved graphs are not overwritten.

For iPhone photos, export HEIC/ProRAW to PNG or JPEG first when the browser does not supply a compatible format. Masks must be RGB/grayscale PNG images with the same displayed orientation and dimensions as the source. White reveals the edit, black protects, grey blends. The app does not include a custom touch-painting mask editor or an automatic SAM download.

**When finished, stop the Pod in RunPod. Closing Safari, leaving the app, or unloading a model does not stop GPU billing.** This is an on-demand Pod with native ComfyUI Apps, not a scale-to-zero serverless service. No automatic idle stop is installed. Starting a stopped Pod depends on GPU availability; a network volume lets you redeploy with the same data if the previous GPU is unavailable. A new Pod can have a different HTTPS URL.

Never terminate a Pod with only a local volume disk until important files have been copied elsewhere. A network volume survives Pod termination, but it still incurs storage charges and is not a substitute for backup.

## Quality settings and what is deliberately omitted

The stack is MOSS-Image SFT BF16, the matching Qwen3.5 9B BF16 text encoder, the full FLUX.2 VAE, and the original armV5 HQ edit adapter. All four apps share these files. No separate Qwen Image diffusion model, 27B/32B prompt model, Ollama service, SAM, SeedVR2, video stack, or automatic face replacement is installed.

Editing uses the native `TextEncodeMOSSImageEdit` node with **ai-toolkit** format on both the positive and negative branches. The file is `weights/armV5/armV5.safetensors`, not the converted `armV5-moss-edit-lora` checkpoint. These formats must not be interchanged. Adapter strength is 1.0. Sampling is **50 steps, CFG 3, Euler/simple, denoise 1**, with the model's native shift 5 and SLG single layer 22. These are reference-conditioned edits, so denoise 1 is intentional, not an instruction to turn this into low-denoise img2img.

The HQ adapter's maintainer currently recommends armV5 over armV6 and reports losses from the 8-step shortcut for final/surgical edits. This package follows that recommendation rather than selecting a newer filename or a faster distilled stack. This is a community adapter recommendation, not independent proof that it beats every Qwen/FLUX/MOSS editor.

Generation preserves the current official Comfy SFT sampling recipe: **25 er_sde/simple steps, CFG 2.5, with SLG off**. It uses the full-precision model components rather than the stock INT8/W4A8 defaults. The OpenMOSS 50-step recommendation is documented for Base, so it is not treated as proof that doubling SFT generation steps improves quality. Editing and generation deliberately use different recipes. This is an adaptation of the official SFT baseline, not an unmodified official JSON or an independently established universal optimum. The existing stock QMOSS workflow remains available for comparison.

Generation offers roughly 1, 1.5 and 2 megapixels and six common aspect ratios. Width/height remain divisible by 16 and match the selected ratio exactly, so some presets sit below the nominal megapixel target. The 2 MP choice increases actual sampling resolution, not just output resizing. Batch size is one.

The optional generation-only prompt expansion reuses the required text encoder, so it adds computation but no extra model. Turn it off for an already polished prompt or lower latency. Edit instructions are never rewritten. An empty expansion, or omission of text quoted with straight double quotes, falls back to the original prompt with a warning. This guard is not a guarantee of perfect semantic fidelity or rendered typography.

## Editing dimensions, preservation and limitations

Images are proportionally resized to approximately 1 MP for editing by default, including upsampling small inputs to a useful working size. The app exposes a working-resolution control with 1, 1.5 and 2 MP choices. Higher settings increase native sampling detail and memory cost, but the adapter has not been quality-benchmarked here at those larger sizes. The secondary reference remains at about 1 MP. Bottom/right edge padding brings the canvas to a multiple of 16; no content is cropped to achieve divisibility. Padding is removed after generation and the image is resized back to the original input dimensions. Small rounding differences during resizing are unavoidable. Native model content can still move or change: matching canvas dimensions is not pixel-perfect generative alignment.

Returning a large original canvas does not mean the model generated that many native pixels. For example, at the default setting a 12 MP input is edited at about 1 MP and returned at 12 MP; choosing 2 MP changes the working resolution to about 2 MP, not 12 MP. There is no claim that this reconstructs true missing detail. For sensitive detail or geometric alignment, compare the result at 100% before using it.

Protected Mask Edit performs whole-image reference editing and then composites the candidate through your mask. It is **not** a mask-conditioned inpainting model and does not crop/inpaint/stitch a local high-resolution region. This keeps the installation small and makes the outside-mask guarantee deterministic, but edits can have seams or internal drift. Grey boundaries blend, and no invisible feathering or mask growth is added. A mask with no white/grey area or the wrong size is rejected before sampling.

The exact-preservation guarantee applies to the decoded original RGB tensor at pure-black mask pixels. Native SaveImage emits an 8-bit PNG. It does not preserve source JPEG bytes, EXIF, a raw/16-bit original, ICC metadata, or alpha transparency. Convert critical assets to the intended working color space and verify output in your production tools. PNGs normally include generation/workflow metadata; do not share them blindly when prompts are confidential.

## Validation and troubleshooting

Local delivery checks include graph/API consistency, native App Mode metadata, model locks, generation/edit sampling settings, source padding/restoration, exact black-mask preservation, grey-mask blending, early mask rejection, lazy prompt expansion, password validation, checksum detection and local installer behavior. Details are in `moss_mobile/reports/validation.json`.

The GitHub build action runs those node tests inside the image and starts actual ComfyUI in CPU mode to check required native/custom nodes and frontend pins. Runtime also checks installed node selectors and model filenames before opening the authenticated gateway. None of these substitutes for CUDA inference or viewing the result on your phone.

For the first GPU acceptance test: run one generation at Standard size, one simple single-image edit, then the reference and masked apps. Confirm that the expected instruction is followed, identity/framing remain acceptable, the downloaded image has the expected dimensions, protected regions are unchanged and no errors appear. Compare both quality and total runtime against your existing QMOSS baseline. Keep the source images and seeds to reproduce comparisons.

For missing node errors, confirm you deployed the **new MOSS image digest**, not the old Qwen container. Restart with the pinned image rather than updating ComfyUI/custom nodes at random. For missing images, upload your own photo: shipped input filenames are intentionally placeholders. For memory errors, start with a single reference at Standard size and inspect the actual memory report. Moving to a GPU with more VRAM does not require changing the precision or adapter.

The live compatibility checker can be run inside the Pod:

```bash
python /opt/moss-mobile/validate_live.py --url http://127.0.0.1:8189
```

For a forced full checksum audit, add `MOSS_VERIFY_ALL=1` to the Pod environment for one restart, then remove it. A checksum mismatch stops startup instead of substituting an unverified model. Dependencies with loose upstream version ranges are resolved at image build and recorded in `/opt/moss-mobile/build-pip-freeze.txt`; the exact Comfy commit is recorded at `/opt/comfy-source-commit.txt`. Use the final container digest to reproduce the complete built environment.

Only upload personal/client images to hosting approved for that content. Check upstream model/license terms for the intended use; local generation does not itself clear privacy, licensing or client-review requirements.

Source links and selection rationale: `moss_mobile/SOURCES.md`.

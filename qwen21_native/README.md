# Qwen 2.1 Native Suite v1.1.0

A separate, additive native-Qwen workflow suite. It does not replace the old
`qwen21_identity` folder, old custom nodes, workflows or Docker tags.

## What is included

38 editable ComfyUI workflow JSON files, 38 matching API graphs, a generator,
small safety/reference helper nodes, an explicit model allowlist, a Runpod
Dockerfile, a separate GitHub Actions workflow, an existing-Pod installer,
and tests. Native editing uses Qwen Image 2.1 plus AusBoss crop/stitch.

- Native identity profile: 34 workflows and exactly three Qwen BF16 weights.
- Upscale profile: 36 workflows; adds the SeedVR2 7B FP16 model and VAE.
- Optional BFS comparison flag: adds two comparison workflows and two adapters.
  It does not enable BFS in any native workflow.

No Flux, KREA, Minimax, Ollama, prompt-rewriter, ControlNet, segmentation-model,
face-restorer, or other generation-model download is included. The AusBoss
node package and ComfyUI still install their software dependencies.

## Start here

**Already-rendered image with a drifted face:** try `12_Rescue_Likeness_From_Originals`
with the best real face photograph, one real supporting angle, and a face/head mask.
Start from the pre-upscale result when available.

**New scene + face + body swap:** try `00_START_HERE_Scene_Face_Body`.
Image1 is the scene crop, image2 is facial identity, image3 is body proportions.
The default keeps the scene outfit. Use 03 for reference clothing or 04 for a
separate garment source.

**Maximum control:** run 05 (body with hard head protection), inspect the result,
then run 08 (head with primary face and supporting views). In the head workflow,
load your selected body result explicitly. Handoff filenames are convenient
candidates, not an automated approval system. 23 combines these operations in
one queue, but prepainted head masks cannot follow a head that moved.

**Expression from another photo of the same person:** 13 takes all identity from
the current image and expression only from image2. 14 is different: it replaces
identity from image2 and takes expression only from image3.

## Masks and reference roles

Paint on the MASK layer, not the RGB Paint Pen layer. White selects editable
pixels. These workflows feed Qwen the complete context crop; they do not feed
that mask into native sampling. AusBoss stitches the generated crop back only
through its effective feathered mask. The model therefore cannot read the
painted mask itself. Write the prompt so the intended local subject/defect is
unambiguous.

The full original scene is not resized for local editing. Crop dimensions are
rounded to multiples of 32. The native encoder runs at resolution=0 after each
reference is independently sized, and its own matching empty latent is wired
to the sampler. Denoise 1 is intentional for this empty-latent edit route; it is
not a low-denoise image-to-image reconstruction of the source latent.

The hard-protected body workflows require a second loader with the identical
original scene and a head protection mask. That mask overrides feathering and
restores the original pixels. In two-person sequential workflows, pass 2
hard-protects the effective blend footprint of pass 1. Overlapping person masks
are rejected so occlusion ownership has to be resolved explicitly.

The audit checks exact RGB tensor equality outside the effective blend mask.
It is not an identity score or a perceptual quality test. Standard PNG exports
and handoffs are 8-bit; the source file's encoding/metadata is not preserved.

Comparison sheets show the raw generated crop BEFORE stitching and hard
protection. The separate full-size candidate PNG is the composited result.

## Generation defaults

Qwen diffusion, Qwen3-VL encoder and VAE all use the supplied BF16 files.
Default: Euler, simple schedule, CFG 1, 40 steps, fixed seed, 2 MP context crop,
lossless/default KV cache. Reference images are capped near 1 MP without deliberate
upscaling. The 4 MP / 50-step head variant is an experiment, not a promised
improvement. The 1 MP variant reduces crop memory without quantizing the model.

The helper exposes eight contiguous image slots including the base crop, so up
to seven additional reference images can be connected. Update the prompt when
adding them. Empty slots in the middle and prompts naming unconnected image
slots stop with a clear error rather than silently shifting reference numbers.
Larger/more references cost memory and can introduce conflicting evidence.

## Existing dedicated Qwen Pod: no Docker rebuild needed

Use this route only when the existing Pod already has native Qwen 2.1, AusBoss
and the three BF16 model files from the old kit. Upload the ZIP to `/workspace`
and run:

```bash
cd /workspace
unzip -q Qwen21_Native_Suite_v1.1.0.zip -d qwen21-native-upgrade
python /workspace/qwen21-native-upgrade/qwen21_native/scripts/install_into_comfy.py \
  --comfy-dir /opt/ComfyUI \
  --user-dir "${DATA_ROOT:-/workspace/qwen21_identity}/user" \
  --profile identity
```

This installs the native helper and 34 workflows, leaving your current models,
core code and old workflows intact. It does not download model weights. Restart
ComfyUI using the existing service's restart mechanism, then refresh the browser.
Do not terminate a stateless Pod just to reload custom nodes. Use the actual
ComfyUI and user-directory paths when your deployment differs from the old kit.

Use `--profile upscale` only when that Pod already has SeedVR2 installed and
its two model files. Add `--include-bfs` only to expose comparison workflows;
the installer does not download those adapters either.

After restart, validate against that running server:

```bash
python /workspace/qwen21-native-upgrade/qwen21_native/scripts/validate_workflows.py \
  --profile identity --server http://127.0.0.1:8188
```

Change the profile and add `--include-bfs` to match what you installed.
Workflows appear in `Qwen21 Native v1.1.0`. Existing saved edits are not overwritten.

## New GitHub / Runpod build

Add these paths to the root of the existing repository:

```text
qwen21_native/
.github/workflows/build-qwen21-native.yml
```

Do not replace the root Dockerfile, old identity action, or old identity folder.
The included `APPLY_NATIVE_SUITE.py` can copy the two paths into a local checkout
without creating a commit or pushing anything:

```bash
python /path/to/extracted/APPLY_NATIVE_SUITE.py --repo /path/to/runpod-comfy-stateless
```

Then inspect and commit the changes. Run **Build Qwen 2.1 Native Suite** on the
new commit. Do not rerun an old failed commit expecting it to use new files.

The new action publishes these intended tags only after a successful build:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:qwen21-native-bf16
ghcr.io/jsanso1497/runpod-comfy-stateless:qwen21-native-upscale
```

Version and commit-specific tags are also emitted. `latest` and the old
`qwen21-identity-*` tags are not modified by this action.

The GitHub validation job does not import Torch. Full helper tests run inside
the Torch-equipped Docker build. A full ComfyUI server is NOT started during
BuildKit. At actual Pod startup, a supervised ComfyUI process is checked using
its real `/object_info` node schemas; errors name the workflow/node instead of
being hidden inside a Docker RUN summary. No inference is performed by that
schema check. The report is saved in `DATA_ROOT/logs/runtime_schema_check.json`.

## Runpod setup

Use HTTP port 8188, 150 GB container disk, zero persistent volume if retaining
the previous stateless design, and leave start-command/entrypoint overrides blank.
A 48 GB GPU is the intended starting target, not a tested minimum for every
multi-reference or 4 MP graph. More RAM is useful for BF16 offload. No guarantee
of fit or generation speed is made without testing your GPU and reference sizes.

```text
DATA_ROOT=/workspace/qwen21_native
COMFY_PORT=8188
ASSET_DOWNLOAD_WORKERS=2
ENABLE_BFS_COMPARISONS=0
```

HF_TOKEN is optional when upstream access requires it. The upscale profile is
chosen by Docker image, not by adding a runtime flag. The old KREA/Minimax/Ollama
environment variables are unnecessary.

With BFS comparisons off, the identity profile downloads exactly:

```text
diffusion_models/qwen_image_2.1_bf16.safetensors
text_encoders/qwen3vl_8b_bf16.safetensors
vae/qwen_image_2.1_vae_bf16.safetensors
```

Approximately 32.38 GB decimal according to the retained size estimates. The
upscale profile adds approximately 17.00 GB. These are planning estimates,
not exact disk requirements. Source commits and model SHA-256 digests are
retained from the previous kit. Downloads are allowlisted, resumed and verified.
Existing verified files are reused; mismatched files are not overwritten.

Save outputs before terminating a stateless Pod. A different Pod does not
inherit the old Pod's local files merely because DATA_ROOT has the same name.

## Comparing native Qwen with BFS

Set `ENABLE_BFS_COMPARISONS=1` before startup only when you want the two adapter
files downloaded and comparison workflows installed. Workflows 90 and 91
compare native with BFS at 0.25, 0.50, 0.75 and 1.00. Same seed, reference sizes,
prompt, crop and sampler settings are used within each graph. The adapter is
the changed factor. Every adapter branches from the unmodified base model.
No candidate overwrites a handoff or is automatically declared best.

These are controlled same-instruction comparisons, not a comparison of each
method's separately optimized prompt. Native Qwen is not guaranteed to outperform
BFS. Compare exact likeness, lighting, anatomy and texture against the real photos.

## Limits that matter

A generated reference can already contain identity drift. Producing more views
of that generated face does not provide independent evidence of the real person.
Prefer genuine photos for primary identity, useful head angles, physique and
expression. Tight clothes in a generated source image do not recover anatomy
hidden under baggy clothes in the real source. Clothing preservation and face
identity inside an edited region remain model instructions, not geometric locks.

Source preparation, one-pass two-person replacement, 4 MP refinement, and
SeedVR2 enhancement are candidate-producing variations. Keep accepted native
results separate and inspect every extra generative pass. Only the source crop
export and Lanczos finishing workflows avoid a generative model entirely.

## Validation status

See `docs/VALIDATION.md` for executed tests and limitations. Docker image build,
publishing, actual Runpod startup, and GPU visual quality are not claimed as
completed by this package. The GitHub connector could not access the
repository during preparation, so no commit or image was published here.

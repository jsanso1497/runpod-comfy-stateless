# Krea Identity 1.0.1: face / body / scene, without custom training

This is an additive module for the attached RunPod ComfyUI repository. It is
available in the general image, H3 Portrait Full, and H3 Portrait Lite after a
rebuild. It does not change the H3 models, prompts, workflows, last-frame export,
or shared user LoRA list. It does not use an external inference API.

## Build failure fixed in 1.0.1

This update corrects the CPU-only SeedVR2 node-registration failure in the
uploaded GitHub Actions log. It changes the two loader schema device-list
fallbacks only. It keeps full real-node validation, the same BF16 identity
models, the same rebalancer, and the same GPU workflow settings.

Merge this cumulative update over the repository that received the earlier
Krea update, commit all changed files, and run the build on that NEW commit.
Re-running the old failed job would use the old source and fail again.
No RunPod environment-variable changes are needed for this fix.

The workflow filenames still end in `v1_0`: image-generation graphs did not
change, so the fix does not replace your edited/saved workflow copies.

## Install the update

Extract the update ZIP and merge its CONTENTS into the repository root. Keep the
`krea_identity` directory in the root, alongside `h3_portrait` and `config`. Include
the hidden `.github/workflows` files. The update intentionally does NOT contain
`config/lora_links.txt`; leave your existing private link list alone.

On macOS, open Terminal in your existing repository directory and use a merge
extract, rather than replacing the entire `h3_portrait` folder in Finder:

```bash
ditto -x -k "$HOME/Downloads/krea-identity-integration-update-v1.0.1.zip" .
python3 tools/verify_snapshot.py
```

This overlay is built against the ZIP supplied in this conversation. If your local
source has changed since that ZIP, review the differences before applying it. Do
not delete local work to silence the source verifier.

Commit the merged files. The existing Actions watch the new module directory and
the source snapshot, so a push can start the existing image builds. A manual run
of the desired existing build Action also works. No new Action or image tag is
required. Wait for a successful build before creating the updated Pod.

The exact snapshot identifier is now:

```
h3-1.5.0-krea-identity-1.0.1-r1
```

The H3 pipeline itself remains version 1.5.0. You do not need to delete your
repository, regenerate user LoRA links, train a subject, or install nodes by hand.

## Enable in either H3 template

Keep the Full or Lite template and its existing ports/secrets. Add:

```text
ENABLE_KREA_IDENTITY=1
KREA_IDENTITY_UPSCALE=1
```

Set the second value to `0` to omit the SeedVR2 weight download and upscale
workflow. Both variables default to `0`, so an ordinary H3 Pod does not suddenly
download extra image-model weights. The small node packages are bundled either
way; weights and saved workflows are opt-in.

Full and Lite use the SAME Krea Turbo BF16 model, BF16 encoder, and full-rank
Identity Edit v1.2 weights. Lite only changes the existing H3/Ollama configuration.
All enabled H3 assets still download in these H3 templates.

Allow approximately **35 GiB for Krea editing weights**, plus **16 GiB for the
optional SeedVR2 upscaler**, in addition to existing H3/Ollama assets, image layers,
outputs and free working space. These are catalog estimates, not measured peak
usage. Already-present matching weights are reused. A newly enabled combined Pod
may need more disk than your current H3-only template. GPU memory has not been
benchmarked; BF16 does not become a lightweight model merely because the H3
template is called Lite.

## Best setup for a Pod used only for these stills

Use the repository's GENERAL image after its build succeeds:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:latest
```

Set:

```text
MODEL_PROFILES=seedvr2
ENABLE_KREA_IDENTITY=1
KREA_IDENTITY_UPSCALE=1
ENABLE_OLLAMA=0
```

This uses the existing SeedVR2 profile plus the new isolated Turbo editor catalog.
It avoids downloading the old Raw diffusion model and native H3 weights. Keep
all existing file-manager/password settings and the HTTP ports `8188,8888`.
The shared link-only LoRA library still follows its existing download behavior;
none of those LoRAs is automatically applied to these Krea graphs.

## Start with one workflow

Open `Krea_Identity_01_Face_Body_Scene_v1_0` in ComfyUI.

1. Upload a FACE reference, a BODY reference, and the SCENE/POSE photograph.
2. In `Krea Identity - Face / Body / Scene`, choose whether to keep the scene's
   clothing or use the body-reference clothing. Keep 1.5 megapixels initially.
3. Queue once and review the saved native image. The reference preview shows the
   exact original-photo sheet supplied to the editor. No reference person is
   generated or beautified before editing.

Internal order is fixed: scene = image 1; reference sheet = image 2. The face is
on the LEFT of the sheet and the body is on the RIGHT. The main result is one
image, not a video frame or a batch.

**Important limit:** this uses Identity Edit's reference-sheet capability. It is
NOT evidence that three independent reference roles will always fuse correctly.
Likeness, body transfer, pose retention and whole-scene preservation require
visual evaluation with your photographs. The stock model can normalize distinctive
facial proportions. A long prompt does not remove that limitation.

If the sheet causes two people, split compositions, or weak facial identity,
change `reference_mode` to `Face only` with `Keep scene clothing`, without
changing the scene or seed. Both
uploads are still required by the three-upload graph. To use only one subject
photograph without dummy uploads, open the simpler alternative below.

## Alternative and optional finishing workflows

| Workflow | Purpose |
| --- | --- |
| `Krea_Identity_02_Scene_One_Reference_v1_0` | Scene + ONE clean subject photograph, using the developer's two-image input arrangement. Also accepts a reference sheet you prepared yourself. |
| `Krea_Identity_03_Face_Refine_v1_0` | Optional head-region refinement on an otherwise approved image. Uses a manually chosen crop, the original face reference, and a 1 MP edit. Stitches back into the original image; pixels outside the crop are copied unchanged. |
| `Krea_Identity_04_Upscale_v1_0` | Enlarge an APPROVED still with the inherited SeedVR2 7B FP16 stack. Starts at 2x with a 4096-pixel long-edge cap. Requires `KREA_IDENTITY_UPSCALE=1`. |

The face-refine crop controls are fractions of image width/height, not pixels.
`left=0.25, top=0.0, width=0.5, height=0.5` selects the upper central half. Adjust
for the actual head location and include hair, ears, neck and surrounding context.
It is not automatic face detection. Check the crop preview and output. Stitching
preserves the outside region, not the original pixels inside the refined crop.

Upscaling is deliberately NOT attached to every generation. This avoids spending
GPU time enlarging a failed identity transfer and keeps the original output for
comparison. 4x and larger caps are available, but larger dimensions do not prove
additional authentic facial detail. Reject an upscale that changes the identity.
No extra face-restoration or beautification model is applied.

## Exact screenshot rebalancer

The installed class is `ConditioningKrea2Rebalance` from the pinned
`nova452/Rebalance-Pack`, not a renamed approximation or a different author's
adaptive node. It sits AFTER the original grounded encoders and BEFORE sampling.

There are positive and negative copies, plus one `Enable Rebalancer (A/B)` switch.
Both copies start with the screenshot settings:

```text
multiplier = 1.0
per_layer_weights = 1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

The switch starts OFF. Queue the baseline, enable it, and queue again with the
SAME fixed seed and all other settings unchanged. The switch selects the raw or
rebalanced conditioning once; it does not stack the effect twice. Keep the two
rebalancer nodes' weights matched when editing them.

**A multiplier of 1.0 does not mean neutral conditioning** when the layer weights
are non-uniform. Turn the A/B switch OFF for a true unmodified baseline. This
node changes encoder-layer contributions; it is not a face identity lock or an
upscaler. Its effect on a particular identity edit is unbenchmarked here.

## Shipped quality baseline

| Control | Value |
| --- | --- |
| Model | `krea2_turbo_bf16.safetensors` |
| Text/vision encoder | `qwen3vl_4b_bf16.safetensors`, loader type `krea2` |
| VAE | `qwen_image_vae.safetensors` |
| General editor LoRA | Full `krea2_identity_edit_v1_2.safetensors`, strength 1.0 |
| Custom subject/style LoRAs | None in these graphs |
| Output | One approximately 1.5 MP image, matching the scene aspect ratio |
| Sampler | Euler / simple, 12 steps, CFG 1.0 |
| Denoise | 1.0 on an empty output latent, NOT ordinary low-denoise img2img |
| Reference geometry | `fit`, pixel-image path, target latent connected for pre-encode |
| Scene / subject reference boost | 1.0 / 4.0 |
| Grounding | 1024; try 768 for duplicated layouts or stubborn edits |
| Rebalancer | OFF baseline; exact screenshot settings available |
| Upscale | Optional, separate, 2x first, zero added input/latent noise |

The work canvas is aligned to the model's 16-pixel grid by a very small center
crop followed by proportional resizing. Subject photographs are fitted without
stretching; the sheet uses neutral letterboxing. Image metadata and file type
alone cannot establish likeness. Compare eye spacing, nose/jaw shape, face
length, body proportions and lighting, not just skin sharpness.

12 steps and BF16 are a conservative quality-oriented starting point, not proof
of a universal optimum. Try 8 or 10 steps to compare composition after the supplied
baseline works. Stronger reference boosts or larger canvases can make edits worse.
Do not exceed 2 MP for this editing stage; upscale an approved result instead.

## What remains to be tested

Local Python/tensor tests, graph links, shell syntax, source inventories and
repository preflight pass. A real ComfyUI node-schema check is added to each
Docker build. That check was NOT executed in the preparation environment, which
has neither Docker nor a GPU. GitHub Actions, RunPod startup, GPU memory use,
identity quality and rebalancer A/B comparisons still need an actual deployment.

The model files remain subject to their upstream licenses. Review the Krea model
license for any organizational or client deployment. See `SOURCES.md` for pins
and primary documentation.

# Krea Identity 1.1 additions

Use `krea_identity/MULTI_REFERENCE_GUIDE.md` for labeled additional references,
text-directed target selection, man/woman single-pass replacement, and protected
separate-identity editing. Workflows 01-04 and all H3 inference graphs are retained.
Rebuild the General/Image image from the new commit; the RunPod settings and model
weights remain the same. The update is a cumulative overlay, not a folder replacement.

---

# Added: Krea Identity still images

Start with `krea_identity/START_HERE.md` for the new face/body/scene editor,
optional exact screenshot rebalancer, and single-image upscale workflows.
The module is available in all three existing builds after rebuilding.
Add `ENABLE_KREA_IDENTITY=1` and optionally `KREA_IDENTITY_UPSCALE=1` to the Pod.
No custom subject LoRA training is required. Existing H3 workflows are preserved.

The new source snapshot ID is `h3-1.5.0-krea-identity-1.1.0-r1`.
The instructions below describe the preserved H3 1.5 release; its pipeline version
has not changed. Use the current Krea integration update ZIP for this update.

---

# H3 Portrait 1.5: role-aware video, MiniMax H3 Ref2VA stills, saved last frame

This is an in-place browser update. **Do not wipe the repository again.**
Use the `h3-portrait-1.5-ref2va-still-update.zip` package for the existing repo.
The update ZIP intentionally does **not** contain `config/lora_links.txt`, so the
existing private LoRA list is not replaced.

The complete backup ZIP contains a neutral placeholder at that path and is meant
for recovery/new-repo use, not as a blind replacement for a populated live repo.

## What changed

- Automatic H3 video now supports explicit per-image roles. In the recommended
  video routing, pose/camera and expression guides are analyzed by Ollama but
  their pixels are not sent to native H3, reducing guide-person identity pull.
- The ComfyUI workflow-default bug is fixed. The serialized seed companion
  (`control after generate`) is present, so values after seed no longer shift to
  `0`, `1`, or `NaN`.
- H3 video export saves a matching PNG of the actual final decoded MP4 frame for
  later chaining experiments.
- A separate Lite container uses full-architecture INT8 MiniMax H3 plus one 8B
  Instruct Ollama helper for fast workflow testing.
- Lite adds `H3_Portrait_Image_Lite_v1_5`, which creates **one finished image
  using MiniMax H3 Ref2VA itself**. There is no separate Qwen Image Edit model.
- Both Full and Lite still include the standard manual native H3 Ref2VA workflow.

## Upload through GitHub in the browser

1. Extract `h3-portrait-1.5-ref2va-still-update.zip`.
2. Open `jsanso1497/runpod-comfy-stateless` in github.dev on `main`.
3. Drag the extracted **contents** into the repository root, merging folders and
   replacing matching files. Do not upload the ZIP or an enclosing folder.
4. Leave your existing `config/lora_links.txt` untouched.
5. Verify these exact hidden paths exist after upload:

```text
.github/workflows/build-h3-portrait.yml
.github/workflows/build-h3-portrait-lite.yml
```

If Finder/browser upload flattens `.github`, manually create/replace those exact
paths with the standalone YAMLs supplied beside the release ZIP.

6. Commit all changes to `main`.

Expected early checks:

```text
CLEAN SOURCE SNAPSHOT VERIFIED: h3-1.5.0-krea-identity-1.1.0-r1
H3 PORTRAIT SOURCE VERIFIED: 1.5.0 | role-routed-reference-still-v5
```

## Build the Lite testing image

For the new single-image workflow, run:

```text
Actions -> Build H3 Portrait Lite template -> Run workflow -> main
```

The separate Full action is only needed when you also want to refresh Full.
The general `:latest` image is independent.

## RunPod Lite template

Use a separate template, for example `H3 Portrait Lite - Testing`.

| Field | Value |
| --- | --- |
| Container image | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-lite` |
| Container disk | `300 GB` |
| Volume disk | `0 GB` |
| Network volume | None |
| HTTP ports | `8188,8888` |
| TCP ports | blank |
| Start command | blank |
| Registry | existing GitHub GHCR credential |

Environment variables:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
FILEBROWSER_PORT=8888
FILEBROWSER_PASSWORD={{ RUNPOD_SECRET_filebrowser_password }}
```

Use the actual secret names already configured in RunPod if they differ. The file
manager password must be at least 12 characters.

## Installed workflows

Full:

```text
H3_Portrait_Full_v1_5
H3_Ref2VA_Standard_Full_v1_5
```

Lite:

```text
H3_Portrait_Lite_v1_5
H3_Ref2VA_Standard_Lite_v1_5
H3_Portrait_Image_Lite_v1_5
```

## New Lite single-image workflow

Open `H3_Portrait_Image_Lite_v1_5`.

A typical identity/composition swap is:

```text
Image 1 -> Face only
Image 2 -> Identity or Body/proportions only
Image 3 -> Pose/camera only OR Scene/environment only
```

Keep:

```text
reference_mode: Still safe swap (pose/scene text-only)
mode: Draft only
```

Ollama sees and analyzes every reference. In safe-swap mode, pose/camera,
expression, and scene-guide images are converted into textual geometry/environment
instructions and are not sent as native H3 picture latents. Subject/appearance
references remain visual H3 references. This is designed to prevent an unrelated
guide person from becoming the generated subject.

Review the routing and generated H3 prompt, then switch only:

```text
mode: Generate image
```

The workflow is a true MiniMax H3 Ref2VA path. H3 renders its native minimum
five-frame packet, the video VAE decodes those frames, and `H3PortraitStillOutput`
selects one frame. `SaveImage` writes only that one result.

Resolution choices:

- `High-res (~2 MP, experimental)` - **shipped default for this requested high-resolution still workflow**. It asks H3 itself to sample a larger canvas.
  This is **not** a Lanczos/post upscale; it can require materially more VRAM and is not guaranteed to improve every face/reference set. If it OOMs or looks less stable, use `Native detail (~1 MP)`, which stays near H3's ordinary native-detail area.
- `Preview (~0.5 MP)` - lower-cost composition/prompt testing.

Quality choices:

```text
High fidelity -> 20 steps, ref_image_size=max
Standard      -> 16 steps, ref_image_size=match
Fast preview  -> 12 steps, ref_image_size=match
sampler       -> res_multistep
scheduler     -> simple
```

`Match guide/primary` uses an assigned pose/scene guide for output aspect when one
exists; otherwise it uses the first visual subject reference. Fixed `9:16`, `2:3`,
`4:5`, `1:1`, and `16:9` are also available.

`All images visual (comparison)` deliberately sends pose/scene pixels to H3 as
normal pictures. It may strengthen composition but reintroduces the identity
competition this release is designed to avoid.

## Video workflow behavior

For the automatic video workflow, keep `Role-aware (recommended)`. Pose/camera and
expression guides are text-only; a scene reference remains visual because the
video may need the scene's direct visual appearance. Start in `Draft only`, review
the routing, then generate.

Each generated MP4 also produces a matching `_last.png` decoded from the finished
video file. That prepares a real continuation frame for later chaining; it does
not yet guarantee a seamless next clip by itself.

## Verification boundary

Local unit/static/source checks are documented in `VALIDATION.md`. The final
Docker images and real GPU H3 inference still need GitHub Actions + RunPod testing.
Visual identity quality, exact VRAM use, and compatibility of any private LoRA are
empirical and are not claimed by the local checks.

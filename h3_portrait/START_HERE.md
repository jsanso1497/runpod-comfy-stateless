# H3 Portrait 1.1 with shared link-only LoRAs

Use this ZIP instead of the preceding H3 Portrait batch. Follow
LORA_LINKS_START_HERE.md to paste your LoRA URLs and upload ALL files to GitHub.

This is the same reference-image -> plain-English direction -> local Ollama ->
MiniMax H3 portrait workflow. It adds the shared library and explicit LoRA selection.
No OmniNode, RefMod, required reference-text grammar, or manual prompt copying.

## Build

Use the same repository: `jsanso1497/runpod-comfy-stateless`.
Upload/replace h3_portrait/, shared_loras/, config/lora_links.txt, the provided root
Dockerfile, scripts/bootstrap.sh and .github/workflows/build-h3-portrait.yml.
Do not replace your current build-image.yml, .dockerignore or build-check fixes.
Wait for Build H3 Portrait template to publish h3-portrait. The existing build
publishes latest separately with the same shared list.

## New portrait template settings

Keep these from the previous batch, or set them when creating the new template:

| Setting | Value |
| --- | --- |
| Template name | H3 Portrait - Automatic |
| Container image | ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait |
| Container disk | 300 GB |
| Volume disk | 0 GB |
| Network volume | None |
| HTTP ports | 8188 |
| TCP ports | Blank |
| Start command | Blank |
| Registry credential | Your existing GitHub GHCR credential |

Use the same GPU type already running your full H3 workflow. This is not a smaller
H3 checkpoint. The base model and encoder remain the full BF16 files.

Use these custom environment variables; preserve your existing secret names:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
```

Do not add MODEL_PROFILES, CONFIG_REPO, CONFIG_REF, COMFY_ARGS, RESTORE_* or a prior
OLLAMA_MODEL_DIGEST to this portrait template. Do not expose port 11434.
Do not alter those unrelated settings on the OLD template for this patch.

## Run

After the new build is green, deploy a FRESH Pod. Open HTTP 8188 after:

```text
H3 PORTRAIT 1.1 | Shared link-only LoRAs
H3 PORTRAIT ASSETS READY
H3 PORTRAIT SCHEMA PASS
H3 PORTRAIT OLLAMA READY
```

The first is the beginning of the startup banner. The last messages can appear in
a different order. Also check SHARED LORA LIBRARY for failed link downloads.

Open H3_Portrait_Auto. Use the updated JSON from this package, not an older saved
copy of the workflow without the LoRA selectors.

1. In **1. Upload references**, upload 1-9 images and check their numbered order.
2. In **2. Tell Ollama what you want**, write your instruction normally. State which
   images show the same person and which image supplies the outfit when important.
3. Select your compatible full-H3 Ref2VA adapter in `lora_1`; set `strength_1` as the
   author recommends. Leave `lora_2=(none)` unless using a second compatible adapter.
   Keep `use_loras=true`. No LoRA is applied when both slots are `(none)`.
4. Civitai triggers, when published, are inserted for your selected adapters. Add
   missing HF trigger words in `extra_trigger_words`. No triggers are inferred.
5. Start with aspect `9:16` or `2:3`, quality `Standard`, seconds `5`, seed `42`,
   mode `Generate video`. Click Run.
6. Review **What Ollama wrote** and **3. YOUR VIDEO**. Download your result.

Example direction:

```text
Images 1 through 5 are the same woman. Image 1 is the main face reference.
Use images 2 and 3 for body proportions and the outfit in image 4.
Image 5 is another view of her, not another person.
Have her walk slowly through a hotel lobby and look toward the camera.
Preserve her identity, age, proportions and outfit. One continuous full-body
shot. No additional people, dialogue, music or on-screen text.
```

Ollama sees every selected photo, writes the prompt and unloads before H3 loads.
Only the LoRAs you selected are applied; the entire downloaded library is not
stacked automatically. A failed/unknown key mapping stops rather than silently
ignoring an incompatible adapter.

## Presets (unchanged)

| Preset | Steps | Reference sizing | 9:16 output | 2:3 output |
| --- | --- | --- | --- | --- |
| Standard | 20 | match | 756 x 1344 | 768 x 1152 |
| Preview | 12 | match | 576 x 1024 | 576 x 864 |
| High fidelity | 25 | max | 756 x 1344 | 768 x 1152 |

No Turbo/Lightning adapter or generative upscale is automatically added. Full-size
9:16 crops six pixels from each side of the native canvas rather than stretching.
Five seconds produces 124 frames at 24 fps, approximately 5.17 seconds.

Your portrait template downloads four H3 components, its 8B Ollama helper, and
ALL LoRA links. It does not download the Krea/SeedVR2 BASE weights. The existing
latest templates keep their configured model profiles but get the same LoRA list.

Save results before terminating a stateless Pod. Check the supplied validation
report: code tests are not evidence of measured GPU speed or identity fidelity.

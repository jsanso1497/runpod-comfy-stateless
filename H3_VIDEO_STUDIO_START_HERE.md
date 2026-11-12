# H3 Video Still Studio

Prepared November 11, 2026 for `jsanso1497/runpod-comfy-stateless`.

A separate RunPod image with one shared workflow interface for MiniMax H3 video-model stills, Eros Beta 7 Fix1, and P-H3 Image. Select one engine per run. This does not load three models simultaneously and does not turn P-H3 into a native video model.

## Before deployment

MiniMax's published H3 community license excludes the United States, European Union, United Kingdom and South Korea from its standard grant. Separate authorization is needed for deployment in those territories. The package does not grant that authorization. Read `h3_video_studio/LICENSE_NOTES.md`. The downloader requires `H3V_LICENSE_CONFIRMED=1` after you have confirmed an applicable right to use the models.

This package contains code and workflow JSON only. No model weights or credentials are included.

## Apply locally

Download `update_h3_video_studio.py` to Downloads. Open a terminal in the repository and run:

```sh
python3 "$HOME/Downloads/update_h3_video_studio.py"
```

The installer adds a new folder, a new manual GitHub Action, and this guide. It does not modify your existing H3/Qwen workspace files, stage changes, commit, contact GitHub, or push. It refuses conflicting files and supports `--check` and `--rollback`.

Review the changes in GitHub Desktop, commit, and push there. Then open Actions, choose **Build H3 Video Still Studio**, and run it. Use the exact image digest from the successful run's summary or `image-reference.txt` artifact. No new image has been built or published by the author of this package.

The action resolves your existing known-good Studio base, builds the new image separately, checks model metadata, checks native CPU contracts and video preprocessing, and runs the inherited browser/login regression before publishing the new image tag. It does not update your original Studio or Lab image tags.

## RunPod configuration

Use the settings in `h3_video_studio/RUNPOD_SETTINGS.md`. The target is **one A100 SXM 80 GB**, not A100 40 GB. INT8 HQ needs less host RAM and disk. Full precision is available for quality comparisons, with managed CPU offloading rather than `--highvram`.

Open HTTP port 8188 in its own HTTPS browser tab for ComfyUI. Port 8888 is the existing authenticated Files/Jupyter service. Do not expose the internal ports 8189 or 8889. Leave Docker entrypoint and start command overrides blank.

Factory workflows appear in:

```text
Workflows > 00_H3_Video_Still_Studio
```

Existing Studio v1/v2 and Lab workflow folders are also retained. The new image changes the shared ComfyUI core inside that image only. Your original image remains the rollback path. Do not operate two Pods against the same writable workspace simultaneously.

## Workflows

| File | Starting setup |
|---|---|
| 00_H3_Unified_Video_To_Still | Video only, native MiniMax H3 reference, one still output |
| 01_H3_Unified_Video_And_Photos | Video plus face/body photographs |
| 02_H3_Unified_Replace_Person | Target scene, face/body references, optional video, painted-mask background protection |
| 03_H3_Unified_Text_To_Image | No media, generation from your prompt |
| 04_H3_Unified_Single_Photo_Edit | One-photo edit |
| 05_H3_Unified_Photo_References | Multiple-photo generation |
| 06_H3_Unified_Keyframe_Comparison | Same extracted video frames for all three engines |
| 07_H3_Reference_Preflight_No_Generation | Preview actual reference inputs and prompt mapping without loading diffusion weights |
| 08_P_H3_Video_Keyframes_To_Still | P-H3 selected, four source-file keyframes, 4 MP output |
| 09_Eros_Fix1_Video_To_Still | Eros Fix1 selected, native video reference, eight-step recipe |

Every generation preset retains the same model selector, precision selector, optional photo/video inputs, and two LoRA slots. The separate files are starting presets, not separate incompatible environments.

Factory graphs intentionally contain no personal source images. Upload the references required by the included prompt before queuing. A prompt that names an unavailable reference stops with a clear error rather than guessing.

## A. Generate a person from a video

Open preset 00. Upload a video in the video node, select a section with the intended person clearly visible, and start with a 3.1-second section at the default 1 MP native-video cap. Crop margins are percentages removed from the original video; crop is applied before scaling. The workflow does not track people automatically.

Leave photo slots empty. Use `References` mode, not `Edit`. Set the desired output composition in your literal prompt. Start at 2.5 MP and keep the seed fixed. The initial encoder is the full 32B INT8 H3 multimodal encoder, not a prompt enhancer.

For native MiniMax/Eros video references, the clip is normalized to 24 fps and its tail is trimmed to the supported `17*n+5` frame grid. The output remains one image. Audio is ignored. Longer native clips can exceed the decoded-memory guard; duration and video resolution must be balanced.

Switching to P-H3 uses four independent image references decoded directly from the original video, not from the smaller native-video tensor. You can specify exact timestamps, such as `0.2, 1.0, 2.4`, in seconds relative to your selected start. This replaces the uniform sampling count. Preview the inputs with preset 07 before generation when checking timing or crops.

The workflow does not train a persistent identity. It conditions that generation on the supplied evidence. It cannot guarantee an exact likeness or recover details that were not visible in your sources.

## B. Generate a person from video plus photos

Open preset 01. Its prompt assumes photo slot 1 is the face/hair reference and slot 2 is the physique/body reference. Upload those photos and your video. Change the prompt when using a different arrangement.

Active photo slots are numbered consecutively in slot order. For example, if only slots 2 and 3 are enabled, they become `<Picture 1>` and `<Picture 2>`. Video keyframes are appended after the active photos. The reference report lists the exact numbering.

For native H3/Eros, reference tags are `<Picture 1>`, `<Picture 2>` and `<Video 1>`. For image-only P-H3, the workflow translates `<Video 1>` into the corresponding appended Picture tags. It does not use an LLM to rewrite your prompt.

P-H3 allows **eight total image references**, including the target photo, all identity photos and all video keyframes. Two photos plus four keyframes is six references. Three photos plus six keyframes is nine and is rejected. The workflow never silently drops evidence. H3/Eros accept up to nine image references plus one native video in this interface.

## Person replacement and unchanged backgrounds

Preset 02 uses the standard Load Image node for the target. Open its Mask Editor and paint the region that may change. White is the replacement area; black is preserved. Include the whole original person and enough space for the replacement silhouette.

With `protect_background=true`, the generated candidate is resized to the original target dimensions and composited over the original. Pixels outside a zero-valued mask remain exactly the original tensor values. This is final compositing, **not masked diffusion/inpainting**. Feathered boundaries blend. An undersized mask can leave parts of the original person visible. Background protection takes priority over any prompt request to change content outside the mask, including the standard text-removal footer.

Without a mask, disable protection. Prompt-only preservation is not a guarantee of pixel-identical surroundings. Target aspect ratio is used for Edit and Replace person, rather than the generation aspect selector. Protected output is restored to the original target's dimensions; output MP controls the intermediate generation.

The retained Lab workflows provide SAM mask preparation, reference crops, contact sheets and RefMods. SAM and optional experimental adapters still use the existing Lab manager and are not downloaded automatically by this update. The new unified path keeps Spark, Spectrum, sparse attention and turbo-stack experiments off.

## Engine recipes

| Engine | Default recipe | Reference route | Decoder |
|---|---|---|---|
| Standard MiniMax H3 | 50 steps, ER-SDE/simple, no added Turbo | FL2VA for no-media/single-photo; REF2VA for multiple references or video | Fizgig single-frame H3 decode |
| Eros Beta 7 Fix1 | 8 steps, Euler/simple, 12/3 sigma shifts; configurable 6-12 steps | Native H3 reference conditioning | Fizgig single-frame H3 decode |
| P-H3 Image | 8 steps, Euler, publisher's manual sigma grid, 1/1 shifts, no CFG | Images only; video becomes keyframes | Matching P-H3 image VAE |

Eros's Turbo is already merged. Do not stack another speed LoRA on it or set it to 50 steps expecting an unaccelerated model. The selected Fix1 HQ file is the newest published INT8 Fix1 file verified during preparation. The author's mention of a future Sulphur-based v1.0 is not treated as an available checkpoint.

P-H3 uses its own matching encoder and VAE, not the H3 32B encoder and video VAE. Its native conditioning automatically prefixes short image-edit instructions with `Edit <Picture 1>.`; the audit makes that prefix visible. Prompts starting with `subject_definitions:` avoid that native prefix, which is useful for new compositions based on references. No prompt upsampler is downloaded or executed.

## Precision and fidelity

| Precision setting | H3 / Eros | P-H3 |
|---|---|---|
| INT8 HQ, default | INT8 diffusion, 32B INT8 encoder, FP16 video VAE | Matching INT8 diffusion, encoder and VAE |
| BF16 diffusion | Publisher's 16-bit diffusion file, INT8 encoder, FP16 video VAE | BF16 diffusion, matching INT8 encoder and VAE |
| BF16 full | Publisher's 16-bit diffusion file, 32B BF16 encoder, FP16 video VAE | BF16 diffusion, matching BF16 encoder and VAE |

The Eros unquantized file follows its publisher's stored dtype. Native ComfyUI chooses compute dtypes and offloads components between stages. Full precision removes more quantization but is not a guarantee of better likeness for a particular seed. It also consumes substantially more host RAM and storage. Do not force all components to remain resident in VRAM.

HQ additions are approximately 84.7 GB beyond the retained legacy assets. Optional BF16 additions are approximately 227.4 GB more. Retaining the older Studio models deliberately preserves those workflows, including their original encoder/VAE. None of those encoders is a prompt enhancer.

Set `H3V_PRELOAD=all` on the Pod to download HQ and BF16 assets at first boot, or use `hq` for the smaller initial setup. To add BF16 later in Jupyter Terminal, after confirming the applicable license:

```sh
H3V_LICENSE_CONFIRMED=1 python3 /opt/h3-still/video_studio/manage.py download bf16
```

Then choose the desired precision in the workflow. Missing weights never trigger a silent downgrade.

For H3/Eros native video, start with 2.5 MP, a short clean clip, and one or two strong photographs. Test 4 MP next. Up to 8 MP is exposed as an experimental setting, not an A100 guarantee. More reference frames, larger references, LoRAs and full precision all change the memory ceiling. P-H3's publisher recommends 4 or 8 MP for edits and 8 MP for text generation; its published memory measurements are not measurements of this complete workspace.

## LoRAs

Put model-compatible `.safetensors` files in:

```text
/workspace/h3-still/models/loras/
```

Refresh ComfyUI. Enable either or both slots, select a file, set the strength recommended by its author, and identify its actual family. The slots use native model-only bypass application so the base quantized weights are not overwritten. Zero matching diffusion keys is an error; matched keys alone do not prove visual or training-family compatibility.

H3 video/Eros LoRAs and P-H3 image LoRAs are not assumed interchangeable. Test the chosen adapter on its intended model. Qwen, Krea, SDXL and Flux LoRAs are not generally compatible. Keep both slots off for the first baseline comparison. Turbo and P-H3 conversion adapters are excluded from these identity/style slots.

## Quality comparison procedure

First use preset 06 with `Keyframes for all engines`, the same selected frames, same output size and no LoRAs. Keep the seed fixed, although identical seeds across different models do not imply identical latent noise behavior or equivalent compositions. Compare identity, proportions, lighting, skin detail and artifacts at the same viewing scale.

Then compare native video against the same keyframes on H3/Eros. Finally compare INT8 HQ against full precision on the selected engine. Change one variable at a time. Do not add an upscaler until likeness is satisfactory; an upscaler is not an identity correction.

## Validation status and diagnostics

Local checks performed while preparing this package: Python compilation, workflow graph/link validation, family routing, prompt-tag mapping, reference limits, exact zero-mask preservation, source-resolution frame extraction and actual FFmpeg crop/video decoding on synthetic input. Full validation details are in `h3_video_studio/VALIDATION.md`.

Not performed here: Docker build, the GitHub Action, browser testing against this new image, native ComfyUI tests in the assembled image, model weight downloads, CUDA inference, LoRA visual validation or A100 peak-VRAM measurement. The action is configured to run build/CPU/browser/metadata checks before it publishes, but those checks have not yet run for this package.

After deployment, this optional low-resolution pipeline test creates a tiny synthetic video and generates one image per engine:

```sh
python3 /opt/h3-still/video_studio/gpu_smoke.py
```

It is a functional smoke test, not a likeness benchmark. Outputs and reports are in `output/H3_Video_Studio`. Each real generation records the model filenames, prompt, precision, reference mapping, timings, GPU name, peak allocated/reserved memory and LoRA warnings. The normal PNG retains ComfyUI workflow metadata.

For memory exhaustion, reduce reference duration/count/resolution or output MP, or use INT8 HQ. The program does not silently enable a lower-quality mode. Read the Pod logs for model download/license/driver problems; authentication and file access remain independently available.

## Rollback

Keep your current working Pod/template until this one passes a generation test. The original image tags are unchanged. Stop the new Pod before reusing the same writable workspace with the old image. New factories and extra cached weights can remain on disk; the old image ignores the new workflow nodes unless you explicitly open those graphs.

To remove only this unmodified local source addition before committing:

```sh
python3 "$HOME/Downloads/update_h3_video_studio.py" --rollback
```

The installer refuses to delete files you have edited. It never pushes changes.

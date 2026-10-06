# H3 Portrait 1.4: role-aware references, saved last frame, Full and Lite

This release is for James's existing `jsanso1497/runpod-comfy-stateless` repository.
Do NOT wipe the repository again. Do NOT delete the existing GHCR package/base images.
No local Terminal or GitHub Desktop is required.

## 1. Update GitHub using the UPDATE ZIP

Use `h3-portrait-1.4-update.zip` for your existing repository. It contains only changed/new
files, with their correct repository paths. It does NOT contain `config/lora_links.txt`.
Your list stays exactly where it is. Do not add earlier patches afterward.

1. Extract the UPDATE ZIP in Finder. Command + Shift + . reveals hidden files.
2. Open your repository on main in github.dev (press the period key on its GitHub Code page).
3. Drag the extracted CONTENTS into the repository root, merging folders and replacing
   matching files. Do not add an enclosing folder. Include the new profiles, node modules,
   JavaScript, tests, tools, updated root documentation and SOURCE_SNAPSHOT.json.
4. Because hidden-folder drag/upload caused trouble before, explicitly check these TWO paths:

```
.github/workflows/build-h3-portrait.yml
.github/workflows/build-h3-portrait-lite.yml
```

Replace the existing full workflow with the included version. Create the new Lite workflow
with the exact path above. In normal GitHub use Add file > Create new file and enter the
whole path; paste the raw YAML file contents. Do not put either YAML at the repository root.
The existing general `build-image.yml` stays unchanged.

5. Stage all changes and Commit & Push to main. Do not build a partially uploaded commit.
   Suggested commit: `H3 Portrait 1.4 roles, final-frame export and Lite profile`.

A COMPLETE repository ZIP is also supplied as a backup. Its LoRA list contains comments only.
Do not overwrite your populated list with that placeholder. You only need one upload method.

## 2. Build Lite first for your testing

On GitHub select:

```
Actions > Build H3 Portrait Lite template > Run workflow > main > Run workflow
```

Watch the run for the final upload commit, not an old run's Re-run button.
The early checks must show:

```
CLEAN SOURCE SNAPSHOT VERIFIED: h3-role-routing-1.4.0-r1
H3 PORTRAIT SOURCE VERIFIED: 1.4.0 | role-routed-reference-v4
H3 FRONTEND SERIALIZATION PASS
```

Inside Docker both profiles test the source, start a real CPU ComfyUI instance, check node
schemas and execute a tiny native CreateVideo -> MP4 export -> last-frame PNG job. Expected:

```
H3 EXPORT CPU SMOKE PASS
H3 PORTRAIT SCHEMA PASS
```

The Lite summary ends with `H3 Portrait 1.4 LITE + Role Routing + Last Frame image published`.
Full is a separate build: `Build H3 Portrait template`. It can run independently; its summary
says FULL instead of LITE. Your general `Build RunPod ComfyUI image` is not required for either.
The code, tests and file manager are shared, but the two portrait image tags are independent.

## 3. Create a separate RunPod Lite template

Keep your current Full template. In RunPod's Templates page choose New Template and enter:

| Field | Lite value |
| --- | --- |
| Name | H3 Portrait Lite - Testing |
| Container image | ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-lite |
| Container disk | 300 GB, same allocation as your current template |
| Volume disk | 0 GB |
| Network volume | None |
| HTTP ports | 8188,8888 |
| TCP ports | Blank |
| Container start command | Blank |
| Registry credential | Your existing GitHub GHCR credential |

Use the same GPU initially to compare behavior. This release does not certify a lower VRAM
minimum. Lite downloads less and uses a smaller canvas/helper; inherited CUDA/Comfy Docker
layers are not dramatically smaller. Keep disk headroom for your personal LoRA collection.

Enter these CUSTOM environment variables, one key/value pair per line:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
FILEBROWSER_PORT=8888
FILEBROWSER_PASSWORD={{ RUNPOD_SECRET_filebrowser_password }}
```

Keep your actual existing secret references if their names differ. The file-browser password
must contain at least 12 characters. The secret value is the password itself, not the braces.
**Do not carry the 32B Thinking OLLAMA_MODEL value into Lite.** The image's internal profile
chooses its H3 weights and render presets; no extra profile/hash setting is required.
Do not expose Ollama port 11434. Do not add CONFIG_REPO, MODEL_PROFILES or legacy overrides.

Your Full template keeps its existing image:

```
ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean
```

Its OLLAMA_MODEL remains `huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M`.
Keep Full's other settings unchanged. Its updated image becomes available after the Full build.

## 4. Deploy a NEW Pod after the corresponding build succeeds

Save current outputs/references before stopping or terminating any old Pod. Your `/workspace`
is temporary in this no-volume configuration. An existing running Pod does not receive this
release when GitHub changes. Do not try loading the new workflow into old 1.3 node code.

Confirm the new Pod prints:

```
H3 PORTRAIT 1.4 | Role-aware references + saved last frame | profile=lite
H3 PORTRAIT RELEASE VERIFIED: 1.4.0 | source=<the Git commit you built>
```

Full says `profile=full`. Compare the source commit with the successful Actions summary.
Wait for ASSETS READY, SCHEMA PASS and OLLAMA READY before generating.

Port 8188 is ComfyUI; 8888 is your password-protected Jupyter file manager.
No new file-manager password is needed if your existing secret is correct.

## 5. Open the NEW versioned workflow

Lite installs:

```
H3_Portrait_Lite_v1_4
H3_Ref2VA_Standard_Lite_v1_4
```

Full installs:

```
H3_Portrait_Full_v1_4
H3_Ref2VA_Standard_Full_v1_4
```

Choose the new name, not an old saved `H3_Portrait_Auto`. Existing saved workflows are retained
rather than overwritten. An additional migration repairs the known 1.3 seed-widget shift,
but the versioned graph is the recommended starting point.

## 6. Your three-reference test

Open H3_Portrait_Lite_v1_4. Upload your three pictures in the intended order. Use the new
per-image role dropdowns in the uploader:

| Image | Role |
| --- | --- |
| 1 | Face only |
| 2 | Body/proportions only |
| 3 | Pose/camera only (text guide) |

Use Subject identity instead of Face only when that picture should also define hair or other
appearance, and explain which attributes matter in your brief. A body-only role does not
choose wardrobe or imply undress. State your target outfit in the brief when it matters.
Replace the example brief with your request. Keep `reference_mode=Role-aware (recommended)`.

First-run controls:

```text
aspect: 9:16
quality: Standard
seconds: 5
seed: 42
control after generate: fixed
mode: Draft only
prompt_variation: 0
lora_1: (none), or select your compatible H3 LoRA
strength_1: 1.00, or the author's intended strength
lora_2: (none)
strength_2: 0.00
```

Do one no-LoRA draft/video baseline by leaving both selectors at (none). Then deliberately
select your LoRA for comparison. Downloading a file does not apply it. Your FP32 file does
not require an FP32 base model; the LoRA still must target the full Ref2VA architecture.

The draft preview must show:

```
Image 1: face -> <Picture 1>
Image 2: body -> <Picture 2>
Image 3: pose_camera -> text only
```

Ollama examines all three. H3 receives the original pixels of images 1 and 2, plus text
geometry extracted from image 3. Image 3's person can no longer compete through raw image
conditioning. Its exact pose/camera geometry is not guaranteed: this is text guidance, not
ControlNet or a hard pose lock. Other appearance roles remain visual, with textual scope.

An analysis clarification gets one bounded review against explicit role selections. If the
question remains unresolved, the job still stops; it is not silently ignored.

Review the prompt and geometry, then change only mode to Generate video. The same draft is
reused. Change prompt_variation for another draft; change seed for another video realization.
All images visual (comparison) deliberately restores the old raw-image behavior for A/B tests.
It may reproduce identity leakage; it is not the recommended default for a different person.

## 7. Last-frame output

Both supplied graphs end with:

```
CreateVideo -> Save exported H3 video -> Save last frame of exported video
```

The export is H.264 MP4 at CRF 18, preserving the generated audio. The next node decodes the
completed MP4 in display order and saves the exact final decoded RGB frame as a PNG:

```
ComfyUI/output/H3_Portrait/video_<unique suffix>.mp4
ComfyUI/output/H3_Portrait/video_<unique suffix>_last.png
ComfyUI/output/H3_Portrait/video_<unique suffix>_last.json
```

The native workflow uses the H3_Standard folder. The sidecar records actual decoded frame
count, last-frame index and dimensions. No guessed seek timestamp or directory-wide latest
file search is used. The PNG is the post-crop, post-encoding frame, not the pre-encoding tensor.
The final-frame node also outputs an IMAGE tensor for later connections.

Download the pair through port 8888. Saving a last frame prepares chaining; feeding it as a
Ref2VA reference alone does not guarantee a seamless continuation or force frame zero. A
future hard-first-frame/continuation workflow needs the appropriate keyframe conditioning.
H3's frame grid means a nominal 5-second request is normally 124 frames, about 5.167 seconds.

## 8. Standard MiniMax Ref2VA workflow, without Ollama

Open H3_Ref2VA_Standard_Lite_v1_4. Upload a subject image in LoadImage, write the prompt directly
in MiniMax H3 Reference to Video, optionally select/enable your H3 LoRA, and Run. This bypasses
both Ollama calls. Add more LoadImage nodes to the native reference image inputs as needed.
Picture numbers follow native connection order. No automatic roles are applied in this graph.
For an identity baseline, do not connect the unrelated person's raw pose photograph here.

The installer selects the correct full/INT8 filenames automatically. For a 2:3 native test,
set width=576,height=864 on Lite (768,1152 on Full), and set the crop node to 2:3. For 9:16,
leave the supplied values unchanged. It saves the MP4 and last-frame PNG just like Portrait.

## Profiles and tradeoffs

| Setting | Full | Lite |
| --- | --- | --- |
| H3 Ref2VA | Full BF16 | Full INT8 convrot, not pruned |
| H3 vision/text encoder | 32B BF16 | 32B INT8 convrot |
| Ollama analysis | 32B Instruct Q4 | 8B Instruct Q4 |
| Ollama director | 32B Thinking Q4 | Same 8B Instruct Q4, text-only |
| Standard 9:16 output | 756 x 1344 | 576 x 1024 |
| Standard 2:3 output | 768 x 1152 | 576 x 864 |
| Preview / Standard / High steps | 12 / 20 / 25 | 12 / 16 / 20 |
| High-reference sizing | max | match |
| Turbo auto-enabled | No | No |
| Sampler / scheduler | res_multistep / simple | res_multistep / simple |

Lite is intended for faster iteration, not equal fidelity to Full. Quantization, fewer steps,
smaller images and the smaller helper are deliberate tradeoffs. Actual speed, VRAM and visual
identity quality have not been benchmarked on your host. No acceleration LoRA is auto-loaded.
Full-model architecture is retained to avoid switching your LoRA to an incompatible pruned
model family; your particular private adapter has not been inspected or inference-tested.

## Source and testing limits

Your live repository read returned Not Found in this session. This release was built from the
supplied, previously checked source package and public upstream code at your pinned ComfyUI
commit. Your personal LoRA text was not read or extracted. The UPDATE ZIP leaves it untouched.
See INVESTIGATION.md and VALIDATION.md for evidence, checks and remaining deployment tests.

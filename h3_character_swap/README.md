# H3 Character Replacement: Akatz V1.5 + optional BFS v1.1

Release: 2026-10-22.1

This is an ISOLATED experiment image added to the existing repository. It does not
replace `h3-hq`, modify the approved SAM build, or alter HMOTION/HCLIPS/HSTILL.
No repository push or image publication is performed by the local installer.

## Start with Akatz

The recommended first comparison is the developer-native Akatz Character Swap
V1.5 workflow, with tracked SAM masks, source-aligned windows and continuation
history. It uses the developer's Ref2VA INT8 / NVFP4-encoder stack, the exact
V1.5 adapter at 1.0, its separately specified FL2V Turbo8 adapter at 1.0,
8 steps, er_sde, normal and CFG 1. This is a separate baseline, not a silent
precision change to your existing full-BF16 workspace.

The qualified upstream runtime is ComfyUI
`79be670e2d9be63e238785af307369d2b9039ed1`, frontend 1.53.6,
PyTorch 2.11.0 / CUDA 12.8 / Python 3.12. Relay is pinned separately to
`e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220` (2.0.1).
The new image does not add the previous pose ControlNet to these recipes.

## Files and installation

Apply the download with the enclosing `install_without_python.rb`. It only adds
this directory and `.github/workflows/build-h3-character-swap.yml`. It refuses
conflicting existing files and never changes other repository files.

After manually committing and pushing through GitHub Desktop, run:

**GitHub Actions > Build H3 character replacement > Run workflow**

This is NOT the old Build HQ workspace action. The new action performs local
kit tests, builds the image, starts its actual ComfyUI on CPU for node/schema
checks, then publishes only if those checks pass. No GPU is available on that
runner, so this does not certify rendering or visual quality.

Expected image tag after a successful build:

`ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap`

Prefer the immutable `ghcr.io/...@sha256:...` reference printed in the action
summary. The action also uploads the exact installed canvases, paired API exports,
asset lock and build provenance as `h3-character-swap-build`.

## Complete RunPod setup

See RUNPOD_TEMPLATE.md for all fields and the copy/paste environment block.
Use a new Pod template. Do not reuse old WB_TASKS/WB_WORKSPACE settings.
The image includes its entrypoint: leave the start command and entrypoint override
blank. Only HTTP 8188 is public; ComfyUI 8189 and Ollama 11434 stay private.

An existing `/workspace` network volume can be reused. Models with matching
filenames and SHA256 are reused; a conflicting model is never overwritten.
The new image keeps its user settings/workflow copies in
`/workspace/comfy-user-h3-swap`, separately from your previous workspace.

## First test: four steps

1. Open the Pod's HTTPS HTTP-8188 service. Log in as `workbench` with the
   `comfy_password` secret. Open `/_swap/status` on the same service and wait
   until `ready` is true. Refresh ComfyUI once preparation completes.
2. In the **ComfyUI Workflows sidebar**, open the folder
   `H3_Character_Replacement_2026-10-22.1`. Use **one** approved character image
   and the same short continuous source shot for the first comparison.
3. Optional but recommended: open **00_Analyze_Clip_Build_Prompts**. Select that
   source video and character image. Describe the source performer to replace,
   leave 124 frames, and run. Review the contact sheet, the action/scene analysis
   and the draft. Copy the reviewed Akatz prompt. This step does not run H3.
4. Open **01_Akatz_V15_Mask_Check**, select the source video, and describe the
   performer in the SAM text field. Review the mask. Then open
   **02_Akatz_V15_Native_Replacement**, use the same inputs and SAM selection,
   paste the reviewed draft into the native H3 reference prompt, and run.

The initial Akatz source duration is 124/24 = 5.1666666667 seconds, taken from
the start of the selected file. For an interior section of a long file, use one
of the HCLIPS files already prepared in `/workspace/input`, or add native video
trimming. Prompt Studio's `start_seconds` must describe the SAME section being
rendered. Easiest first trial: use the same already-trimmed 124-frame file in all
three workflows and keep both start controls at zero.

Prompt Studio has two reviewed outputs: an Akatz prompt and BFS character details.
It samples eight frames from the selected window, not every video frame. It does
not listen to source audio or infer unheard dialogue. Rapid actions between
samples can be missed, so inspect the draft against the actual source clip.
It can describe visible action; it cannot turn reference conditioning into a
pixel or pose lock. No automatic text rewrite runs inside the rendering loop.

The helper uses LOCAL `qwen3-vl:8b`, a 16,384-token context, 1,800 output-token cap,
one request at a time, cached drafts, no cloud backend and explicit unloading.
The source frames and character image never leave the Pod for prompt inference.
The only external transfers are software/model downloads during preparation.

## Mask selection and what is actually preserved

Use a specific SAM description such as `person in the blue shirt on the left`
when more than one person appears. Check the selected mask in the preview graph,
then reproduce that selection in the full graph. `largest` is only a default,
not proof that the correct performer was selected. The seed and expansion settings
are editable in the developer graph. Reference clothing is the documented default.

Akatz uses the source/reference inputs, tracked edit mask and history during
sampling. Its output is NOT pixel-composited back into the source through that
mask. Background, expression, occlusion and identity can still change.
Final audio is the original source soundtrack, trimmed to delivered duration;
that does not establish exact regenerated lip sync.

Do not change the matched Turbo adapter, sampler, scheduler or step count merely
because a larger number sounds higher quality. Compare the un-upscaled render
first. Only then change one setting at a time and keep the same seed/reference.

## Optional BFS v1.1 comparison

Read the BFS custom license before enabling it. The image build retrieves the
license into `/opt/h3-swap/licenses/BFS_CHARACTER_SWAP.md`; it is also included
in the GitHub Actions build artifact. The kit's source-code license does not
replace upstream model licenses.

Then update the Pod environment and restart with the same persistent volume:

```
H3_SWAP_VARIANTS=akatz,bfs
H3_SWAP_BFS_LICENSE_REVIEWED=1
```

This prepares the v1.1 adapter and its separate DMD Ref2VA Turbo8 adapter. It does
not download another H3 base or encoder. Open **03_BFS_V11_Guide_Comparison**.
Use the same source and character, then paste Prompt Studio's reviewed
`bfs_character_details` into the details field. The rest of the documented
`body_swap:` template is preserved automatically.

This comparison is a SIMPLIFIED ADAPTATION of the developer's first pass, not
an unmodified copy of the enormous upstream canvas. It retains the native
reference-image path, source video as Add Guide at frame 0, DMD8 + v1.1 adapters,
12/3 shifts, Euler/simple 8-step full-denoise sampling and matching source/target
frame counts. It removes the optional latent-upscale pass, background-removal
branch, third-party routing widgets and automatic native VLM branch. The local
reviewed helper supplies character details instead.

The default 384-pixel short edge follows the developer's low-resolution first-pass
baseline. After checking adherence, 576 and 768 are available on the source-window
node. Higher-resolution execution has not been benchmarked here. Outputs include
a raw video and a source-left/generated-right comparison, with original source
audio. The BFS template transfers the reference character's BODY AND CLOTHING;
it is not an identity-only face swap. Start with one prominent source performer.

## Longer clips

Akatz's native loop/history graph is retained. Setting Prepare Source's duration
to zero requests the full file, but do NOT start with a three-minute run. Validate
one 5.17-second shot first, then a longer continuous shot and each join. Scene cuts,
identity resets and long-source preparation can still cause issues. The fact that
a control allows a long video is not a hardware or quality guarantee.

Previously prepared HCLIPS outputs are visible here when the same input volume
is attached. This isolated image does not add the old HMOTION/HCLIPS UI or models;
return to the existing h3-hq image to run those tools. Their repository files and
published images remain unchanged.

## Output and diagnostics

- Akatz renders: `/workspace/output/h3_character_swap/akatz_v15*`
- BFS renders/comparisons: `/workspace/output/h3_character_swap/bfs_v11*`
- Prompt drafts/contact sheets: `/workspace/output/h3_character_swap/prompt_drafts/`
- ComfyUI log: `/workspace/h3-swap/logs/comfy.log`
- Ollama log: `/workspace/h3-swap/logs/ollama.log`
- Preparation status: `/workspace/h3-swap/status.json`
- Downloaded model verification: `/workspace/h3-swap/verified-models.json`
- Ollama model digest: `/workspace/h3-swap/prompt-model.json`

Inside the running Pod:

```bash
python -m swapkit.check --live
nvidia-smi
OLLAMA_HOST=http://127.0.0.1:11434 ollama ps
tail -n 100 /workspace/h3-swap/logs/comfy.log
```

An empty `ollama ps` after prompt analysis is expected. The helper is intentionally
unloaded. Old `WB_TASKS`, `WB_TOOLBOX`, `ENABLE_H3_MEDIA` and similar workspace
variables do not configure this isolated image.

## Validation boundaries

The local deliverable's tests cover graph topology/API consistency, model metadata
locking logic, safe installation, authentication, bounded prompt payloads and
unloading, prompt-template handling, exact local video extraction and source audio.
Actual FFmpeg tests use synthetic 60 fps video and verify 124 output frames at 24
fps plus matching audio. They do not measure perceptual quality.

This environment could not fetch full upstream repositories into the container or
build/run Docker. The two complex Akatz canvases are therefore generated from the
PINNED upstream repository during your image build, not hand-recreated from partial
text. The image build and CPU smoke test must pass before publishing. GPU inference,
A100 peak VRAM, identity/motion adherence, interactive frontend behavior and long-clip
continuation remain untested here. No workflow is marked user-approved.

# H3 character swap with actual pose control

Release: 2026-10-22.1. Source-reviewed candidate. GPU inference and visual quality are not certified.

## What changed

The earlier no-Ollama graph supplied the video only to H3's reference-conditioning input. That can request motion without enforcing the sequence's structure. This update adds the native MiniMax H3 Fun ControlNet Union **2.0** branch. SDPose extracts the selected performer's pose on every sampled source frame. Those pose frames enter the native `MiniMaxH3FunControlNetApply` node during every denoising step. The original video still enters the reference-conditioning path for scene, clothing and camera context.

The model author recommends 2.0 over v1. It has ten control-injection blocks instead of five. The existing pinned ComfyUI core already detects this layout. The new build contract verifies that implementation, and a runtime node rejects a loaded five-layer patch. No broad ComfyUI upgrade is made.

This is a substantive change to conditioning, not a stronger prompt alone. It should provide more direct motion guidance, but the exact combination of this graph, full-precision Ref2VA weights, two identity references and your footage needs a GPU test. No benchmark is claimed.

## Included workflows

In Workbench, choose **HMOTION: Character swap: frame-by-frame pose control (no Ollama)**.

1. **00_H3_Motion_Preview.json**: upload the video, set start/window, and run. It creates a SOURCE | POSE CONTROL comparison without loading H3's diffusion model, text encoder or ControlNet. This is the first check to run.
2. **01_H3_Character_Swap_Pose_Control.json**: upload two photographs of the same replacement character and the source video. It extracts motion, validates alignment, then runs the pose-controlled character swap. It saves the control check, a generated MP4, a synchronized SOURCE | GENERATED comparison and an RGB16 lossless MKV master.

Outputs are under `/workspace/output/video/H3_Motion_Control/`.

The render graph also has a **Preview motion only** gate. All four large H3/ControlNet loaders depend on it, so preview mode cannot quietly load those models. The separate preview graph does not require character photos at all.

## Defaults

- Existing Ref2VA BF16 diffusion weights and H3-specific BF16 encoder.
- Existing FP16 video VAE. No additional image decoder.
- Comfy-Org Fun ControlNet Union 2.0 pruned **BF16**, not its INT8 variant.
- SDPose whole-body FP16 plus RT-DETR FP16 person detection.
- Source aspect ratio selected automatically, approximately 1.03 MP native canvas; axis sizes are multiples of 32. Small letterboxing avoids a forced portrait crop or stretched anatomy.
- `res_multistep`, `beta`, 40 steps, full denoise, native shifts 12/3, BasicGuider (guidance 1).
- Control strength 1.0 over the complete denoising interval (start 0, end 1).
- Body, head orientation and hands drawn. Dense facial landmarks are off to reduce transfer of the original performer's facial geometry. Ankle positions remain part of the body skeleton.
- Two full image references at `ref_image_size=max`; no intermediate face enhancement.
- Fixed seed 42 for controlled comparisons. Silent output, no original or generated soundtrack.
- No Ollama, Reference Pack, Turbo LoRA or reduced-precision substitution.

Forty steps follows the control model author's example count; the complete sampler/reference combination here is an adaptation, not a reproduction of an independently measured optimum.

## Timing correction

Only one bounded source window is decoded. Frame-rate conversion uses FFmpeg's timestamp-based `fps=24`, not a change of playback speed. Source frames, detected pose frames, the H3 reference-video input and the generated latent length all use the same count.

Frame counts round **down** to a valid `17*n+5` count. No last-frame hold, looping, temporal interpolation or unconditioned generated tail is inserted.

- Default source window 5.25 seconds: 124 frames, or 5.167 seconds, when enough footage is present.
- A 15.000-second source: 345 frames, or 14.375 seconds. The remainder is deliberately not rendered.
- A source and selected window of at least about 15.084 seconds can supply 362 frames, or 15.083 seconds. Set the window to 15.1 when that much footage is actually available.
- Output never exceeds 362 frames. Shorter than 124 decoded frames is rejected before H3 sampling.

Use the timing report for the actual output duration, especially with variable-frame-rate sources. A longer uploaded file is permitted, but only the selected window is decoded. HDR footage is rejected rather than silently being interpreted as SDR; convert it to SDR first.

## Performer selection and pose quality

Default `first_person=-1` selects the largest person in the first frame, then associates detections geometrically. Set 0, 1, etc. to select first-frame people ordered left to right. This is **not face recognition** and can fail when performers cross, disappear or overlap.

The helper can reuse a crop for a short detector gap, but never copies or freezes the underlying video frames. It stops when the track is lost for too long. It also stops before rendering if pose extraction is substantially missing. Always inspect SOURCE | POSE CONTROL, especially for dance, occluded hands and fast turns. A correct source video cannot compensate for a broken skeleton.

The render graph unloads Comfy-managed preprocessing models before loading H3. It does not kill other processes, stop unrelated jobs, or evict an independently running Ollama service. On a fresh Pod with the settings below, Ollama is not started.

## Using the update

Apply the package locally with its Ruby installer. It merges the new task and model entries without replacing unrelated catalogs. Existing H10/H11, HFL, HSTILL and the approved SAM workflow remain intact. An existing local file with conflicting new content stops installation. Backups are stored alongside the extracted download, outside the repository; a failed write attempts rollback.

Review/commit/push in GitHub Desktop, then run **Build HQ workspace** for **h3**. Use the digest from that successful build. Do not load the new graph in an old image and expect missing nodes/models to appear.

For just this motion task:

```text
WB_WORKSPACE=h3
WB_TASKS=HMOTION
WB_TOOLBOX=standard
WB_DISABLE_COMFY_COMPILER=1
```

For motion plus the previous video/stills package, use `WB_TASKS=HMOTION,HSTILL,HFL,HFL4K`. Keep authentication, Hugging Face secrets, HTTP port 8188, `/workspace` persistence and the default container entrypoint unchanged. Do not add Ollama to the toolbox.

The three new weights total **10,422,903,562 bytes**, approximately 10.4 GB. They are downloaded on the Pod through the existing checksum-verified downloader when HMOTION is selected. They are not in the ZIP or Docker build. The `model_patches` directory is now persisted alongside the existing model directories.

An A100 80 GB still needs model offloading with this full-precision stack. The exact peak memory for this control graph is unmeasured. Keep the short default clip and approximately 1 MP canvas first; your existing 300 GB volume target leaves additional headroom if it is not already full. This update does not claim that a larger GPU is required or that 80 GB is guaranteed sufficient.

After model preparation, the following Pod command performs read-only source/schema checks, not a generation:

```bash
python -m workbench.h3_motion_contract --core /opt/ComfyUI
python -m workbench.h3_motion_contract --live
```

## Adjusting adherence

Start at strength 1.0. Only after confirming that the pose preview follows the desired performer, compare the same seed at 1.15, then 1.3. Increasing strength can hurt appearance or introduce artifacts; it is not a monotonic quality control. Keep guidance at 1 through BasicGuider and do not add a Turbo LoRA to this recipe.

The direct prompt is editable. Keep the identity-reference roles and video-reference role clear. The graph uses the video as scene/camera context and the pose sequence as structure. Dense source-face landmarks remain off by default.

## What this does not promise

This is not a pixel-locked replacement of the original performer. Source backgrounds, clothing, small objects and facial expressions can change. Pose control gives limb/head/hand structure, not exact optical flow or a guarantee that every motion detail is reproduced. Precise background preservation would require an additional tracked-mask/inpainting/compositing pipeline; that is not silently claimed here. The source video can still influence the original person's appearance. Judge both identity and motion before accepting a take.

## Verification scope

The package includes CPU/offline tests for graph links, patched-model routing, shared timing, FPS decoding, performer selection, pose-health failures, preview gates, pinned assets and the Mac installer. A real synthetic 60 fps MP4 was decoded and checked. Installer tests use disposable local Git fixtures, not the entire repository.

The complete repository CI suite, Docker build, model downloads, native GPU execution and generated-image quality were not run here. GitHub Actions and the actual Pod remain deployment gates. The workflows are explicitly unapproved pending visual testing.

## Primary sources

Checked October 22, 2026:

- Control model author, v2 model card and example settings: https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0
- ComfyUI native control workflow, supported control types and strength advice: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet
- Audited native core: https://github.com/Comfy-Org/ComfyUI/tree/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c
- Native control application: `comfy_extras/nodes_minimax_h3.py`
- v2 ten-block model loading: `comfy_extras/nodes_model_patch.py`
- Native SDPose and detector: `comfy_extras/nodes_sdpose.py` and `comfy_extras/nodes_rtdetr.py`
- Control weights and checksum: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/bf92c4091e333e69b8ca1998e0a669f15cb0832b/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors
- Pose/detector weights: https://huggingface.co/Comfy-Org/SDPose/tree/acb43fbe8142b54d9ebd52b720b81dd4e14c8d81

Model weights retain their upstream licenses. None are redistributed in this package. This package calls the existing native ComfyUI implementations rather than vendoring a competing ControlNet or pose implementation.

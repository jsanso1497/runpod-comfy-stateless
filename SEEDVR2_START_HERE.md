# SeedVR2 Video Clarity

## The one setup to use

Use **Build SeedVR2 Stock FP16**, the existing image tag `seedvr2-stock-fp16`, and the single workflow **00_SeedVR2_Stock / 01_Video_Clarity**.

This update replaces the competing full-frame/4K/crop presets with one stock-node graph. It is based on GitHub main `d6e5f6a15cd03d31d262d6a31a6c825751c121e9` and carries forward the latest SeedVR2 Workflow Setup chat's first-frame comparison. The v2/v2.1 packages discussed there were not present on that main revision; their relevant requirements are consolidated here, not assumed to have been deployed.

**Goal:** feed in ordinary 1920 x 1080 video and get a clearer interpretation at 1920 x 1080, or crop first and restore only the selected region at its existing pixel dimensions. More output pixels are not treated as proof of improved image quality.

No Qwen pass, custom inference node, custom worker, source blend, custom FFmpeg exporter, runtime FPS assertion, face model, motion interpolation, sharpening pass or 4K detour is included. ComfyUI and the official SeedVR2 extension are unmodified. The current repository's standard Nginx password protection is retained. No new Python gateway, cookie/session service, or custom UI plugin is added.

## Install once, on the Mac

Save `update_seedvr2_clarity.py` to Downloads. Open Terminal in the root of your local `runpod-comfy-stateless` repository and paste:

```bash
python3 -c "from pathlib import Path; import runpy; runpy.run_path(str(Path.home() / 'Downloads' / 'update_seedvr2_clarity.py'), run_name='__main__')"
```

The self-contained installer verifies hashes, refuses unreviewed conflicts/symlinks, backs up changed files inside Git's private metadata, and archives the obsolete video build actions and the two old factory workflow JSONs. It does not fetch, stage, commit, push, start a build, deploy, or delete Pod data. Repeat runs are idempotent. `--check` previews the changes when invoking the script directly.

Review the changes in GitHub Desktop, commit and manually Push origin. Run a **new** **Actions > Build SeedVR2 Stock FP16 > Run workflow > main**. A rerun of an old build uses the old commit.

Use this update instead of stacking the older Faithful, Native, v2, v2.1, FPS or login installers. If the conflict check finds local edits, it stops before changing anything rather than overwriting them.

The installer retires **Build Faithful Video Minimal** and the old combined **Build Qwen + SeedVR2 video repair** action. It also retires the previous Native action if that exact provided action exists locally. The Qwen, MOSS, H3 and approved image-editing source/workflow files are not modified. Old runtime folders can remain as historical source, but are not installed or run by this image.

## RunPod template

After the new build succeeds, use the exact image digest from its summary. The existing rolling tag is retained:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:seedvr2-stock-fp16
```

The installer does not publish this image. An already running Pod does not acquire the new code merely because the tag was rebuilt.

| Field | Value |
| --- | --- |
| HTTP port | 8188 only |
| TCP ports | None required |
| Container start command / entrypoint override | Blank |
| Container disk | 40 GB planning allocation |
| Network volume | 100 GB or larger, mounted at `/workspace` |
| Required environment | `COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}` |
| Login username | `workbench` |
| Jupyter / SSH / transfer utility | Not required |

The password secret must resolve to a unique 16-72 byte value. The browser may show its normal username/password dialog. There is no separate sign-in website. Use RunPod's **HTTPS** Connect link for the active Pod, not an old Pod's URL. Do not expose internal port 8189. No `WB_*`, `FAITHFUL_*`, Hugging Face token, CivitAI token or RunPod API key is required. Public weights are downloaded by the official model loader on the first run.

Nginx is retained solely to avoid exposing an unauthenticated ComfyUI server. This is not a compliance certification for sensitive media. Follow your organization/client's policy for any confidential recordings.

Use a **network volume** when you need data to survive termination. A regular Pod volume is deleted with its Pod. This update cannot recover the terminated Pod's output if its storage was deleted.

## Normal use: upload, optionally crop, Run, download

Open **Workflows > 00_SeedVR2_Stock > 01_Video_Clarity**.

1. Upload the original video in **1. Upload the original video**.
2. Leave the native **2. Optional crop** rectangle at all zeros to keep the full frame. To isolate the subject, use that crop node's rectangle controls before running. Use the values below for a 1920 x 1080 source.
3. Click **Run**. First use downloads the selected model and VAE. Official SeedVR2 runs inside normal ComfyUI, with its native progress, queue and logs. No separate hidden restoration worker is used.
4. When **4. Finished video** completes, use its native preview and save/download option. **5. First-frame comparison** saves a PNG with the source on the left and restored frame on the right.

The comparison uses the same crop, frame 0 and pixel size on both sides. It does not create a second AI pass, scale the full video for comparison, or imply that an AI-moved edge is geometrically identical. Check the MP4 during motion too.

### Crop settings for a 1920 x 1080 source

The crop rectangle uses `x`, `y`, `width`, `height`. These are **pixels**, not percentages. `x` increases toward the right; `y` increases downward. Width and height zero mean full frame in the native VideoCrop node.

| Keep | x | y | Width | Height | Restored output |
| --- | ---: | ---: | ---: | ---: | --- |
| Entire frame | 0 | 0 | 0 | 0 | 1920 x 1080 |
| Square, left | 0 | 0 | 1080 | 1080 | 1080 x 1080 |
| Square, center | 420 | 0 | 1080 | 1080 | 1080 x 1080 |
| Square, right | 840 | 0 | 1080 | 1080 | 1080 x 1080 |
| Portrait 2:3, left | 0 | 0 | 720 | 1080 | 720 x 1080 |
| Portrait 2:3, center | 600 | 0 | 720 | 1080 | 720 x 1080 |
| Portrait 2:3, right | 1200 | 0 | 720 | 1080 | 720 x 1080 |

You can position the rectangle anywhere, not just at these anchors, and use another aspect ratio. Keep it within the source, use **even dimensions and preferably even x/y coordinates**, and keep the subject inside it throughout the shot. This is a fixed crop, not automated subject tracking or reframing.

The graph reads the crop's actual width and height and sets SeedVR2's short edge to the smaller number. It does not enlarge a 720 x 1080 crop to 1080 x 1620, put it back into the full frame, or resize it to 4K. Thus the 2:3 example has 62.5% fewer model-output pixels per frame; the square example has 43.75% fewer. These are arithmetic pixel reductions, not measured speedups or guaranteed billing savings. Model loading and other fixed costs remain.

Native VideoCrop is marked experimental by ComfyUI at the pinned version; it is an upstream node, not code added by this package. Its crop and output behavior are covered by the required build-time native media tests. Ordinary even-sized crops in the table are the intended use.

## Quality settings: one model pass, no larger-by-default output

The shipped baseline is **standard SeedVR2 7B FP16 plus FP16 VAE**, LAB color correction, zero added input/latent noise, batch 9, uniform batches, temporal overlap 3 and four prepended startup frames. VAE encode/decode tiles are 1024 with overlap 128; CPU offload and 36-block swapping are enabled. PyTorch SDPA is used; optional compilation is not connected. The official extension controls actual compute dtypes.

These are documented upstream controls and conservative starting values, not a footage-benchmarked optimum. Standard is the default to avoid deliberately selecting the detail-emphasizing Sharp model. Both models are generative and can alter or invent detail.

For a stronger-detail comparison, change only the **model** in the existing DiT loader to:

```text
seedvr2_ema_7b_sharp_fp16.safetensors
```

Keep the other settings identical for that comparison. This is the Sharp option from the other chat without duplicating eight workflows or silently switching temporal settings. Only the selected checkpoint downloads. The standard checkpoint remains cached for switching back. No Lite/quantized model is substituted.

Do not equate a sharper still with better video. Check skin, eyes, fingers, text, straight edges and texture during motion. Missing information cannot be guaranteed to be recovered accurately. A larger 4K render is deliberately not included as a default quality improvement.

Native Save Video produces an MP4 with its ordinary H.264 encoder defaults. It is compressed delivery, not a lossless master. Its own built-in codec/re-encode settings remain available, but no custom exporter is added. The ordinary decoded source audio and source FPS are connected to Create Video; arbitrary VFR timestamps, HDR/log metadata, extra audio tracks and archival metadata are not promised to survive this image-batch path.

## Memory and storage

This is the upstream whole-clip graph. A nine-frame temporal batch is not a nine-frame RAM limit. There is no hidden three-second cap; all uploaded frames are processed, so use a short shot for the first acceptance test.

For short 1080p shots, a **48 GB CUDA GPU with 128 GB allocated system RAM** is a conservative first target. An existing 80 GB GPU provides additional VRAM headroom; it is not required by a measured benchmark here. Long videos can still exhaust system RAM. At 1080p/30 FPS, 60 seconds of RGB float32 frames alone occupy about 41.7 GiB; source/output copies, weights and intermediate buffers add more. Do not assume a one-minute full-frame clip will fit 128 GB. Prefer shot-length clips, or validate a larger RAM allocation. Cropping reduces the selected frames' spatial size before restoration but does not turn this into a streaming architecture.

No precision/resolution/model fallback is hidden. If a live render runs out of GPU memory, the upstream controls are temporal batch 5 (not 1 for video), and smaller VAE tiles such as 512 for VAE-stage OOM. This is optional troubleshooting, not a required normal-use sequence.

Files persist in the existing Stock layout:

```text
/workspace/models/SEEDVR2/
/workspace/seedvr2-stock/input/
/workspace/seedvr2-stock/output/SeedVR2/
/workspace/seedvr2-stock/user/
```

Unmodified old factory 1080p/4K workflows are moved out of the workflow list into `/workspace/seedvr2-stock/workflow-backups/pre-clarity-v3/`. User-edited copies are left untouched. Outputs and models are never deleted by this migration.

## What is verified, and what is not

Local checks validate graph links, official-node-only structure, model/settings consistency, crop coordinates, same-size geometry, same-frame comparison routing and installer safety. Local Nginx tests use a **mock upstream**, not ComfyUI. They exercise authentication, a multi-megabyte response, HTTP ranges, chunked uploads and a sustained WebSocket. Local Chromium is blocked by the environment's browser policy; no local browser pass is claimed.

The new GitHub Action builds before publishing. It must then run actual pinned **CPU ComfyUI** through the actual Nginx configuration and test full-frame, left/center/right square and 2:3 crops, a custom crop, fractional/unusual FPS, audio, native comparison PNGs, native MP4 export, byte-exact downloads, HTTP range responses, sustained WebSocket events and log subscriptions. Actual Chromium must load and execute the native CPU control graph, stay connected, subscribe with a real clientId and download the MP4 in-browser.

The three SeedVR2 GPU nodes are explicitly removed only in the CI control graph. Production has no bypass. GPU selectors are not patched to fake a GPU. Static validation checks the unmodified SeedVR2 execute signatures separately. CI does **not** download weights, run CUDA inference, benchmark peak RAM/VRAM, prove public RunPod routing, or certify restored quality. A red build publishes no updated image. Test logs and browser evidence are uploaded even when a check fails.

## Source basis

- Official SeedVR2 integration, model loader and quality controls: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tree/4490bd1f482e026674543386bb2a4d176da245b9
- Developer video example: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/example_workflows/SeedVR2_HD_video_upscale.json
- Native VideoCrop/CreateVideo/SaveVideo: https://github.com/Comfy-Org/ComfyUI/blob/2c75dc1b379a59e0880619d029607af5169a62ef/comfy_extras/nodes_video.py
- Native math helper: https://github.com/Comfy-Org/ComfyUI/blob/2c75dc1b379a59e0880619d029607af5169a62ef/comfy_extras/nodes_math.py
- Standard Nginx WebSocket proxying: https://nginx.org/en/docs/http/websocket.html
- RunPod volume lifetimes: https://docs.runpod.io/pods/storage/types

Pinned source revisions and environment freeze are retained in the image. Use its published digest when deploying.

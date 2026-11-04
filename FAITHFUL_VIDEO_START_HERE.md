# Faithful Video Minimal

Prepared November 3, 2026 for `jsanso1497/runpod-comfy-stateless`.
Reviewed repository commit: `250a33cf28b3ce36b00e496684a7dd819eaf53b7`.

**Status: source package with passing local CPU/media tests. No Docker build, CUDA restoration, GPU-memory measurement, or visual restoration comparison was performed here.** The new GitHub Action contains additional real-image integration gates that run when you build it. A green build still does not certify visual quality.

## Purpose

A separate, video-only ComfyUI workspace for soft, ordinary 1920 x 1080 camera footage. The goal is modestly clearer edges and texture while retaining the source, not aggressively inventing a new high-detail interpretation.

Use the actual original video, not the attached JPEG crop. A still does not establish motion blur, flicker, interlacing, or how neighboring frames might help. There is no promise of recovering genuine information that is absent from the recording. Any newly legible screen text or tiny facial detail may be synthetic.

## What is included

One workflow, two processing nodes:

`Load Video -> Faithful Video | Enhance + Save`

The only large model files are the standard SeedVR2 7B FP16 checkpoint and its matching FP16 VAE, approximately 17 GB combined. Fixed text embeddings shipped with the pinned SeedVR2 source are used internally; no text-encoder model or prompt-helper service is downloaded. There is no Qwen, SAM, Ollama, Sharp checkpoint, face restorer, LoRA, optical-flow interpolator, post-sharpening filter, or second generative pass.

The new build is isolated under `video_fidelity/`. It does not edit the Qwen/MOSS/H3 containers, their workflows, catalogs, image tags, or approved-workflow status.

## Install into your local repository

Download `install_faithful_video_v1.py` to your Downloads folder. Open a terminal in the root of your local `runpod-comfy-stateless` repository and run:

```sh
python3 -c "from pathlib import Path; import runpy; runpy.run_path(str(Path.home() / 'Downloads' / 'install_faithful_video_v1.py'), run_name='__main__')"
```

The script contains its entire compressed payload. It verifies payload and file hashes, checks repository location, refuses symlink paths or conflicting existing files, and adds only this package and its new build action. It does not fetch, stage, commit, push, start an Action, or deploy a Pod. Re-running it is safe when installed files are unchanged. `--check` can be used when invoking the script directly to preview without writing.

Review the added files in GitHub Desktop, commit there, and manually click **Push origin**. Then run **Actions > Build Faithful Video Minimal > Run workflow**.

After the build succeeds, use the exact image digest shown in its summary. The action publishes a commit-specific tag and `ghcr.io/jsanso1497/runpod-comfy-stateless:faithful-video-v1`. These are future build outputs, not an image already published by the assistant. Do not deploy an assumed tag before a successful build. A private GHCR package requires appropriate RunPod registry credentials.

## RunPod setup

| Setting | Value |
| --- | --- |
| GPU | 1 x A40 48 GB, as the first validation target |
| Host RAM | Prefer 128 GB for offloaded weights and temporal buffers |
| Container disk | 30 GB |
| Persistent volume | 150 GB, mounted at `/workspace` |
| Container image | Exact digest from the successful new build |
| HTTP port | 8188 only |
| TCP port | None needed |
| Start command / entrypoint override | Leave blank |
| Login username | `workbench` |

These are planning allocations, not measured hardware minimums. Native 4K and larger temporal batches need live memory testing. Disk-backed RGB16 frames can consume tens of GB. The job checks a conservative storage estimate before inference. Longer videos may need more than 150 GB.

Required environment variable:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

Optional environment variable:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Use a unique password of 16-128 UTF-8 bytes. Unresolved secret syntax or a short password stops startup. Both model files are public, so an HF token is optional. No CivitAI or RunPod API key is required by the runtime.

Use RunPod's **HTTPS** connection for HTTP port 8188. Internal ComfyUI listens only on loopback port 8189. Do not expose 8189. The first start prepares and SHA-256-verifies the two locked weights. Existing matching files in `/workspace/models/SEEDVR2` are reused through symlinks instead of downloading duplicate weights.

Data is under `/workspace/faithful-video`. Keep inputs, outputs and important work on persistent storage, not the container filesystem. A regular Pod volume does not follow you automatically to another Pod; a compatible network volume can be attached to another deployment. Back up outputs before terminating a Pod. Closing the browser does not stop GPU billing; stop the Pod in RunPod when finished.

Do not reuse the old Qwen environment-variable list. This standalone image uses only the settings shown above.

## First run

Open **Workflows > 00_Faithful_Video_v1 > 01_Faithful_Video**. Upload your original video in the first node, then Run.

| Control | Initial value | Meaning |
| --- | --- | --- |
| Output size | Original size | 1920 x 1080 in, 1920 x 1080 out. No crop or geometric resize. |
| Restoration strength | 0.50 | Equal blend of the source frame and model restoration. |
| Start | 0 seconds | Beginning of the clip, rounded to the nearest frame. |
| Duration | 3 seconds | Explicit acceptance-test segment. Set 0 for all remaining frames. |
| Temporal batch | 9 | Nine frames per model batch; not a nine-frame clip limit. |
| Keep 16-bit frames | Off | Delete successful intermediate frames, retaining the MP4, settings and logs. |

For a video shorter than three seconds, use duration 0. The system does not silently extend a short clip or cut a longer requested range. For an edit containing multiple shots, individually trimmed shots are still the safest quality check. Automatic hard-cut detection is included, but dissolves and subtle edits are not guaranteed to be detected.

### Adjusting the result

Start with the defaults on the hand, phone edge, door molding and any moving face in the actual footage. Compare at normal playback speed and at 100% scale.

- Use 0.25-0.40 strength when invented texture, edge movement or excessive smoothing appears.
- Use 0.60-0.70 only when 0.50 remains too soft and the higher setting stays faithful in motion.
- Strength 0 bypasses model inference for a conventional source/export control. Strength 1 is the unblended model result, not the fidelity-first default.

These strength values are engineering starting points, not developer-prescribed or footage-benchmarked optima. The blend is computed as `source + strength * (restored - source)`. It reduces the amplitude of all model changes, including both improvements and errors. It does not identify or remove hallucinations, guarantee anatomical identity, or prevent all ghosting when a model moves an edge.

Keep **Original size** until the restoration looks acceptable. Choose **2x size** only when a larger deliverable is useful. For 1920 x 1080 input, SeedVR2 then actually restores at 3840 x 2160 before the source blend; this is not a plain resize disguised as 4K inference. The source side of the blend is conventionally resized to that same canvas. More output pixels do not prove that more true detail was recovered.

Temporal batches 13 and 17 are available for controlled comparisons after 9 succeeds. More temporal context is recommended by the integration developer when memory allows, but it is not a guarantee that every larger setting will look better or fit your GPU. On OOM, reduce the explicit batch to 5 and re-test. The workflow never secretly switches to batch 1, lower precision, another model, or smaller output dimensions.

## Quality and reliability choices

The standard model avoids deliberately selecting the enhanced-detail Sharp variant. Both remain generative, so standard is not intrinsically hallucination-free. LAB color correction follows the integration maintainer's fidelity recommendation. Added input noise and added latent noise are both zero. There is no automatic relighting, white-balance correction or extra sharpening step.

FP16 refers to the checkpoint files. The pinned upstream engine retains its own native compute-dtype policy, including BF16 where supported; this package does not force every operation to FP16 or add quantization.

Uniform temporal batches are on; inner overlap is 3 frames; four warmup frames are prepended and explicitly discarded for each window. VAE encode and decode use 512-pixel tiles with 128-pixel overlap. All 36 transformer blocks can be offloaded to CPU, along with VAE and intermediate tensors. These memory optimizations do not substitute quantized weights. They can affect speed and, for tiling/batch choices, generated pixels.

The outer streaming layer has five-frame overlap. It retains previous generated overlap and blends it with the next window's predictions for the **same source frame indices**. It never crossfades adjacent moments or uses frame interpolation. It flushes overlap at detected hard cuts. This corrects the old implementation's unblended outer boundary behavior; it does not certify completely flicker-free footage.

A CPU prepass uses FFmpeg `scdet` at threshold 10 on a small analysis copy. Its analysis resize never changes model-input geometry. The prepass is a heuristic and may miss or over-detect cuts.

Only one bounded window and overlap are processed at a time. The GPU process is separate from ComfyUI and exits before CPU output encoding. Model caches are reused inside that one worker to avoid reloading weights for each small window. They cannot persist after its process exits. Cancellation terminates the owned process group; Linux parent-death signals also protect the supervisor and GPU worker. Fused SDPA is required; there is no silent high-memory math-attention fallback.

Output dimensions are either exactly the even source dimensions or exactly twice those dimensions. This avoids the earlier arbitrary-short-edge rounding mismatch. Unexpected native dimensions cause an error rather than an invisible corrective stretch. Canvas agreement does not guarantee pixel-perfect model content alignment.

## Media limits and outputs

This preset accepts common **8-bit SDR, progressive, square-pixel, constant-frame-rate video** at 1-120 FPS, with even source dimensions and an output long edge no larger than 4096 pixels. HDR, known wide-gamut, high-bit-depth/alpha, interlaced and non-square-pixel inputs are rejected rather than silently changed. Log footage should be explicitly color-managed to SDR first; missing metadata cannot reliably identify log encoding. Untagged HD input is interpreted as BT.709 and that assumption is recorded.

Actual prepared timestamps are checked. Variable/discontinuous timing is rejected; there is no quiet conversion to average FPS. The first audio track is trimmed with the video and encoded to AAC, not copied bit-for-bit. Large original audio/video start offsets are rejected for deliberate normalization. Additional audio tracks and subtitles are not carried through this minimal path.

The default MP4 is H.264, slow preset, CRF 14, 8-bit YUV420, BT.709, AAC 256 kb/s and faststart. It is a high-quality delivery file, not a lossless master. Native generated frames are written as RGB16 PNG before encoding, avoiding an additional intermediate 8-bit rounding step. That does not make the 8-bit camera input a 16-bit original or recover lost sensor precision. To retain these frames, enable Keep 16-bit frames. They remain under the job directory, along with the prepared source; export/color-manage deliberately for a graded master.

Paths:

```text
/workspace/faithful-video/input/
/workspace/faithful-video/output/faithful_<id>/enhanced.mp4
/workspace/faithful-video/output/faithful_<id>/settings.json
/workspace/faithful-video/jobs/faithful_<id>/
/workspace/faithful-video/logs/comfy.log
```

Failed jobs retain their own diagnostics and intermediates. No unrelated outputs or model files are deleted. Successful jobs remove only their intermediates when the toggle is off. Saved outputs stay on disk even if ComfyUI's transient job history does not survive a restart.

Do not connect Get Video Components or a second Save Video node downstream: this node already saves, and its returned VIDEO is file-backed.

## Verification and honest limits

Local tests cover frame-index continuity, same-frame overlap blending, hard-cut resets, strength endpoints, integer output geometry, source validation, trim bounds, credential exclusion, process cleanup, graph/API contracts, authentication and actual FFmpeg/OpenCV decode/encode with synthetic media. Mock-inference tests are explicitly mock inference, not model benchmarks. See `video_fidelity/reports/validation.json`.

The new build Action additionally checks exact source pins and CLI blob hash, imports the actual pinned SeedVR2 CLI, starts actual ComfyUI in CPU mode, and executes the two-node graph with AI strength zero. Only after these checks pass does it publish. The Action does not download weights or claim a GPU quality test.

Before approving this workflow, run the three-second default on the actual recording, inspect hands, faces, phone boundaries, door edges and flat wall texture through motion, then compare one stronger and one weaker blend as needed. Check frame count, audio sync and actual output dimensions. Approve the result, not merely a successful job or a sharp paused frame.

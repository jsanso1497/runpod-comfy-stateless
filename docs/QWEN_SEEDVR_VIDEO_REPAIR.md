# Combined Qwen BF16 + SeedVR2 video repair

Release: 2026-11-03.video-repair-1
Reviewed repository base: f310e8f119b0bdf853ffefd1153ab6459a66d567.

## Scope

One Qwen workspace contains the existing Qwen editing workflows and the new video task RVID. Qwen diffusion weights, required vision/text encoder, VAE, prompts, native graphs, LoRA graphs and sampling settings are not rewritten. WB_QWEN_ONLY=1 disables optional prompt-helper LLMs and MOSS downloads. The image does not install an Ollama binary.

The repaired video graphs contain LoadVideo, one purpose-built streaming restoration/save node, and a note. No AusBoss loading, saving, cropping or memory-purge node is used in these graphs. The already-pinned AusBoss package remains installed because other existing workflows can depend on it; it is not globally removed.

The visual baseline is the earlier Sharp FP16 + FP16 VAE workflow, wavelet color matching, no added input or latent noise, no sharpening filter. It does not switch to INT8. The source-reference 512-pixel VAE tiles and 128-pixel overlap remain the starting configuration. Streaming boundaries can change the generated pixels, so these are not pixel-identical recreations.

## What is repaired

1. Full-clip frame accumulation: the parent ComfyUI graph holds a disk-backed VIDEO and scalar settings, not all decoded/restored frames. A separate job supervisor prepares the source on disk and runs the pinned single-GPU SeedVR2 CLI with a nonzero chunk_size. Each chunk is written to PNGs. No list of all restored chunks is concatenated in ComfyUI.
2. End-of-render memory spikes: GPU inference exits before CPU FFmpeg performs the final resize/encode. The finished VIDEO handle refers to an MP4 already on disk. Do not attach Get Video Components, another full-frame conversion or another video saver downstream.
3. Cancellation: the Comfy node catches BaseException, terminates and reaps its owned process group in finally, and its subprocesses install Linux parent-death signals. Abandoned GPU workers should not survive a Comfy crash. No unrelated processes are killed.
4. Legacy cancellation: the image verifies the original pinned SeedVR2 file hash, then patches its Exception-only handler to catch BaseException, discard caches on failure, and re-raise the original error. The upstream revision stays pinned; unexpected source changes fail the build.
5. Qwen/video handoff: --cache-none and --disable-smart-memory plus a job-boundary cleanup hook reset execution caches and unload Comfy-managed models after success, failure or cancellation. Error status and history messages are preserved. This costs reloading/recomputing time. It does not purge arbitrary third-party globals or external programs.
6. Inference safety: the GPU child disables the high-memory SDPA math fallback. It uses PyTorch fused attention without installing flash-attn. If no supported fused kernel is available, it fails clearly rather than silently trying a large math allocation.
7. Diagnostics: each video job retains phase-specific logs, attempted settings and status, including the failing phase. Comfy logs rotate on the persistent volume. A protected on-disk media browser and memory/error view use the existing 8188 gateway.

These are architecture and lifecycle repairs. The exact original OOM cause was not established because no complete failing traceback/memory dump was supplied. These changes are not a claim that an A40 can fit every requested setting.

## Memory defaults

| Mode | Inference batch | New frames per disk-backed chunk | Swapped blocks | Tiles |
| --- | --- | --- | --- | --- |
| 1080p or 2:3-source | 9 | 33 plus overlap context | 24 | 512 / overlap 128 |
| UHD 4K | 5 | 9 plus overlap context | 36 | 512 / overlap 128 |

All use CPU offload for DiT, VAE and intermediate tensors; I/O-component offload on; model caches off; no torch.compile; temporal overlap 3 and uniform batches.

One confirmed GPU OOM can trigger ONE fresh-process retry: 5-frame inference, 9-frame chunks, all 36 blocks swapped and 256-pixel VAE tiles with 128 overlap. The retry does not change the model, precision, requested resolution, seed or noise. It restarts that video's inference, not the pod. There is no silent single-frame/quantized/1080p fallback for a requested 4K job. Non-OOM failures are not retried. If the retry also fails, the error and logs remain available.

## Local installation and build

Apply the downloaded update package using its apply_update.py, from your local repository. It verifies the two edited files against the reviewed Git blobs, checks overlay hashes, and backs up original files under .git/workbench-repair-backups. It has no fetch/commit/push commands. A changed source file or an unrelated file collision stops the install before writing.

Review and commit/push in GitHub Desktop. In GitHub Actions run **Build Qwen + SeedVR2 video repair**. This is one combined image, not separate Qwen and restoration images.

The build first runs existing repository CI, the new offline/real-CPU-media tests, and then an actual ComfyUI CPU startup and node-schema smoke inside the built container. It publishes only after these pass. No GPU or model weights are used by those checks.

After a successful build, copy the exact image digest from its summary. The rolling tag is:

    ghcr.io/jsanso1497/runpod-comfy-stateless:qwen-video-hq

The commit-specific tag is qwen-video-<the commit you just built>. The old qwen-hq and H3 tags are not replaced by this action. A tag is not available until the user-triggered build succeeds.

## New RunPod template

Planning recommendation, not a measured minimum:

- 1 x A40 48 GB, or your existing compatible 48 GB GPU.
- Prefer 128 GB system RAM. Do not assume GPU VRAM implies adequate system RAM. Full-precision weights are offloaded to CPU.
- 80 GB container disk.
- 250 GB persistent volume mounted at /workspace. Disk-backed lossless frames can require tens of GB for a 24-second 4K clip. The job calculates a conservative free-disk check before starting.
- HTTP port 8188 only. No additional file-browser daemon or port needed.
- Container start command and entrypoint override empty.
- Use a fresh template, not an old approved-only or H3 template carrying incompatible variables.

Environment rows:

    WB_WORKSPACE=qwen
    WB_QWEN_ONLY=1
    WB_APPROVED_ONLY=0
    WB_TASKS=Q01,RVID
    WB_TOOLBOX=sam,seedvr2
    WB_VIDEO_REPAIR=1
    WB_CLEAN_AFTER_JOB=1
    WB_SEEDVR_FUSED_ONLY=1
    WB_SEEDVR_OOM_RETRY=1
    WB_DISABLE_COMFY_COMPILER=1
    WB_SAVE_METADATA=0
    WB_AUTH_MODE=basic
    WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
    HF_TOKEN={{ RUNPOD_SECRET_hf_token }}

WB_TASKS prepares those assets at startup; it does not hide the other Qwen editing tasks. RVID adds the checksum-verified Sharp FP16 checkpoint to the existing standard FP16 model and VAE. The necessary Qwen encoder is not an optional chat LLM and stays installed.

The new behavior flags are WB_VIDEO_REPAIR, WB_CLEAN_AFTER_JOB, WB_SEEDVR_FUSED_ONLY and WB_SEEDVR_OOM_RETRY. Defaults are already in Dockerfile.qwen-video; include the rows in the template to make the intended configuration explicit.

Advanced preflight overrides exist as WB_SEEDVR_MIN_FREE_VRAM_GB (default 32 for 4K, 20 otherwise) and WB_SEEDVR_MIN_FREE_RAM_GB (default 24). They are guards, not certified requirements. Do not lower them to hide occupied-memory problems.

## Workflows in ComfyUI

Connect through RunPod's HTTP 8188 proxy. Login: workbench and your comfy_password secret. Wait for model preparation to finish.

Workbench > Toolbox > **RVID | SeedVR2 video: isolated streaming FP16** provides:

- Sharp 1080p: full frame, 1920x1080 for a 16:9 source.
- Sharp 4K: full frame, 3840x2160 for a 1920x1080 source. True SeedVR2 4K restoration, no conventional-resize substitute.
- Sharp 2:3: preserve left edge, crop right side BEFORE restoration, restore to 1080x1620 and return to cropped source size. A 768-high source finishes at 512x768. Heights not divisible by 6 lose at most 5 bottom rows for exact even 2:3. Narrower-than-2:3 sources are rejected, not stretched.
- Standard FP16 1080p: control comparison, same processing but original 7B checkpoint.

All default to 24 seconds starting at 0. Duration 0 means the remainder. Source shorter than the requested duration is rejected clearly. There is no 9-frame/33-frame final clip limit: those are internal streaming chunks. The result is one joined output with source-rate timing and audio, not many tiny clips.

For the first live validation, set duration_seconds to 2, inspect the result and memory report, then restore 24. This is a test choice, not a hidden output cap. Do not run an external terminal GPU renderer while ComfyUI jobs run.

## Files and logs without File Browser

Use the ComfyUI sidebar tab **Files and video jobs**. It links to /_workbench/repair on the same 8188 hostname.

- Saved files reads /workspace/output recursively, independently of transient execution history or an optional asset database. It does not recreate Comfy's old job history.
- Video jobs shows exact job state, failing stage, error and log downloads.
- Memory/errors shows GPU process owners, container RAM limits/events, last cleanup and recent Comfy execution errors. The unload button refuses to run when work is queued.

Final MP4s: /workspace/output/video/seedvr_<mode>_<id>/restored.mp4
Video job logs: /workspace/workbench/video_jobs/<job-id>/
Comfy log: /workspace/workbench/logs/comfy.log and rotated copies.

Completed outputs and logs are not deleted by memory cleanup. Successful intermediate frames are removed by default to save disk; turn on keep_intermediates to retain them. Failed jobs retain their own intermediate files and diagnostics.

If only encoding failed after all PNGs were written, an optional CPU-only recovery command is:

    python -m workbench.seedvr_job --job /workspace/workbench/video_jobs/JOB_ID --finish-only

Use the real job folder from the diagnostics page. This validates all frames and does not start model inference. It refuses to overwrite a completed video.

## Data persistence and limitations

Keep important files under /workspace. A regular volume survives restart on the same pod but does not automatically appear on a new pod. Download or transfer old outputs before deleting the current pod. Attach the correct network volume during deployment when reusing one.

This is SDR processing. PQ/HLG inputs and non-square pixels are rejected rather than silently tone-mapped. VFR input is normalized to constant frame rate using its average source rate. Output is H.264 CRF 16, 8-bit YUV420 with explicit BT.709 conversion/tags and AAC audio. It is not a lossless or HDR delivery format.

No code can promise that full FP16 4K inference fits every A40 input. System RAM, available VRAM, GPU kernels and scene/frame count matter. The repair makes errors attributable and cleans up owned worker lifetimes; it does not create memory. GPU execution, speed, peak memory and visual quality remain untested by the package's offline validation.

## Sources verified during development

- Current repository commit f310e8f119b0bdf853ffefd1153ab6459a66d567.
- Pinned ComfyUI 52f98af2e2e42c421070a3e147c161c47cdeaf22.
- Pinned SeedVR2 integration 4490bd1f482e026674543386bb2a4d176da245b9.
- https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (streaming, offloading, attention, tiling).
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html (unused allocations vs live tensors).
- https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_7b_sharp_fp16.safetensors (published SHA256).
- https://docs.runpod.io/pods/storage/types (volume persistence).

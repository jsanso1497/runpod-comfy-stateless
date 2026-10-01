# Stateless video-restoration first test

Read **START_HERE.md** for the deployment steps.

## Selection

The default is SeedVR2 7B FP16, standard rather than the extra-sharp variant.
It is a practical quality-first candidate for restoring a noisy or mildly blurred
1080p clip and testing a 2x upscale to 4K. It is not asserted to be the universal
best model, and no actual source clip has been examined or restored here.

SeedVR2 is a generative video-restoration model. Its official ComfyUI integration
also provides a standalone CLI with temporal overlap, streamed chunks, CPU
block/offloading, VAE tiling, model caching and attention-backend options. This
package wraps that CLI with upload, trimming, comparison and validation logic.

FlashVSR is a relevant speed-oriented alternative, but its published recommended
operating point is 4x upscaling with its specific sparse-attention implementation.
That is not the direct 1080p-to-4K 2x first test selected here. A definitive quality
ranking requires running the actual clip through competing methods.

This version deliberately avoids a chain of multiple generative models, face
replacement, face enhancement, frame interpolation and LoRAs. Input-noise and
latent-noise injection are zero. Those settings reduce unnecessary sources of
variation; they do not guarantee faithful identities or absence of hallucinations.

## Pipeline

1. Probe the source and reject unsupported resolution, timing and color cases.
2. Select a short frame-aligned segment at the original rational frame rate.
3. Normalize to an 8-bit RGB FFV1 lossless intermediate without spatial resizing.
4. Optionally apply an explicitly selected gentle gamma adjustment.
5. Render a conventional reference using mild hqdn3d and unsharp filtering.
6. Run SeedVR2 at 2160 short edge / 3840 long edge, or at native 1080p.
7. Encode 4K and/or 1080p H.264 CRF16, slow preset, square pixels, Rec.709 SDR.
8. Restore the selected first audio track, encoded AAC 192 kbit/s.
9. Verify frame count, frame rate, output dimensions, sample aspect and audio
   presence. Create an original-left comparison and a matched center crop.
10. Save a reproducibility report and delete bulky intermediate frames.

The 1080p file in the 4K mode is a downsample of that same AI result. The standalone
1080p mode is available for a separate native-resolution experiment. The baseline
is not fed into the AI by default, to avoid blindly stacking denoisers.

## Reproducibility and startup

The Dockerfile reuses the existing published CUDA 12.9 / Torch 2.9.0 / SageAttention
foundation by immutable digest. It pins the SeedVR2 integration to:

```
4490bd1f482e026674543386bb2a4d176da245b9
```

Build-time constraints protect the Torch/torchvision ABI and selected ML packages.
Other transitive packages are resolved at build time and recorded by pip freeze;
this is NOT a complete independently reproducible transitive lockfile. Deploying
the built image by digest fixes that resolved environment for subsequent Pods.

Startup never updates source code or pip packages. A newer model or library should
be evaluated in a new image rather than silently changing an existing test.

The model manifest lists exact SHA-256 hashes from the upstream model registry.
Weights are downloaded from public Hugging Face repositories to ephemeral disk
and checked before use. Only the chosen model and shared VAE are downloaded.
No credentials, input footage or model files are included in this ZIP.

`RESTORE_ATTENTION=sdpa` is the first-test default. The inherited SageAttention
wheel is intentionally removed because its compiled extension is not ABI-compatible
with the pinned Torch build used by this image. SeedVR2 supports PyTorch SDPA natively,
so this avoids a build-time/runtime failure while keeping the restoration test valid.
Once the baseline works on your footage, SageAttention can be rebuilt from source
against the exact Torch/CUDA stack and benchmarked separately for speed.

## Conservative initial resource presets

| GPU memory | Frames per batch | Frames per streamed chunk | VAE tile | 7B blocks swapped |
| --- | --- | --- | --- | --- |
| About 80 GB | 9 | 33 | 1024 | 0 |
| About 48 GB | 5 | 25 | 768 | 16 |
| Below 40 GB | 5 | 17 | 512 | 32 |

All use temporal overlap 3, VAE overlap 128, CPU offload between phases, CPU tensor
storage, and cached models across chunks. Exact settings appear in report.json.
These are initial safety-oriented presets, not measured optimal throughput values.
One clean retry is permitted for a recognized CUDA out-of-memory or Sage kernel
failure. The report discloses the fallback. Persistent failures stop the job.

Torch compilation is disabled for the short disposable first test to avoid adding
unmeasured compilation startup cost. It can be benchmarked later for longer runs.

## Configuration

Required:

```
RESTORE_PASSWORD=<a private password of at least 12 characters>
```

Defaults:

```
RESTORE_USERNAME=james
RESTORE_MODEL=7b
RESTORE_ATTENTION=sdpa
RESTORE_PORT=8188
RESTORE_HOME=/workspace/video-restore
RESTORE_MAX_UPLOAD_GB=5
RESTORE_MAX_JOB_SECONDS=7200
```

`RESTORE_MODEL=3b` selects the included smaller FP16 model manifest instead. This
requires a fresh Pod/config change; do not restart a disposable Pod before saving
its output. Increasing job limits does not provision more VRAM or disk.

The GPU process has a default two-hour limit, but the Pod is NOT automatically
terminated. Cancellation and process timeouts do not stop infrastructure billing.

## Files and security

This is a private single-user Python standard-library HTTP utility. It provides
Basic authentication behind the RunPod HTTPS proxy, CSRF checks, upload size
limits, generated file IDs, an output allowlist, range downloads and no arbitrary
shell-command input. It is not an internet-scale hardened application server.

The HTTP process itself does not load Torch, keeping CUDA allocations out of the
web server. Hardware checks and the engine run in separate processes. One render
runs at a time. Only the operator's Pod performs inference; model download requests
do not include uploaded footage. This is not a guarantee about the cloud host's
retention, encryption or compliance. Apply your normal provider approval rules.

The application does not start the old ComfyUI, Jupyter or SSH services. The base
image still contains those tools. Only expose HTTP 8188 in this template.

## Validation and troubleshooting

See VALIDATION.txt. Local tests are run with:

```
python -m pytest -q tests/test_restore.py
```

They require Python, pytest, FFmpeg, Pillow and NumPy. The test suite does not require
model weights or a GPU. Fake AI tests are explicitly marked and do not establish
actual model quality or runtime compatibility.

Docker build checks include pip check, actual CLI --help imports, supported flags,
manifest agreement with the pinned source and required FFmpeg filters. Actual
GPU/model checks occur only on the deployed Pod.

If the new GitHub build fails, inspect its first failing step rather than launching
a Pod. If startup fails, use the RunPod log. If a render fails, download run.log
and report.json. Keep original filenames and frame/timing metadata when reporting
an issue. No GPU speed estimate is promised without measuring your actual clip.

## Primary references

Research and implementation sources checked during preparation:

- SeedVR2 official research repository: https://github.com/IceClear/SeedVR2
- Official integration and standalone CLI: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- Pinned implementation: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/tree/4490bd1f482e026674543386bb2a4d176da245b9
- Model registry: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/4490bd1f482e026674543386bb2a4d176da245b9/src/utils/model_registry.py
- FlashVSR authors' implementation: https://github.com/OpenImagingLab/FlashVSR
- FFmpeg filters: https://ffmpeg.org/ffmpeg-filters.html
- RunPod templates: https://docs.runpod.io/pods/templates/create-custom-template
- RunPod secrets: https://docs.runpod.io/pods/templates/environment-variables
- RunPod storage: https://docs.runpod.io/pods/storage/types
- GHCR image digests: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry

Upstream software and model license terms apply. SeedVR2 code and its integration
are distributed under their upstream licenses, retained in the cloned repository
inside the image. Review model/provider terms for your intended commercial use.

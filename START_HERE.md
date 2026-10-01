# Start here: video restoration test

This update turns the existing RunPod image into a focused video-restoration test.
It opens a password-protected upload-and-compare page on port 8188, not the ComfyUI
node canvas. It uses the standalone interface from the official SeedVR2 ComfyUI
integration. Keep the original RunPod template as a fallback.

## 1. Upload this update to the existing GitHub repository

1. Unzip this package on your computer.
2. Open `jsanso1497/runpod-comfy-stateless` in GitHub.
3. Choose Code > Add file > Upload files.
4. Upload the CONTENTS of the extracted folder to the repository root. Replace the
   root Dockerfile and README.md; merge the scripts, config and tests folders.
   Do not upload the ZIP itself or an extra enclosing directory.
5. Commit directly to main.

Keep `.github/workflows/build-image.yml` already in the repository. This package
intentionally does not replace that working workflow. Changing Dockerfile triggers
its existing build. No Docker installation is needed on your computer.

## 2. Wait for the new image build

Open Actions and the newest build. After it succeeds, copy the NEW digest from its
Build and push step. This package is source code, not an already-published image.

The Dockerfile's existing sha256:e444... image is the OLD foundation used to build
this update. Do NOT reuse that digest as the new restoration template image.

The new RunPod image reference will have this form:

```
ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:NEW_DIGEST_FROM_SUCCESSFUL_BUILD
```

Replace the entire placeholder with the new digest's 64 hexadecimal characters.
Do not use :latest. Keep the GHCR access/public-package settings that worked for
the original image. The GitHub workflow also needs permission to pull that base.

## 3. Create a separate RunPod template

Name: Video Restoration Test

| Setting | Value |
| --- | --- |
| Container image | The NEW digest reference from step 2 |
| Container disk | 150 GB |
| Volume disk | 0 GB |
| Network volume | None |
| HTTP port | 8188 |
| TCP ports | None needed |
| Container start command | Leave blank |

In RunPod Secrets, create `restore_password` with a unique private password of at
least 12 characters. Add these environment variables to the template:

```
RESTORE_PASSWORD={{ RUNPOD_SECRET_restore_password }}
RESTORE_USERNAME=james
RESTORE_MODEL=7b
RESTORE_ATTENTION=auto
```

Use the secret selector where available. Do not put the real password in GitHub.
No Hugging Face token, CivitAI token, LoRA or custom-node list is needed.
Old COMFY_REF, CONFIG_REPO and ComfyUI-specific settings are not used by this app.

## 4. Deploy the first test

Use one NVIDIA A100 80 GB for this quality-first test. Prefer 128 GB system RAM
when selecting a host. This is a conservative starting recommendation, not a
measured speed or cost guarantee. A 48 GB GPU uses more CPU offloading; a 24 GB
GPU is not the preferred starting point for 7B restoration to 4K.

Check the quoted hourly total before deploying. The image uses CUDA 12.9; the host
must have a compatible NVIDIA driver. Startup executes actual CUDA operations.

Open Connect > HTTP service 8188 using RunPod's HTTPS link.
Log in with username `james` and your chosen password.

The page can open before startup finishes. Wait for Ready. Startup checks the GPU,
tests the attention implementation, downloads the 7B model plus VAE, and verifies
both SHA-256 hashes. Downloads happen again on a fresh disposable Pod.

## 5. Run a three-second comparison

1. Upload your original 1920 x 1080 video. Use a short continuous shot for the first
   trial. If a very large upload fails at the network/proxy layer, use a shorter
   source excerpt rather than repeatedly uploading the entire recording.
2. Choose a start time showing a face or fine detail, noise, and some movement.
3. Leave Duration at 3 seconds.
4. Leave Test mode at `4K restoration + 1080p downsample`.
5. Leave Tonal adjustment at `None` for the first run.
6. Click Run comparison test.

No image generation prompt, LoRA or model selection is required.

The first test uses one AI restoration to 3840 x 2160, then downsamples that result
to 1920 x 1080. It is not two AI passes. This lets you judge quality improvement at
matched 1080p size separately from the bigger 4K frame.

To test restoration without increasing resolution, choose `1080p restoration only`.
For a second low-light comparison, try `Gentle gamma lift`. This changes overall
tonality intentionally; it is not a dedicated learned low-light exposure recovery
model. Compare it against the no-lift result before choosing.

## 6. Review, download, then terminate

The results include:

- Original and conventional cleanup reference at 1080p.
- AI restoration at 4K and a matched 1080p downsample.
- Original-left / AI-right side-by-side video and a lossless matched detail crop.
- report.json with frame counts, rate, size, pixel aspect, audio presence, settings,
  model hashes, hardware information and commands; plus run.log.

The original preview is a high-quality re-encode, not a byte-for-byte source copy.
The first audio track is preserved in timing and re-encoded as AAC, not copied
bit-for-bit. The original uploaded source is not overwritten.

Watch at normal speed and inspect faces, hair, fingers, text, fine fabric and moving
edges. Reject invented detail, flicker, waxy skin, halos or changed identities.
The side-by-side export is more reliable for exact synchronized comparison than
pressing Play in two browser players separately.

Download everything you need BEFORE stopping or terminating the Pod. Models,
uploads and outputs are disposable. Terminate the Pod when finished. Closing the
browser, cancelling a render, or finishing a job does not stop RunPod billing.
This app does not automatically terminate the Pod.

## Important limits

- This first profile is SDR, progressive, square-pixel, constant-frame-rate 1080p.
  HDR, interlaced, rotated/anamorphic and irregular-frame-rate inputs are rejected.
- The selected engine's video I/O is 8-bit RGB. This is not a true HDR or 10-bit
  mastering pipeline, even when the source is a 10-bit SDR file.
- AI cannot guarantee faithful recovery of details absent from the source.
  Severe motion blur, missed focus and clipped shadows may not be recoverable.
- The application accepts at most 120 seconds per test and asks for confirmation
  above 15 seconds. Start with 3 seconds, not the entire recording.
- The application is a private single-user test tool, not a multi-tenant production
  web service. Use only the HTTPS proxy; do not expose its raw HTTP port publicly.
- Do not upload confidential client or personal footage unless the account,
  provider, storage and workflow are approved for that footage.

## What has actually been tested

See VALIDATION.txt. CPU tests exercised actual FFmpeg processing, audio handling,
frame-rate/size checks, upload validation, authentication and HTTP behavior. A fake
AI renderer tested file plumbing only. No real SeedVR2 restoration, GPU throughput,
VRAM peak, Docker build or RunPod deployment was executed in the assistant's
current environment. GitHub's new build and your three-second GPU trial are the
remaining acceptance checks.

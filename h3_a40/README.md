# H3 A40 Identity Studio

Version 2026-11-13.a40.1. Focused MiniMax H3 VIDEO model to a single still PNG, with an Eros selector. This is not P-H3 or MiniMax-H3-Image.

## Recommended starting profile

Use `00_H3_A40_Identity_Quality.json`. Select a clear 3.1-second section of the reference video. With enough source footage this retains 73 frames at 24 fps, not just the four preview thumbnails. The native resolution control is 1.0 MP; the unchanged upstream conditioning can further resize to its 768-short-edge canvas. The encoding report records the effective dimensions.

The output is 2.5 MP, batch size one. Standard H3 uses 50 ER-SDE/simple steps, no Turbo adapter. The same model dropdown can select Eros Beta 7 Fix1, using its baked eight-step Turbo with Euler/simple and 12/3 sigma shifts. Two optional H3-compatible model-only LoRA slots remain available. The model family and matching key checks cannot guarantee a LoRA's visual compatibility.

For identity work, use `01_H3_A40_Video_Plus_Photos.json` with a sharp face/hair photo in slot 1 and a body/proportion photo in slot 2. This is a suggested allocation of the reference budget, not a measured proof that it outperforms video-only. Choose consistent, useful views of the same person. Crop unnecessary background before generation; do not crop away needed anatomy. Review the preflight contact sheet and tag mapping before rendering.

`03_H3_A40_4MP_Final_Test.json` changes output MP to 4.0. It does not raise the video reference to 2 MP. A larger output is not a guarantee of better likeness and has not been qualified on an A40 here. Keep duration fixed when comparing output resolution. For a lower-workload test, reduce `native_video_megapixels` to 0.5 while retaining the same precision, photographs, prompt and seed.

## What changed

The profile has independent node names and clean native-video preparation. It does not import the previous 2 MP conditioning override or uncapped media code. The original 2 GiB decoded-pixel check is restored. The duration UI offers up to six seconds; that is a profile choice, not a hardware or model maximum.

Model storage precision is INT8 diffusion, INT8 Qwen3-VL-32B encoder and FP16 video VAE. ComfyUI manages activation precision. This is deliberately not a four-bit encoder or a BF16-full configuration. Encoding and generation are separate nodes. Ordinary ComfyUI execution caching can reuse encoded references for seed-only or LoRA-only reruns. Input or settings changes can cause re-encoding; restarting clears this cache. This is NOT Spectrum/TeaCache or approximate denoising reuse. Encoder/VAE GPU weights are offloaded after encoding to leave room for diffusion. Normal model offloading remains enabled.

The template downloads four assets, approximately 76.2 decimal GB / 71.0 GiB, excluding the container software. They are the standard REF2VA INT8 diffusion checkpoint, Eros Fix1 protected-edges HQ INT8 checkpoint, INT8 32B encoder and FP16 video VAE. P-H3, BF16 diffusion/encoder sets, FL2VA and extra Turbo adapters are not downloaded. Existing verified model files are reused after checksum verification, never deleted. An inherited `H3V_PRELOAD=all` does not change this new template's four-file policy.

Photo references, video-plus-photo references and masked person replacement are included. In replacement mode, photo 1 is the target, photo 2 the face and photo 3 the body. The mask protects original pixels outside the selected area by final compositing. It does not condition the diffusion model as an inpaint mask; pose drift and mask-boundary artifacts can still occur. Text-only generation is intentionally not part of this reference-only template. The older image tags and source folders are not modified.

## Local repository installation and new template

Save `update_h3_a40_identity.py` in Downloads. From the repository terminal:

```bash
python3 "$HOME/Downloads/update_h3_a40_identity.py" --repo .
```

The installer accepts local Python 3.7+, adds only the new `h3_a40/` directory and `.github/workflows/build-h3-a40-identity.yml`, and refuses conflicting files. It does not stage, commit, push, or write remotely. Review and commit/push manually through GitHub Desktop. In GitHub Actions run **Build H3 A40 Identity Studio**. Use the actual digest printed by a successful run, not an invented digest or the old A100 image tag.

See `RUNPOD_SETTINGS.txt`. Leave container start/entrypoint overrides blank. Use a CUDA 13-compatible A40 host with R580 or newer NVIDIA driver. Target 128 GB system RAM or more and 160 GB or larger persistent storage. These are planning recommendations, not measured minimums. The A40 has 48 GB VRAM; system RAM and disk are different resources.

Open Workflows > **00_H3_A40_Identity**. The inherited login gateway may still display an older Studio version. The dedicated workflow folder and the image digest identify this profile. Login, files, Jupyter and the existing origin policy are unchanged.

## Test on a currently running H3 Video Still Studio Pod

The new JSONs require the new custom nodes; importing the JSON alone is insufficient.

Upload the same installer into `/workspace`. After the generation queue is idle:

```bash
python3 /workspace/update_h3_a40_identity.py --pod --restart
```

This adds the independent A40 nodes and workflow folder and restarts only ComfyUI. It requires the four model files to be present already. It does not download weights, stop the Pod, change authentication or switch the physical GPU. On the old image, this live install does NOT replace the original startup downloader. Build the new image for the four-file-only template.

The existing 2 MP experiment files are left intact but are never used by these new nodes. This avoids overwriting prior work and also avoids mixing the two experimental patches with the normal path.

## Persistence and rollback

Before stopping or replacing the old Pod, verify that `/workspace` is genuinely persistent. The earlier `overlay ... /` result suggested the data was on container storage. Check:

```bash
findmnt -T /workspace -o TARGET,SOURCE,FSTYPE
```

Back up needed input videos, outputs and workflows before discarding that Pod. A network volume can be shared between Pods; container-disk storage is not a safe migration location. The new profile never deletes the old 375 GB collection automatically. Reusing a volume that already contains those files does not shrink it.

Local repository rollback, only while added files remain unchanged:

```bash
python3 "$HOME/Downloads/update_h3_a40_identity.py" --repo . --restore
```

Live-Pod rollback:

```bash
python3 /workspace/update_h3_a40_identity.py --pod --restore --restart
```

Rollback removes only unchanged files added by the installer. It refuses to remove files you modified. Saved personal copies and all model weights remain untouched.

## Validation boundary

Local tests cover graph connections and input schemas, model routing, download scope, actual FFmpeg decoding, frame counts, restored memory guard, mask protection, and synthetic encoding handoff. The installer is tested in temporary repositories. No A40 or A100 image generation was run for this release. Peak VRAM, generation time, likeness, model loading, Docker build and real ComfyUI runtime behavior have not been measured here.

The supplied GitHub Action builds the container, verifies publisher metadata, starts real ComfyUI on CPU, runs media preflight and upstream contracts with synthetic encoder/VAE fixtures, and exercises the inherited browser gateway before publishing. Those CI checks have not run merely because this package was downloaded. Even a passing CPU build is not GPU qualification.

The A40 configuration is a fidelity-oriented starting point with a greatly reduced reference workload, not a proven global optimum. Standard H3 at 50 steps may still be slow. Eros is the faster alternative by sampling configuration, not a promise of better identity retention. Do not interpret step count alone as a quality score.

## Primary references

- Comfy-Org model files and precision guidance: https://huggingface.co/Comfy-Org/MiniMax-H3
- Pinned stock native conditioning: https://github.com/Comfy-Org/ComfyUI/blob/a592c0a654af22f75d97f214d6e3e34eb13cd9db/comfy_extras/nodes_minimax_h3.py
- Eros creator's sampling guidance: https://huggingface.co/TenStrip/10Eros-Max
- Single-frame latent/decode implementation: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still
- A40 specifications: https://www.nvidia.com/en-us/data-center/a40/
- CUDA 13 driver compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- RunPod storage: https://docs.runpod.io/pods/storage/types

Use models only with applicable authorization. The inherited license-confirmation requirement remains; see LICENSE_NOTES.md. No prompt enhancer or external generation service is included.

# ComfyUI + SeedVR2 update

This update removes the custom restoration web UI and makes normal ComfyUI the
service on port 8188.

## What is prepared

- Current ComfyUI is installed at Pod startup.
- The already pinned SeedVR2 v2.5.24 source from the parent image is copied into
  ComfyUI as a custom node. No Git pull is needed for SeedVR2 at Pod startup.
- SeedVR2 7B FP16 and the FP16 VAE download automatically to ephemeral container
  storage and are SHA-256 checked.
- Two workflows are installed into ComfyUI:
  - `00_SeedVR2_4K_Video_Restore.json`
  - `01_SeedVR2_1080p_Video_Restore.json`
- The workflows use ComfyUI's native video load/component/create/save nodes so
  audio and source FPS can flow through the graph.
- PyTorch SDPA is used intentionally. SageAttention was removed from the parent
  image because that wheel was ABI-incompatible with the pinned Torch build.

## First test

Use `00_SeedVR2_4K_Video_Restore.json` on the A100 80 GB Pod.

1. Open ComfyUI on HTTP port 8188.
2. Open Workflows and select `00_SeedVR2_4K_Video_Restore`.
3. In `Load Video`, upload the 1920x1080 source.
4. Queue the workflow.
5. The SeedVR2 node targets a 2160-pixel short edge with a 3840-pixel maximum
   long edge, which gives 3840x2160 for normal 16:9 footage.
6. The workflow preserves the video's source FPS and passes its audio to the
   created output video.

The 4K workflow starts conservatively at batch size 9, temporal overlap 3,
VAE tiling 1024/128, standard 7B FP16, LAB color correction, no added input or
latent noise, CPU tensor offload, and SDPA attention.

The 1080p workflow keeps the same restoration model but targets 1080 short edge
with a 1920 maximum long edge and uses batch size 17.

## GitHub update

Upload the contents of this folder over the repository root. You can leave your
existing `.github/workflows/build-image.yml` in place. Commit the changes and let
the existing action build a new image.

When it finishes, copy the NEW image digest and use that digest in the RunPod
template. Do not keep using the previous `56b51c...` digest after rebuilding.

## RunPod template changes

The registry credential stays the same because the GHCR package is private.

The old restoration-specific environment variables are no longer required:

- `RESTORE_PASSWORD`
- `RESTORE_USERNAME`
- `RESTORE_MODEL`
- `RESTORE_ATTENTION`

You can delete them from the template. Add or keep:

```
COMFY_REF=master
ATTENTION_BACKEND=pytorch
VRAM_MODE=auto
ENABLE_DYNAMIC_VRAM=auto
ASSET_DOWNLOAD_WORKERS=2
```

Port 8188 remains the only HTTP port required for normal use.

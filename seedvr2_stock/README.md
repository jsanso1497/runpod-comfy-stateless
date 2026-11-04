# SeedVR2 Video Clarity

This is the single maintained SeedVR2 video setup for this repository.

**Build:** Build SeedVR2 Stock FP16  
**Image tag:** `seedvr2-stock-fp16`  
**Workflow:** `00_SeedVR2_Stock / 01_Video_Clarity`

Upload ordinary 1920 x 1080 video, optionally select a native crop, run the official
SeedVR2 nodes at the selected region's existing size, then download the native
MP4 and first-frame comparison PNG. No forced 4K pass or custom renderer.

See [SEEDVR2_START_HERE.md](../SEEDVR2_START_HERE.md) for the one installation
procedure, exact crop coordinates, RunPod settings, safeguards and test limits.

ComfyUI and the official SeedVR2 extension remain unmodified. Standard Nginx
password protection is retained. Checks under `checks/` run only during build/CI;
they are not additional runtime services or custom ComfyUI nodes.

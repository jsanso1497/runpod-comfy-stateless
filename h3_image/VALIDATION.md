# Validation: Native H3 Image manual workspace v2

Prepared November 10, 2026. Package release 2026-11-10.2.

## Executed locally

- 58 workflow/helper/control unit tests passed. Tests cover deterministic graph regeneration, link types and dependencies, blank literal prompts, fixed-seed defaults, source-slot geometry, actual input ordering, six-step schedule validation, non-stretched comparisons using small real tensors, direct invocation of stand-in native sampler/decode interfaces, saved JSON reports, queue guards, and streamed ZIP64 export with file/symlink/disk-space protections.
- 18 installer tests passed against temporary real Git repositories. Output is in the accompanying installer test log. They cover clean installation, recognized .1 upgrade, conflict refusal before mutation, backups, idempotence, Python 3.8 parsing, symlink refusal, no staging/commit/push, and preservation of unrelated workspace files.
- Actual HTTPS gateway authentication, wrong-password rejection, authenticated API proxying, chunked uploads/downloads, duplicate-file protection, origin checks, Jupyter contents API and a real shell command through Jupyter's secure WebSocket proxy passed. Local library versions were JupyterLab 4.5.3 and Jupyter Server 2.17.0, not the exact pinned container versions.
- The full Chromium browser attempt was blocked at its first navigation by this environment: `net::ERR_BLOCKED_BY_ADMINISTRATOR`. This is not a passing browser-render test. The protocol test does not prove that the user's browser or RunPod proxy will behave identically.

## Not executed locally

- No Docker daemon/build was available.
- Outbound container DNS was unavailable. Live network reads used the web/GitHub tools; model metadata could not be fully fetched into the container. Model LFS revisions, exact file sizes and checksums are resolved by the GitHub build before constructing/publishing the image. No placeholder checksum is used.
- No full weight download, native GPU generation, real-likeness comparison, A40/A100 memory-capacity benchmark, speed benchmark, or live RunPod connection was performed.
- Native ComfyUI numerical smoke tests require the pinned source and dependencies inside the image. They have not run locally. Stand-in CPU tests are not presented as equivalent evidence.

## GitHub Action gates included

The new build action runs source checks, freezes exact model revisions and hashes, builds from pinned ComfyUI commit `ede4e5fe8d15445c6fcd6828d70b202c22f9502c`, starts native CPU ComfyUI with Manager, verifies all workflow input names against actual registered schemas, and exercises native conditioning/sampling geometry with fake encoder/VAE objects. It also performs the existing full Chromium/JupyterLab/terminal access test in the built image. Publication occurs only after those gates pass. Their inclusion here does not mean the user's Action has already run or passed.

The selected commit contains the required image implementation even though its version string is still 0.44.0 development. It is not interchangeable with the 0.44.0 release tag. No unverified 0.45.0 SHA is invented.

## Model behavior intentionally unchanged

- Native MiniMax-H3-Image pruned INT8 diffusion, shared H3 NVFP4/AWQ encoder, native image FP16 VAE.
- Exact uploaded six-step sigma sequence, Euler, BasicGuider, full denoise.
- No prompt rewriting, extra prompt model, negative prompt, legacy video LoRA, or replacement of native inference algorithms.
- Output default 4 MP; fixed seed; one result per run.
- Timing wrappers forward the same native parameters. These tests do not prove identical numerical GPU output.
- Comparison PNGs are separate, downscaled review artifacts. They do not change the final PNG.
- Background preservation/SAM is not silently claimed. These graphs perform whole-image reference-guided generation.

## Release safety

Local installer touches only `h3_image/` and `.github/workflows/build-h3-image.yml`, with recognized-version upgrades and backups. No remote write, commit or push is performed. Old H3 Lab, Eros, Qwen and model files are not overwritten. Runtime downloads verify each asset, reuse only checksum-identical encoders, and refuse to overwrite conflicting files.

Queue-safe controls do not interrupt work without explicit authorization. Backend/supervisor singleton locks prevent duplicate launches. A restart frees only the stopped process's resources; it does not increase available physical memory. No test establishes that 16 MP or 16 references fits an A40.

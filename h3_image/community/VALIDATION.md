# Validation record

Package: Native H3 Image Community Tests, 2026-11-10-community.1

## Executed locally

- Workspace plus community unittest suite: **105 tests run; 104 passed and 1 skipped**. The skipped test requires the real ComfyUI installation inside the built image.
- Local repository updater: **17 tests passed**.
- Python compilation: updater, optional downloader, workflow generator, native smoke script and regression tests passed.
- All nine GUI workflows and nine API templates regenerated deterministically and their linked node inputs/outputs, topological order, types and widget values were checked.
- Blank manual prompts, fixed seeds, unchanged INT8 base/encoder/native VAE, correct picture ordering, reference counts, six/four-step choices, simple scheduling, and LCM's explicit noise controls were checked.
- The decoder comparison has one sampling node, one shared sampled latent, separate native/Pruna decoding branches, and separate full-resolution outputs. Preview downscaling does not modify the full-resolution saved images.
- The optional decoder downloader was tested using small synthetic files, including successful checksums, wrong checksums, existing-file protection, persistent reuse without modification, low storage, unexpected paths and symlink traversal checks. These are not full-weight download tests.
- Installer tests cover Python 3.8 syntax, wrong-directory help, explicit repository selection, dry-run behavior, all-path preflight, additive-only scope, idempotence, conflicts before writes, symlink protection, completed-write rollback, no staging and no model/image payloads.
- The supplied JPEG was inspected using Pillow metadata plus byte-level scanning. It contains no recoverable ComfyUI workflow or prompt; its EXIF has only ordinary image information. See source_image_metadata.json.

## Checked against source

Existing user repository reviewed at `8323001d15aae2f41e7446d25b1846f998c3bf5b`. The local baseline builder, runtime backend and baseline test file matched the remote Git blob IDs inspected for this update. No remote changes were performed.

The new graphs use native BasicScheduler, KSamplerSelect and SamplerLCM interfaces from the workspace's existing ComfyUI pin. SamplerLCM exposes `s_noise`, `s_noise_end`, and `noise_clip_std` in that pinned source. Existing native helper nodes are reused; no new custom node or dependency is installed.

The optional Pruna INT8 VAE's size, checksum and file commit were confirmed from its publisher's Hugging Face file page and commit. The source screenshots do not establish which precision the commenters used.

## Deferred to GitHub build

The existing native CPU smoke test checks every added API template's input names against actual registered nodes. The new build-only test additionally constructs the real four/six-step simple schedules and Euler/ER-SDE/LCM sampler objects on CPU, without loading weights. It records the actual six-step simple schedule's difference from the uploaded schedule rather than assuming byte-identical floating-point values.

Existing browser, authenticated gateway, Jupyter and terminal checks remain unchanged. This update does not skip or replace the previous CPU bootstrap repair.

## Not executed here

- The new Docker build and its native-core constructor check.
- Live RunPod installation or proxy access.
- The 3.15 GB decoder download or full-weight verification.
- H3 or Pruna GPU inference or real image decoding.
- Facial likeness, expression fidelity, sampler preference or image-quality improvements.
- Peak GPU/container RAM at 16 MP or maximum reference capacity on A40/A100.
- Recovery of the author's original graph, prompt, seed, model precision or VAE version from the resized JPEG.

A passing schema or synthetic safety test does not prove model compatibility, output quality, memory capacity or a speed improvement. Test one change at a time on the Pod.

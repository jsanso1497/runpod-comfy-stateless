# Qwen Mobile + Jupyter validation

Source baseline: 05e957f3381e3d0e8ffa8fc1b7d7a9cd96a429fb.

## Executed locally

- 37 focused pytest checks passed for graph, catalog and Jupyter lifecycle/security contracts.
- Native ComfyUI graph: 10 core nodes, 12 consistent links, one image input, one saved image output.
- Five App Mode widget selections resolve to declared widgets. App Mode is the default view.
- Exactly three existing BF16 model files, with no enhancer, LoRA, SAM or restoration dependency in QAPP.
- Native image-conditioning latent feeds KSampler directly; no mismatched empty-latent canvas.
- Sampling: 25 steps / CFG 1 / Euler / simple / denoise 1. Lossless/default KV-cache precision.
- Local real Jupyter HTTP/API/WebSocket smoke passed 9 checks. Detailed results and package
  versions are recorded in qwen-mobile-jupyter-local-integration.json.
- 9 updater checks passed: dry run, full apply, backups, idempotence, source conflicts,
  new-file conflicts, CRLF compatibility, symlink rejection and rollback after a simulated write failure.
- Python syntax compilation passed for added/modified Python source.
- Original Dockerfile and config.py reconstructed from connector reads match their Git blob IDs.
  Only the entrypoint/build/port hook and additive catalog/secret-stripping hook are changed.

## Not executed here

- No Docker build or registry publication.
- No full baseline repository test suite. Existing CI must run after local commit/manual push.
- No real ComfyUI frontend rendering, iPhone browser session, RunPod proxy or GPU inference.
- No model weights downloaded, quality benchmark, peak-memory measurement or cost benchmark.
- The local Jupyter smoke used installed versions 4.5.3 / 2.17.0, not the Docker pins 4.5.11 / 2.21.1.

Source/contracts inspected: ComfyUI native Qwen node definitions at the pinned core revision;
ComfyUI frontend App Mode store at 1.55.17; official App Mode and Qwen 2.1 documentation;
Jupyter server authentication/configuration documentation; published PyPI release listings.

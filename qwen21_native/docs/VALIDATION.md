# Validation: v1.1.0

## Executed here

- 99 local tests passed, including all 38 UI/API workflow pairs.
- 65 lightweight CI tests passed with imports of `torch` explicitly blocked.
- All 12 custom helper schemas match the workflow generator's socket types and widget order.
- Workflow generation was rerun and checked for deterministic output.
- Python compilation passed for scripts and custom nodes.
- Bash syntax passed for the entrypoint.
- Installer tests confirm that old workflows, current model files and personal edits are retained.
- Model selection checks: three native weights, five with SeedVR2, and exactly two additional adapters only with BFS opt-in.
- Controlled A/B checks confirm identical seed, prompt, crop settings, reference budget and sampler within each native/BFS comparison.
- Eight contiguous image slots, invalid-reference errors, mask alignment, hard pixel protection, and outside-blend audits were tested.

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, CUDA unavailable. Tests do not
require model weights. Docker targets the previously selected PyTorch 2.9.1 /
CUDA 12.8 base image and runs the full tests in that environment when built.

## Not executed here

- Full Docker build or GHCR publishing. No Docker engine is available here.
- Full ComfyUI startup with external nodes or the real `/object_info` check.
- GPU inference on any of the 38 workflows.
- Perceptual likeness comparison using the user's original photographs.
- Browser/front-end interaction or visual inspection of generated photographs.
- New downloads or checksum verification of the multi-gigabyte model files.
  Their expected checksums and source pins are retained from the previous kit.

The actual Pod performs runtime node/schema checks before reporting readiness.
That check does not prove generation quality or GPU memory fit. The workflows
remain candidates to test, not a benchmark claim that native Qwen beats BFS.

See `validation_logs/` for the local unit-test and graph-validation output.

# Latest integration validation: Krea Identity 1.0.1

All 573 local Python tests passed, including 15 new tests for the CPU-only
SeedVR2 schema failure shown in the uploaded build log. The 23 static graphs,
2 frontend checks and all 3 repository preflights passed. No model or workflow
settings were changed. The full corrected Docker build and GPU inference have
not been run here. See `krea_identity/VALIDATION.md` for the validation boundary.

The historical H3 validation record follows.

---

# H3 Portrait 1.5 validation

This document records local validation for the release source package. It does
not claim a completed Docker/GPU deployment.

## Executed locally

The release was checked with:

```text
python -m unittest discover -s h3_portrait/tests -p 'test_*.py'
node tools/test_widget_serialization.mjs
python h3_portrait/verify_release.py --source h3_portrait
python scripts/check_workflows.py static --config config
python tools/check_repository.py --target portrait
python tools/check_repository.py --target portrait-lite
python tools/verify_snapshot.py
```

Executed result: **511 Python unit tests passed** across portrait, shared-LoRA, file-manager, general, and snapshot suites. The frontend widget script reports **2 serialization checks passed**. Static repository validation checked **19 workflow graphs**, both Dockerfiles, and all three GitHub Actions workflows.

Release-specific assertions include:

- pose/camera refs are text-only in recommended automatic video routing;
- scene remains visual in automatic video routing;
- pose/camera, expression and scene are text-only in still safe-swap routing;
- all visual refs can be restored explicitly with comparison mode;
- Ollama pass 1 still sees all uploaded references while native H3 receives only
  the routed visual subset;
- the still workflow uses `H3PortraitConditioning` -> native
  `MiniMaxH3ReferenceToVideo`, not a separate image generator;
- still target length is five frames;
- Native/Preview/~2MP experimental geometry is generated on a 32-pixel grid;
- High fidelity/Standard/Fast presets use 20/16/12 steps;
- best-stable frame selection and exact centered crop are deterministic;
- Lite asset catalog contains H3 assets only;
- Full installs two workflows and Lite installs three versioned workflows;
- ComfyUI seed companion is serialized as `fixed` so later widget values cannot
  shift into `NaN`/raw indexes;
- source/version/pipeline manifests reject mixed or older files.

The existing exported-video helper has separate regression coverage that decodes
the completed MP4 and compares its actual final decoded frame.

## Docker-build checks configured but not executed here

`h3_portrait/Dockerfile` also runs source verification, Python tests, the shared
LoRA downloader tests, the HTTP-fix self-test, and a real CPU ComfyUI schema/export
smoke before an image can publish.

## Not established locally

The following still require GitHub Actions and/or RunPod:

- completion of the Full/Lite Docker builds for this exact release;
- real GPU MiniMax H3 video generation;
- real GPU MiniMax H3 single-image five-frame generation;
- comparative visual identity/pose adherence;
- whether the experimental ~2MP preset improves a particular reference set;
- peak VRAM and render time for each preset/GPU;
- compatibility and visual effect of the user's private LoRAs.

No local validation requires reading the user's live `config/lora_links.txt`.

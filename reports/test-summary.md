# Comfy Workbench HQ 0.1.0: delivery and validation

## Publication status

This is a repository source candidate. No new remote GitHub repository was created, no Docker image was built or published, and no RunPod was launched in this environment. This is the historical validation report from the initial 0.1.0 package. In 0.1.1 the new-repository scripts were removed and the distribution was adapted for replacing files in the existing repository without a CLI. See repackage-validation.md for checks rerun on that distribution. No remote repository was modified here.

## Delivered

- Four model workspaces: Qwen Image 2.1 BF16, FLUX.2 klein 9B distilled BF16, MiniMax H3 Full BF16, and SeedVR2 7B FP16.
- 57 catalog entries: 51 retained tasks, four source-native/integration Standard HQ baselines, and two new standalone SAM utilities.
- 69 editable workflow graphs and 42 migrated source API exports.
- A migration disposition for all 82 original editable workflow files.
- A self-contained, offline HTML configurator with RunPod settings, private-secret references, and task selection.
- Runtime asset preparation, family-filtered private model libraries, an authenticated gateway, and a ComfyUI Workbench extension.
- KREA, Lite/INT8 configurations, BFS alternatives/comparisons, and torso-specific tasks removed from the active package.

The private LoRA text file from the source archive was not opened, extracted, copied, or included. Runtime configuration uses the existing secret identifiers hf_token and civit_token. A new access secret named comfy_password is required. Optional private library secrets are comfy_loras and comfy_checkpoints.

## Completed checks

| Check | Result |
|---|---|
| Python tests | 59 passed; two non-failing aiohttp test warnings |
| Configurator logic assertions | 111 passed |
| Static catalog/graph validation | 57 entries, 69 graph files, 82 source dispositions; no reported errors |
| Headless Chromium desktop interaction | Passed |
| Headless Chromium mobile layout | Passed; no horizontal overflow |
| Model selection, secret-reference generation, JSON download | Passed |
| Browser JavaScript errors | None observed |
| External browser network requests | Zero during test |
| Python compilation and shell syntax | Passed during preparation |

The browser test rendered the complete standalone HTML inline because local file navigation is restricted in the test environment. The delivered HTML is self-contained. Screenshots and machine-readable reports are included in this directory.

## Required release gates

The following have not been executed and must not be inferred from the passing offline tests:

1. Clean Docker builds and external dependency installation.
2. Pulling published container images onto RunPod.
3. Provider authentication, gated-model access, actual model downloads, and complete model-file integrity validation.
4. Importing custom nodes into the real, pinned ComfyUI installation and comparing live node schemas.
5. GPU inference, memory/VRAM measurement, visual quality, identity fidelity, audio quality, and video output tests.
6. End-to-end gateway and frontend-extension behavior against the built images.

H3's formerly Lite-only still-image task was migrated to the Full/BF16 configuration and remains GPU-unvalidated. HQ designates the specified model precision and compatible source settings, not a measured quality guarantee. The source's distilled FLUX schedule is retained. Standard HQ workflows are source-native/integration baselines, not claimed to be byte-identical official upstream template downloads.

Hardware and disk targets in the configurator are conservative planning estimates, not benchmarked minimums. H3's optional two FP16 prompt-helper models require approximately 134 GB of additional storage.

Run the included live-schema validator and representative workflows on each built workspace before treating the release as production-ready.

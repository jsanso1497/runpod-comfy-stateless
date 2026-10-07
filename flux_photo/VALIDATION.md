# Validation record: FLUX Photo 1.0

Prepared on 2026-10-06 from `New Compressed (zipped) Folder(4).zip`.

## Executed locally

**844 Python unit tests passed**, in separately executed suites:

| Suite | Tests | Result |
| --- | ---: | --- |
| New FLUX Photo module | 70 | Passed |
| Preserved general workflows | 230 | Passed |
| Preserved shared LoRA downloader | 69 | Passed |
| Preserved file browser/supervisor | 19 | Passed |
| Preserved H3 Portrait | 211 | Passed |
| Preserved Krea Identity | 153 | Passed |
| Preserved H3 Media | 50 | Passed |
| Preserved model safety | 20 | Passed |
| Source snapshot verifier tests | 22 | Passed |

Additional checks passed: two deterministic FLUX UI graphs and matching API graphs; node/link topology and reference-order checks; Python syntax; shell syntax; comparison JavaScript syntax; new GitHub Action YAML parsing; existing General, H3 Full and H3 Lite source preflights; final complete-snapshot verification.

Actual CPU tensor geometry tests used **7001 x 4003** and **4003 x 7001** source canvases with **2500 x 2500** masks. Native mode processed a **2768 x 2768** padded crop with no resampling, restored the exact canvas dimensions and passed row-by-row equality checks on every zero-mask pixel. These tests used synthetic replacement patches, not generated photographs.

Other tests exercised soft masks, mask holes, disconnected masks, edge-touching masks, thin brush marks, strict crop limits, empty/mismatched masks, inward feathering, optional local color matching, unchanged source tensors, preview-only resizing, model-loader selection, LoRA bypass, sampler schedule/mask wiring, simulated OOM, official-download validation and corruption/auth/disk-failure handling.

A saved PNG was reopened and its zero-mask RGB pixels compared with the source's original 8-bit RGB values. ICC, workflow and verification metadata were checked. Model and Hugging Face API interactions in unit tests are mocked; they are not downloads or model loads.

## Added to the new Docker build, but not executed here

The Dockerfile runs the FLUX tests, starts the actual pinned ComfyUI in CPU mode, retrieves `/object_info`, checks the `flux2` CLIP loader mode, and validates all workflow node schemas, required fields, dropdown values and output socket types before an image can be published by the supplied Action.

That real-core CPU smoke test is distinct from local mock tests. Its inclusion is not a claim that it has already passed.

## Not executed here

- Complete Docker build or GHCR publication.
- Real pinned ComfyUI backend/schema smoke test.
- RunPod startup/deployment or networking.
- Actual gated model downloads, full-size model SHA verification or private LoRA downloads.
- GPU inference, GPU peak memory, speed or compatibility benchmarks.
- Visual identity, color, texture or seam evaluation on generated photographs.
- Browser/UI testing of the interactive comparison widget or Mask Editor.
- RAW, 16-bit, wider-gamut or original-metadata roundtrip, which is not implemented.

No GPU-quality or identity-transfer score is claimed. Native large-crop support is implemented and its geometry is tested; inference quality at that size remains experimental and unmeasured. See `SOURCES.md` and the root setup guide for the published-resolution and color-format limits.

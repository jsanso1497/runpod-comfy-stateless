# Validation: one-person multiple-reference update 3.2.0

## Executed in this delivery

- 174 local unit/regression tests passed: the prior 123 tests, with version assertions updated, plus 51 new one-person tests.
- The 10 existing HTTP-policy tests passed without changing the HTTP patch.
- Python compilation, Bash syntax, JSON parsing, graph-link checks and catalog validation passed.
- The new checks exercise one required headshot plus up to eight body references, blank-slot handling, numeric ordering across skipped slots, input tensors and aspect-ratio preservation, duplicate-image detection, face/wardrobe prompt priorities, one-subject tags, individual Full Reference extraction requests, budget-error behavior, stable filenames, saved role validation, and execution gating.
- Transport and upstream node execution in the local tests are mocked. Tests use actual CPU tensors but no H3 VAE, diffusion weights, Ollama model, or GPU.
- All 10 model/LoRA catalog entries are unchanged from the corrected 3.1 package. The Krea graphs, SeedVR2 graphs, prior H3 graphs, custom-node pins and HTTP patch are byte-for-byte unchanged.

## Included in the GitHub build, NOT run in this environment

- `check_single_person_runtime.py` imports the ACTUAL pinned RefMod pack and this new helper in CPU mode. With a tiny synthetic VAE, it exercises the real creator and real version-5 bundle serialization/reload, checks separate image shapes and role metadata, and verifies the wrapper's V3 NodeOutput handling. This tests interfaces, NOT the learned VAE or likeness.
- The existing real ComfyUI CPU startup/object-info schema validation now also checks all four new workflows.
- The existing model/node dependency constraints, Rebalance tensor check, unit tests and HTTP checks remain in the Docker build.

## Not performed

No complete Docker build, live ComfyUI browser interaction, real model download, live Ollama inference, H3 GPU generation, peak-memory measurement, or comparative face/body fidelity test was performed here. A Docker daemon and external network access from the working container were unavailable.

One file contains separate full-reference VAE encodings. It is not identity training or lossless photo storage. The headshot/body hierarchy is expressed in prompts and saved metadata; it is not an independently verified facial weighting mechanism. More reference images and more tokens do not guarantee better identity retention.

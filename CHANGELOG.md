# 1.5.3: verified startup and model-family safeguards

See `H3_1.5.3_START_HERE.md` and `VALIDATION_1_5_3.md`. Adds explicit readiness, resumable progress-reporting pulls, confirmed Reference Pack model unloading, protected native loaders, a conservative saved-workflow repair tool, and a non-dynamic VRAM default. Full model precision and generation settings remain unchanged.

# H3 Portrait 1.5.2: local Hearmeman Reference Video

- Add `H3_Reference_Video_Swap_Local` to Full and Lite builds from the attached 1.5.0 source baseline.
- Install Hearmeman24 `ComfyUI-MiniMaxRefPack` 0.3.5 at pinned commit `7012734eabf6f98063d6eaf8ce1f9264ee803664`.
- Patch the baked Reference Pack to local-only prompt writing through the existing Ollama service. OpenRouter model discovery and hosted execution are disabled.
- Full uses the existing 32B Instruct local analysis model; Lite uses the existing 8B Instruct model.
- Preserve native H3 video, image, soundtrack and standalone audio reference sockets.
- Add Draft-only prompt review, Full 25-step High fidelity mode, exact 9:16 / 2:3 / 16:9 delivery geometry, and existing final-MP4 last-frame export.
- Preserve Krea Identity 1.2.0, H3 Media 1.2.0, existing H3 workflows, weights and private shared LoRA configuration.

# Krea Identity + H3 Media 1.2.0

- Add Krea workflow 08: two separate references, text-described scene, exact selectable image aspect and existing rebalancer.
- Add native full BF16 H3 HQ media workflow: MP4 voice-only, selected-frame identity, motion or combined roles; new scripted dialogue and generated audio export.
- Add bounded local FFmpeg extraction, independent reference numbering and optional standalone audio inputs.
- Bundle native adapter in all three image recipes; model downloads opt in with ENABLE_H3_MEDIA=1.
- Extend real CPU ComfyUI registry build gate without bypassing the SeedVR2 schema fix.
- Make Krea graph-generator schema fixtures self-contained for isolated Docker module tests.
- Preserve prior workflow JSONs, model weights/catalogs and private LoRA list.

# Krea Identity 1.1.0 - 2026-10-06

- Added chainable original-image reference lists with labels and free-text purposes, up to six per person.
- Added directed replacement fields for the exact scene target, replacement description and wardrobe behavior.
- Added one-pass man/woman replacement with separate A/B reference groups and an explicit binding map.
- Added optional two-pass protected-region replacement, painted-mask inputs, crop context, inward feathering and second-pass protection of all first-pass pixels.
- Added scene-aspect matching without discarding original edges and exact original dimensions for protected composites.
- Retained the four existing Krea graphs, all model filenames/precision/hashes/upstream pins, H3 inference graphs, screenshot rebalancer and SeedVR2 CPU build fix.
- Added 77 CPU tests, for 650 total; GPU image quality and the new full Docker build remain untested.

---

# Krea Identity 1.0.1: CPU build-schema hotfix (2026-10-06)

- Fix the failure in logs_101498979872.zip: SeedVR2's loader schemas indexed
  devices[0] when the CPU-only Docker builder reported no GPU devices.
- Add a guarded, idempotent fallback in BOTH the DiT and VAE loader schemas.
  Nonempty CUDA/MPS lists, model execution code, precision and settings are unchanged.
- Apply the fix to the bundled copy before the real ComfyUI registry smoke test.
  Keep validation of all four Krea workflows; do not skip or fabricate SeedVR2 nodes.
- Add 15 regression tests: original failure, CPU fallback, unchanged GPU choices,
  repeat installation, source-drift errors and correct installer sequencing.
- Rerun all 573 local Python tests, 23 static graphs and 2 frontend checks.
  Corrected full Docker builds and GPU inference remain untested locally.

# Krea Identity 1.0 additive integration (2026-10-06)

- Added the same Turbo BF16 still-image capability to general, H3 Full and H3 Lite builds.
- Three original-photo inputs via a face/body reference sheet, plus a direct two-image alternative.
- Existing Identity Edit v1.2 and exact ConditioningKrea2Rebalance node, with true A/B bypass.
- Optional manually cropped face refinement and separate SeedVR2 single-image upscale.
- Opt-in runtime downloads; pinned source and hash-verified weights; no new inference API.
- Retained all existing workflows and private shared LoRA configuration.
- 47 new local tests; actual Docker/GPU/visual testing still pending.

# H3 Portrait 1.5 - role-routed video + MiniMax H3 Ref2VA single-image workflow

- Fix automatic H3 reference dominance by allowing pose/camera and expression guides to be analyzed by Ollama but withheld from native H3 visual conditioning in the recommended video mode.
- Add a stricter Lite still safe-swap mode: pose/camera, expression and scene guides are text-only while subject appearance refs remain native H3 pictures.
- Add `H3_Portrait_Image_Lite_v1_5`, which uses MiniMax H3 Ref2VA itself for single-image generation. H3 renders its five-frame minimum packet and the workflow saves one selected frame.
- Reuse the existing Lite H3 INT8 model stack; remove the abandoned separate still-image model path and its extra asset downloads.
- Add Native (~1 MP), Preview (~0.5 MP) and experimental H3-native ~2 MP still canvases plus High fidelity/Standard/Fast quality presets.
- Add best-stable/middle/first/last decoded-frame selection for H3 still delivery.
- Correct ComfyUI workflow widget serialization by storing the seed companion control explicitly, preventing shifted enum/numeric values and NaN defaults.
- Save the exact final decoded MP4 frame from H3 video exports as a matching PNG for future chaining.
- Add versioned Full/Lite workflows and regression coverage for role routing, still geometry, still frame selection, widget values and H3-only Lite assets.

# File-manager snapshot: h3-clean-filemanager-2026-10-05-r2

- Complete source package, including both build workflows and all 17 generation workflows.
- Password-protected JupyterLab file manager on HTTP 8888 in BOTH images.
- Browser and ComfyUI use isolated Python environments; no GPU dependency changes.
- Real password/CSRF/file-operation smoke test added to both Docker builds.
- File manager starts before model downloads; no open-access fallback.
- Browser-only upload instructions replace desktop/local-terminal setup directions.
- Finder metadata excluded from completeness checks. LoRA link contents remain editable.
- Portrait pipeline/version/models/quality/workflow JSONs unchanged from 1.3.0.

# Clean reset snapshot: h3-clean-2026-10-05-r1

- Complete source replacement; application code remains H3 Portrait 1.3.0.
- Added exact source file-set/hash checks before Docker, with LoRA links editable.
- Added 21 regression tests for upload completeness and mixed source versions.
- Added the `h3-portrait-clean` tag alongside existing portrait tags.
- Replaced partial web-upload instructions with a single GitHub Desktop commit.
- No model, LoRA, rendering, or prompt logic changed from the consolidated 1.3.0 source.

# 2026-10-05 complete source consolidation

This supersedes the earlier source ZIPs and live hotfix scripts. It contains both
image recipes, the general 3.2 workflows and H3 Portrait 1.3.

## H3 Portrait 1.3

- Replaces single-pass/Thinking-only prompt analysis with an explicit, separate
  32B Instruct model for compact vision mapping and 32B Thinking model for the
  text-only MiniMax director. Both are abliterated Q4 variants.
- Corrects the earlier unsupported think=false assumption for Qwen3-VL Thinking.
- Downloads both helpers automatically and loads them sequentially.
- Uses bounded, fresh Instruct regeneration for malformed/truncated answers.
- Keeps every reference and the original brief; no rigid user input grammar.
- Keeps current H3 weights, exact portrait output handling, LoRA selection and
  Preview/Standard/High fidelity presets unchanged.
- Checks VERSION, settings, source files and installed copies before model
  downloads. Automatically records the source Git commit and image node hashes.
- Preserves the existing uploader and node widget contract.

## Repository/build consolidation

- Preserves the runner disk cleanup, diagnostic reserve and live low-space guard.
- Preserves no-external-cache builds and source-commit-specific tags.
- Includes corrected Rebalance import checks, general test selection and the
  complete shared-LoRA disk-fixture test file.
- Adds early source-layout checks to both build workflows.
- Retains Krea-only Rebalance, SeedVR2, optional RefMod workflows and shared LoRA
  URL downloads for the separate general image.
- Does not install Jupyter or a file-browser service.

The only user-specific catalog missing is the unseen live config/lora_links.txt.
Copy that existing file into this package before uploading. Unknown independent
live-repository edits cannot be reconciled without their source files.

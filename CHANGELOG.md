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

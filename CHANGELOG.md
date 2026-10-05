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

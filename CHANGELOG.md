# 0.1.1 - Existing repository / Mac no-CLI distribution

- Retargeted the configurator to the existing runpod-comfy-stateless repository.
- Removed the scripts that created a new private repository and their CI dependency.
- Added GitHub Desktop and browser-only installation instructions.
- Added checks for required dotfiles and Actions workflows.
- Preserved all 69 editable workflows, 42 API exports, model definitions, node code, and runtime behavior from 0.1.0.
- Supplied a companion browser-upload kit with three sub-100-file batches and visible copies of all six dotfiles.
- No repository push, image build, or GPU test was performed by repackaging.

# 0.1.0 - HQ workbench source candidate

- Migrated the retained source workflows into task-based folders with stable decision IDs.
- Removed KREA, Lite/INT8 model variants, BFS workflows/comparisons and torso-specific implementations.
- Ported H3 still-image gating and model configuration to Full/BF16; GPU validation remains required.
- Added four fixed Standard HQ baselines and independent manual/SAM masking/compositing examples.
- Added the offline model/task-to-RunPod configurator, optional Pages deployment, and authenticated in-Pod task/file browser.
- Added runtime-only private libraries, explicit model-family loaders, conservative full-precision validation, and exact secret-name mapping.
- Replaced historical private-data-bearing base images with source-pinned clean builds and separate data directories.
- Preserved available source API exports and source workflow migration records.
- Added static, CPU contract, mask preservation, browser and configuration tests. No Docker/GPU result is implied by those tests.

# RunPod ComfyUI: complete source with browser file manager

Start with **START_HERE.md**. It contains browser-only GitHub upload instructions
and the complete H3 Portrait RunPod setup. No GitHub Desktop or local Terminal.

This repository includes BOTH independent GitHub Actions/Docker builds:
- H3 Portrait -> h3-portrait-clean / h3-portrait
- General ComfyUI -> latest

It includes all 17 ComfyUI workflow JSONs, shared link-only LoRA downloads,
H3 Portrait 1.3 split-model prompting, and password-protected JupyterLab on 8888.
Only the H3 Portrait build is required for the user's current portrait workflow.

**HTTP 8188 = ComfyUI. HTTP 8888 = password-protected file manager.**
Ollama stays on internal loopback 11434 and is not exposed.

Add personal Civitai/Hugging Face URLs only to config/lora_links.txt. Each line is
one link, not JSON or manually maintained metadata. Keep actual tokens/passwords
in RunPod Secrets. Downloading every link makes LoRAs available; only selected,
compatible LoRAs are applied in a workflow.

The source snapshot detects incomplete/mixed browser uploads. The LoRA text is
editable without changing it. Finder metadata and Git/Python caches are ignored.
Do not combine this repository with older ZIPs. The Dockerfiles depend on the
existing pinned GHCR base images: do not delete that container package.

See VALIDATION.md for executed checks and untested Docker/GPU boundaries. This
archive contains source/configuration, not model weights or a prebuilt image.

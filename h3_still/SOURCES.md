# Primary sources and provenance

Checked for this update on November 5, 2026. These are source references, not a claim that GPU inference has been tested.

## Developer implementation and presets

- Fizgig H3 Still: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still
- Exact source: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/10d5171f6e45b0757a03df977f53a660e060c33d
- README: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/blob/10d5171f6e45b0757a03df977f53a660e060c33d/README.md
- Three upstream graphs: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/10d5171f6e45b0757a03df977f53a660e060c33d/example_workflows

The developer describes a single-frame latent paired with a special decoder that repeats the latent into a five-frame group before selecting a decoded frame. The same video VAE is used. The native H3 node provides conditioning but not the sampled latent. The documented high-quality example is 8 MP, 50 steps, er_sde, without Turbo. The smaller example uses 20 steps and v4 step600 EMA at strength 0.38. Keep these developer claims distinct from independently measured performance.

The README recommends 2.5 MP resizing for editing, whereas the saved edit JSON has 2 MP. Both portable variants are included. The saved high-resolution selector uses a multiple of 16 despite the README's stated multiple-of-32 rule; the portable version uses 32. Originals remain unmodified for inspection.

## ComfyUI and the resize dependency

- ComfyUI v0.43.0, exact commit: https://github.com/Comfy-Org/ComfyUI/tree/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
- Required frontend: comfyui-frontend-package 1.57.4, from that commit's requirements.txt.
- RMBG exact source: https://github.com/1038lab/ComfyUI-RMBG/tree/a66a5e641a54988d80ed6aad18946081c85f309e
- Unmodified resize class: https://github.com/1038lab/ComfyUI-RMBG/blob/a66a5e641a54988d80ed6aad18946081c85f309e/py/AILab_ImageMaskTools.py
- Resize helper module: https://github.com/1038lab/ComfyUI-RMBG/blob/a66a5e641a54988d80ed6aad18946081c85f309e/py/AILab_utils.py

Only the official resize class is registered. The complete RMBG repository and its license remain at /opt/vendor/ComfyUI-RMBG in the image. No background-removal model is necessary for that resize operation.

## Model sources

- Official H3 model card: https://huggingface.co/Comfy-Org/MiniMax-H3
- Pinned H3 revision: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707
- Diffusion file: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/54472221d27c695fca9621b0c3ed52f06c4af707/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
- Encoder file: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/54472221d27c695fca9621b0c3ed52f06c4af707/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- Video VAE: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/54472221d27c695fca9621b0c3ed52f06c4af707/vae/minimax_h3_video_vae_int8_convrot.safetensors
- Larry v4 step600 EMA LoRA: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/43a74557ac3f6539db8e0f2a959d03feb7a81480/minimax_h3_turbo_v4_step600_ema.safetensors

Displayed file sizes were approximately 21 GB, 15.7 GB, 2.81 GB and 0.78 GB. Plan for about 40 GB of weights plus the application, downloads, outputs and free-space headroom. SHA-256 hashes are locked in models.lock.json. The no-Turbo graph keeps a zero-strength LoRA loader, so the shared four-file cache includes the LoRA even when that graph is selected.

The official H3 model card recommends CUDA 13.0 PyTorch for the INT8/ConvRot variant and says the Qwen3-VL NVFP4 encoder does not require a Blackwell GPU. This is the reason for the CUDA wheel choice, not a claim that all GPU hosts or every resolution will work.

## PyTorch

- Official previous-version installation commands: https://pytorch.org/get-started/previous-versions/
- PyTorch 2.9.1, torchvision 0.24.1 and torchaudio 2.9.1 are installed from the official cu130 wheel index.

## Integration base

Target repository inspected at commit b0162d440e7da28fa8f563e8beec5cb955bee095:
https://github.com/jsanso1497/runpod-comfy-stateless/tree/b0162d440e7da28fa8f563e8beec5cb955bee095

This addition follows the repository's existing isolated-workspace pattern. It does not patch the legacy catalogs, broad base Dockerfile or other workspaces.

## License handling

The build retains upstream source and copies the available Fizgig, RMBG and ComfyUI license files into /opt/h3-still/licenses and the build artifact. Fizgig code uses MIT; RMBG's image tools identify GPL-3.0. Consult each source license and the separate model/LoRA licenses. Existing copyright notices are not removed. The registration shim does not replace the original resize implementation.

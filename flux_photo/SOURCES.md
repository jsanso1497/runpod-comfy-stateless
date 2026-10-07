# Upstream sources and implementation boundaries

Sources checked during preparation on 2026-10-06. This file documents decisions; it does not claim a GPU run.

- BFL model card, 9B distilled, Qwen3 8B, BF16 example, four steps, guidance 1, gated non-commercial weights:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/README.md
- Official BFL model repository and access agreement:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- Official ComfyUI guide, including 9B distilled image-edit workflows and matching encoder/VAE families. Its FP8 quick-start choices are NOT substituted for the non-quantized files selected here:
  https://docs.comfy.org/tutorials/flux/flux-2-klein
- Official BFL FLUX.2 overview and published up-to-4MP context. Native 2768 square is beyond that range and is marked experimental:
  https://docs.bfl.ai/flux_2/flux2_overview
- Official Comfy encoder/VAE package, immutable filenames/revisions/hashes recorded in config/models.json. The spelling "encorder" is the actual repository name:
  https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b
- Pinned Comfy native FLUX.2 scheduler and 128-channel latent definition:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_flux.py
- Pinned native sampler API with noise-mask support:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy/sample.py
- Pinned native model loaders, CLIPLoader type flux2, model-only LoRA loader, tiled VAE encoder/decoder:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/nodes.py
- RunPod custom-template documentation:
  https://docs.runpod.io/pods/templates/overview

The crop, protected stitch, optional-LoRA wrapper, download guard, reference-conditioning wrapper, comparison widget and export nodes in this module are new local code. They are not an official BFL inpainting product or a claim that BFL certified this crop size. The workflow implements masked denoising with a reference-capable editor, not a separately trained FLUX Fill model.

The Docker foundation digest and existing shared LoRA/file-browser components are retained from the user-supplied repository. Old H3/Krea settings are not imported into the new inference graph.

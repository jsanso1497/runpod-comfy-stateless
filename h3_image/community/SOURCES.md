# Sources and provenance

## User-supplied evidence

1. Reddit conversation screenshot: 16 MP, SamplerLCM, s_noise 0.95-1.20, optional Pruna VAE testing. Mixed reports about whether the VAE is better. No exact graph or complete settings.
2. Discord recommendation screenshot: 16 MP, er_sde, simple, 4-6 steps.
3. Discord example comparison screenshot: described as 6, 12 and 50 steps; commenter says 50 is too high. Not a controlled reproduction record.
4. `minimax_h3_image_t2i_00066_.jpeg`: delivered at 1536 x 657 without recoverable workflow/prompt metadata. See source_image_metadata.json. Original image files and screenshots are deliberately NOT copied into the public repository update.
5. Supplied MiniMax-H3-Image README and pdd_config.json: 16 MP quality target, six-step official configuration.

## Verified implementation sources

- User repository reviewed read-only at `8323001d15aae2f41e7446d25b1846f998c3bf5b`.
- Existing workspace ComfyUI pin remains `ede4e5fe8d15445c6fcd6828d70b202c22f9502c`.
- BasicScheduler and KSamplerSelect:
  https://github.com/Comfy-Org/ComfyUI/blob/ede4e5fe8d15445c6fcd6828d70b202c22f9502c/comfy_extras/nodes_custom_sampler.py
- SamplerLCM including s_noise, s_noise_end and noise_clip_std:
  https://github.com/Comfy-Org/ComfyUI/blob/ede4e5fe8d15445c6fcd6828d70b202c22f9502c/comfy_extras/nodes_advanced_samplers.py
- ImageScaleToMaxDimension and ImageStitch:
  https://github.com/Comfy-Org/ComfyUI/blob/ede4e5fe8d15445c6fcd6828d70b202c22f9502c/comfy_extras/nodes_images.py
- Native H3 Image model:
  https://huggingface.co/MiniMaxAI/MiniMax-H3-Image

## Optional Pruna decoder

The screenshot did not specify a version or precision. This package intentionally chooses the smaller published INT8 VAE for a controlled decoder-only test, not as a claim of exact recipe recovery or official cross-model compatibility.

- Model card: https://huggingface.co/PrunaAI/P-H3-Image-Comfy
- File: https://huggingface.co/PrunaAI/P-H3-Image-Comfy/blob/main/vae/p-h3-image_vae_int8_convrot.safetensors
- Published file commit: https://huggingface.co/PrunaAI/P-H3-Image-Comfy/commit/83e4891dd80a33e639e6e75b466fdccecbcad4a9
- Size: 3146996440 bytes
- SHA256: 583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe
- Pinned download revision: 83e4891dd80a33e639e6e75b466fdccecbcad4a9

The native base model and encoder pins are not changed. Model licensing and authorization requirements are unchanged. This update ships no model weights, screenshots, sample photographs, credentials, font files or third-party workflow claims.

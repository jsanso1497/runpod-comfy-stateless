# 1.5.3 verification sources

## Supplied materials

- `H3_Portrait_1.5.2_Hearmeman_Local_Full_Source.zip`, used as the code baseline.
- The supplied Node 7 model-not-found report and successful later Ollama request.
- The supplied H3 VAE `VRAM grow failed` trace.
- The supplied KSampler workflow JSON showing H3's 32B encoder, `type=lumina2`,
  and H3's audio VAE selected alongside KREA's diffusion model.
- The supplied H3 screenshot showing Qwen image VAE selected in the H3 audio slot.

## Primary upstream references checked

- Pinned ComfyUI CLI memory flags:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy/cli_args.py
- Pinned ComfyUI encoder dispatch:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy/sd.py
- Pinned Reference Pack class and output contract:
  https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack/blob/7012734eabf6f98063d6eaf8ce1f9264ee803664/minimax_refpack/nodes.py
- Ollama running-model endpoint:
  https://docs.ollama.com/api/ps
- Ollama unload, model-cache directory and context settings:
  https://docs.ollama.com/faq

The source proves the reported model-family mismatch and the absence of a verified
unload in the previous Reference Pack path. It does not prove that Ollama residency
was the sole cause of the observed GPU allocation failure. That is why this update
adds both a confirmed handoff and a configurable non-dynamic memory path, with
no claim of a measured VRAM or render-quality improvement.

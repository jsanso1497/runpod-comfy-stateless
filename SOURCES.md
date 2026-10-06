# H3 Portrait 1.5 sources

Current portrait implementation sources and release-specific interpretation are
in `h3_portrait/SOURCES.md`.

Key references:

- MiniMax H3 ComfyUI guide:
  https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- Pinned native H3 Ref2VA implementation:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py
- MiniMax reference-prompt guide:
  https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- MiniMax H3 model package:
  https://huggingface.co/Comfy-Org/MiniMax-H3
- H3 still-image community reference used only for corroboration, not installed:
  https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio
- Ollama chat / structured outputs:
  https://docs.ollama.com/api/chat
  https://docs.ollama.com/capabilities/structured-outputs
- RunPod templates:
  https://docs.runpod.io/pods/templates/create-custom-template

The separate general `:latest` image retains its own historical Krea/SeedVR2 and
other provenance files. H3 Portrait 1.5 does not change those general workflows.

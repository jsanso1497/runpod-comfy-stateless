# H3 Portrait 1.5 implementation basis

Pinned ComfyUI core reference:

`65787d668397d230bf5839d69a0a7239e2dad378`

Primary sources:

- MiniMax H3 native workflows and guidance:
  https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- Pinned native H3 reference implementation (`MiniMaxH3ReferenceToVideo`):
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py
- MiniMax full-reference prompt-writing guide:
  https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- H3 weights used by the Full/Lite profiles:
  https://huggingface.co/Comfy-Org/MiniMax-H3
- Community evidence that H3 can be used as a still-image renderer by producing a short/minimum frame packet and selecting one frame:
  https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio
- Ollama structured chat/output:
  https://docs.ollama.com/api/chat
  https://docs.ollama.com/capabilities/structured-outputs
- Full helper models:
  https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M
  https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
- Qwen3-VL separate Instruct/Thinking model behavior:
  https://github.com/ollama/ollama/issues/16945
- RunPod custom templates:
  https://docs.runpod.io/pods/templates/create-custom-template

## Release-specific interpretation

The pinned H3 Ref2VA node accepts image references as an ordered collection and,
with a video VAE connected, converts each supplied visual reference into native
H3 reference conditioning. It does not provide a per-reference "identity only" or
"pose only" strength. Release 1.5 therefore filters role-only guide pixels before
native H3 conditioning and carries their analyzed pose/camera/scene information in
text instead.

The Lite still workflow deliberately reuses those same H3 Ref2VA weights. It does
not install the community Image Studio plugin and does not copy its implementation;
that project was used only as corroborating evidence for the practical short-frame
still-output pattern. This release uses the pinned native ComfyUI H3 path plus its
own small frame-selection node.

Model/LoRA licenses are not granted by these scripts. Actual visual fidelity,
VRAM, speed, and private LoRA compatibility require live GPU testing.


## H3 Portrait 1.5.2 Reference Pack

- Hearmeman24/ComfyUI-MiniMaxRefPack 0.3.5
- Pinned commit: `7012734eabf6f98063d6eaf8ce1f9264ee803664`
- Repository: `https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack`
- The Docker build patches the pinned source to local-only prompt routing before installing it. No hosted inference API is added.

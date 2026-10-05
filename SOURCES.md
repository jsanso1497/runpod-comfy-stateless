# Implementation basis and verification

Integration base: the successful image recorded in the user's supplied
logs_100917806949.zip, build/8_Build and push image - no external build cache.txt:
sha256:49d305115a3cfe5763d5c9694eb20429803cf3e1a209567506dba15de594f672.
The user's private repository returned 404 through the GitHub connector; no
unseen live repository files have been represented as inspected or modified.
Existing code/model metadata was taken from the supplied multi-reference package.

Primary sources consulted:

- H3 native reference limits, role tagging and native reference conditioning:
  https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
- H3 native resolution, schedules and full/pruned LoRA compatibility:
  https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- MiniMax full-reference prompt-writing guide (basis, not an unmodified copy):
  https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Exact pinned native H3 reference node:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py
- Exact pinned native sampler calls:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_custom_sampler.py
- Exact pinned LoRA mapping/patch implementation:
  https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy/lora.py
- Ollama chat, images, structured output and keep_alive:
  https://docs.ollama.com/api/chat
  https://docs.ollama.com/capabilities/structured-outputs
- Selected reference-analysis model (separate Instruct edition, approximately 21 GB):
  https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M
- Selected text director model (Thinking edition, approximately 21 GB):
  https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
- Ollama maintainers: Qwen3-VL is not a hybrid thinking-toggle model; use Instruct
  for non-thinking operation. Checked on 2026-10-05:
  https://github.com/ollama/ollama/issues/16945#issuecomment-4825564941
  https://github.com/ollama/ollama/issues/16945#issuecomment-4834882315
- Separate original model editions:
  https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking
  https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- Civitai model-version file metadata, hashes and downloadUrl:
  https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md
- RunPod custom templates:
  https://docs.runpod.io/pods/templates/create-custom-template

Model and LoRA licenses are not granted by these scripts. The optional user LoRA
has not been identified, downloaded or independently checked. Native model output
quality and exact peak memory remain empirical deployment questions.

## What is new in this consolidation

Source/version manifests, two distinct sequential prompt models, compact analysis
bounds, text-only director input, bounded Instruct regeneration, and early
source/copy verification are implementation choices in this release. They were
unit-tested locally. They are not claims from model authors about guaranteed
fidelity, inference time, or memory use. See VALIDATION.md.

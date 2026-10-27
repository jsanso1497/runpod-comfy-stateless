# Reviewed sources and evidence boundary

Review date: October 22, 2026. Repository delivery is additive; no remote writes.

## Akatz and runtime

- https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
- https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/examples/README.md
- https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/example_workflows/toolkit/README.md
- https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/example_workflows/toolkit/H3-Character-Swap.api.json
- https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/requirements.txt
- https://hub.docker.com/layers/pytorch/pytorch/2.11.0-cuda12.8-cudnn9-runtime/images/sha256-eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

Published character-adapter hash:
8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f

## BFS comparison

- https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/character-swap.md
- https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/minimax_h3_character_swap_workflow_native_nodes_v1.1.json
- https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors
- https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/LICENSE_CHARACTER_SWAP.md
- https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors

Published v1.1 adapter hash:
ab21167bed71db99ac09527ad1f26f5f3517687ebf2e1f1bd5e91cc0bdf9b9c3

The inspected first-pass native graph uses Euler, simple, eight steps, 12/3 shifts
and match-sized references. The guide video is connected at frame zero. The local
comparison omits the optional upscale/RMBG/third-party routing branches and uses
an explicit reviewed description instead of the graph's VLM branch.

## Local prompt assistance and hardware

- https://ollama.com/library/qwen3-vl:8b
- https://docs.ollama.com/api/chat
- https://github.com/ollama/ollama/releases/tag/v0.35.1
- https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/nvfp4.md
- https://docs.runpod.io/pods/templates/create-custom-template

The new prompt helper is custom integration code. It is not MiniMax's hosted
Context-IR, not the third-party Prompt Writer extension, and not a claim of
frame-complete video understanding. It samples eight frames from one selected
window and returns a constrained, reviewable draft.

Native Akatz source is pinned before use. Model metadata is resolved to full HF
commit hashes and SHA256 at build time; runtime downloads cannot silently follow
main. The two character-adapter expected hashes above are additionally enforced.
The Ollama model registry tag is recorded by content digest on the Pod, but is not
claimed to be immutable in the source package. Prompt caches include that digest.

No controlled ranking, verified identity lock, frame-exact replacement or GPU
certification is claimed by this package.


October 26 continuity research, source links and integration-vs-upstream distinctions
are recorded in CONTINUITY_RESEARCH.md. Models and pins remain unchanged.

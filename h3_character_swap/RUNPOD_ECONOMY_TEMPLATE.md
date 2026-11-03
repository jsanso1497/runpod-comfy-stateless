# RunPod: H3 BF16 Economy, no prompt LLM

| Field | Setting |
|---|---|
| Name | H3 BF16 Economy - A100 80GB |
| Deployment | Secure Cloud GPU Pod, Linux amd64 |
| GPU | One A100 SXM 80 GB |
| Host offer | 125 GB system RAM, 16 vCPUs currently listed |
| Published GPU rate | $1.79/hour, checked November 2, 2026; storage/taxes additional |
| Container image | Exact digest from successful **Build H3 BF16 economy** action |
| Tag after publication | ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap-economy |
| Container disk | 80 GB |
| Network volume | 300 GB clean dedicated setup; retain larger volume for old models/outputs |
| Volume mount | /workspace |
| HTTP ports | 8188 only |
| TCP ports | Blank |
| Container start command | Blank |
| Entrypoint override | Blank |
| Browser login | workbench / comfy_password secret |

The GPU offer is a planning target. Actual BF16 memory, speed and image quality
were not benchmarked here. Do not select A100 40 GB by mistake. No pooled multi-GPU
memory configuration is provided. Normal model offloading stays enabled.

Use all rows in RUNPOD_ECONOMY_ENVIRONMENT.txt. Remove H3_SWAP_PROMPT_MODEL and BFS
variables. Economy enforces no optional helper and metadata enabled. No cloud LLM
key or 262K Ollama context setting is needed. The mandatory H3 Qwen BF16 encoder is
NOT the removed Ollama prompt helper.

Open HTTP 8188. Add /files/ for the built-in H3 Files page. It does not depend on
model-ready status. Do NOT use the old RunPod File Browser 8080 card. Keep only 8188
exposed. ComfyUI itself waits until /_swap/status reports ready:true and
model_profile:bf16_economy. Folder: H3_Character_Replacement_2026-11-02.2_ECONOMY.

300 GB is not a promise that your existing library fits; use the current volume
if its model files and outputs already need more. Network-volume data must be
secured before deleting any older Pod. Never manually delete models just to make
the first download proceed without checking which workflows still use them.

Prices source: https://www.runpod.io/pricing
See ECONOMY.md for the native test, automatic batches and right-click recovery.

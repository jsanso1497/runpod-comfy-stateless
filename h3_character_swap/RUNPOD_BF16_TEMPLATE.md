# RunPod: H3 BF16 high precision

| Field | Value |
| --- | --- |
| Template name | H3 Akatz BF16 - Native + Low-memory Chain |
| Deployment | GPU Pod, Linux x86_64 / amd64, Secure Cloud |
| Image | New digest from **Build H3 BF16 high precision** |
| Image tag, only after successful publication | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap-bf16` |
| GPU | One NVIDIA B200, 180 GB VRAM |
| GPU count | 1 |
| Host RAM | At least 256 GB recommended; listed B200 allocation is 283 GB |
| CPU | At least 24 vCPUs recommended; listed B200 allocation is 28 |
| Container disk | 80 GB |
| Network volume | 1,000 GB recommended when retaining your previous library and outputs; 500 GB for a fresh setup |
| Volume mount | `/workspace` |
| HTTP ports | `8188` only |
| TCP ports | Leave blank |
| Container start command | Leave blank |
| Entrypoint override | Leave blank |
| Registry credentials | None if GHCR package is public; otherwise your saved registry credentials |
| Jupyter / SSH | Not configured by this image; do not expose 8888 / 22 |
| Login | `workbench`, using `comfy_password` |
| Environment | Copy each name/value from RUNPOD_BF16_ENVIRONMENT.txt |

Capacity values are recommended starting allocations, NOT measured H3 minima or a
runtime guarantee. System RAM/vCPUs come with the host offering, not necessarily
independent template controls. If the network volume already provides /workspace,
do not buy an additional same-size Pod volume. Attach persistent storage in a data
center where B200 is available. Never terminate an old Pod before securing its data.

The image includes PyTorch 2.11.0 / CUDA 12.8, ComfyUI and the selected custom nodes.
Do not enter the plain PyTorch image as your Pod image; it would omit this setup.
Do not request multiple GPUs expecting their memory to combine automatically.
H200 141 GB is a lower-headroom comparison with normal offloading, not the default.
B300's newer architecture is outside this package's compatibility qualification.

The BF16 profile is compiled into this separate image. There is no RunPod model-
profile variable that turns the old quantized image into this one. At startup,
`/_swap/status` must show `model_profile: bf16_full`.

Keep `H3_SWAP_VARIANTS=akatz`. Remove BFS flags and unrelated WB_TASKS/WB_WORKSPACE
settings. No 262144-token Ollama context. Optional analysis uses 32768 context and
16384 output allowance by default. The simple native/chain workflows need no
analysis request. H3_SWAP_PROMPT_ASSIST=0 skips the analysis service and its large
model download but does not disable the video workflows.

Use /files/ on HTTP 8188, not the old direct-8080 FileBrowser. Use /_swap/chain to
review chunks, /_swap/recovery for saved workflows, /_swap/status for preparation.

See BF16_HIGH_PRECISION.md for first tests, precision verification and limitations.

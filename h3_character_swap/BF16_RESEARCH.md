# Source verification and design decisions, November 1, 2026

## Verified source files

Comfy-Org publishes the original unquantized full Ref2VA and BF16 H3 encoder:

- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors
- https://huggingface.co/Comfy-Org/MiniMax-H3/raw/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_ref2va_bf16.safetensors
- https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
- https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors

Published SHA256s and exact pointer sizes are recorded in precision.py and this
guide. Both are requested from the existing immutable Comfy-Org revision during
the build, with expected-hash refusal on any mismatch. The browser could not fetch
the encoder's revision-specific raw URL; its public current pointer and file page
agree on SHA256/size. Exact revision/file resolution remains a build-time gate.
No full model checkpoint was downloaded in the authoring environment.

- Ref2VA full BF16: 66,280,487,368 bytes, SHA256 e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026.
- H3 encoder BF16: 51,506,295,256 bytes, SHA256 600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d.

## Why retain the adapters and eight-step recipe

https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/examples/README.md

The author recommends V1.5, the separate FL2V Turbo8 adapter, both strengths 1.0,
CFG 1, er_sde, normal, eight steps. The published qualification uses the INT8 base.
A BF16/full base plus BF16 encoder is an experimental comparison, not a newly
published author-qualified recipe or proof of better character identity. More
steps, removing Turbo or swapping models unrelated to character replacement are
not assumed quality improvements.

## Attention and inference flags

https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy_extras/nodes_model_advanced.py
https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy/cli_args.py

The source maps 'comfy kitchen attention' to 'comfy_kitchen_int8'. Merely changing
checkpoint filenames would therefore leave another low-bit path. The BF16 image
uses the available PyTorch attention backend and explicit BF16 diffusion/text
storage flags, leaving VAE handling and normal offloading unchanged. This does
not claim every arithmetic operation is BF16 or that numerical results are equal
across CUDA kernels.

## Hardware and storage

https://www.runpod.io/pricing
https://www.runpod.io/gpu-models
https://docs.nvidia.com/cutlass/4.4.2/overview.html
https://pytorch.org/blog/pytorch-2-7/
https://docs.runpod.io/pods/storage/types
https://docs.runpod.io/pods/templates/secrets

RunPod lists B200 with 180 GB VRAM, 283 GB RAM and 28 vCPUs; H200 with 141 GB VRAM,
276 GB RAM and 24 vCPUs. The GPU capacity is not pooled across cards by this code.
B200 architecture support began with CUDA 12.8. B300-specific accelerated kernels
need a newer compatibility qualification, so this package does not require a
blind runtime upgrade just to advertise the largest available card.

B200 is a planning recommendation for weight + activation headroom, not a measured
minimum, rendering-time benchmark or cost-optimal result. Network storage of 500
GB fresh / 1 TB retaining existing assets is likewise a planning allowance.

## Optional helper

https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16

The publisher lists 33.4B parameters, F16, approximately 67 GB. This is retained
because the user preferred its earlier output, not because a new benchmark
proved it is the best VLM. It is separate from the 51.5 GB H3 encoder. Helper
inference remains local, optional, bounded and unloaded before generation.

## What was not claimed

No native-1080p upgrade, no skin-color-fix guarantee, no exact-reference or motion
lock, no full-FP32 inference, no automatic quality certification, and no actual
GPU H3 acceptance. The lightweight preview decoder remains approximate and never
replaces the final video VAE. Generation remains single-reference by default;
the two-reference workflows are retained as optional experiments.

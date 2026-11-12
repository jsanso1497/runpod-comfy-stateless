# RunPod template settings

Use a new template named **H3 Video Still Studio - A100 80GB**.

| Setting | Value |
|---|---|
| Container image | Exact `ghcr.io/...@sha256:...` from the successful **Build H3 Video Still Studio** action summary |
| GPU | 1 x NVIDIA A100 SXM **80 GB** |
| Host RAM | At least 128 GB for INT8 HQ; 256 GB recommended for BF16 full and multiple high-resolution references |
| vCPU | Prefer 12 or more for model loading and video decoding; this is a provisioning recommendation, not a measured minimum |
| Container disk | 50 GB |
| Persistent disk | 250 GB for HQ and inherited assets; 500 GB for HQ plus all BF16 sets |
| Persistent mount | `/workspace` |
| Storage kind | A real network volume for data that must survive Pod deletion; do not use object-storage-style Global Volume for the model cache's symlinks and lock files |
| HTTP ports | `8188,8888` |
| Public TCP ports | None needed |
| Internal ports | Do not expose `8189` or `8889` |
| Docker start command | Leave blank |
| Docker entrypoint override | Leave blank |
| CUDA / host driver | Existing base uses PyTorch cu130; choose a host compatible with CUDA 13 and NVIDIA R580 or newer |
| Built-in RunPod Jupyter override | Do not replace the image's own launcher with another Jupyter startup command |

Environment variables:

```text
COMFY_PASSWORD=<your strong password>
H3V_PRELOAD=hq
H3_STILL_DATA_ROOT=/workspace/h3-still
```

Set `H3V_PRELOAD=all` instead of `hq` to download the full-precision assets too. After confirming applicable authorization to use H3, also set:

```text
H3V_LICENSE_CONFIRMED=1
```

Optional `HF_TOKEN` can be a read-only Hugging Face token when access/authentication requires it. Never put the token in the repository or workflow JSON. Use RunPod secrets where available. Do not expose ComfyUI's private internal port to bypass the authenticated gateway.

The new image is **not published yet**. Do not paste a guessed tag/digest from these documents into RunPod. Build the action first and copy its actual image reference.

Weights download to the persistent volume on first boot and are checked against immutable publisher revisions and SHA-256 values. Existing verified files are reused. Expect the first start to be a large download; Files/Jupyter are independent of the model backend. Do not repeatedly delete/redeploy the persistent volume while downloads are in progress.

Primary documentation reviewed:
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
- https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage
- https://www.runpod.io/blog/global-volumes-arent-network-volumes

RAM, disk headroom, reference length and output recommendations above are design choices for this package, not measured A100 benchmark results.

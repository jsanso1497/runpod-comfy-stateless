# Stateless RunPod ComfyUI

A disposable ComfyUI environment for RunPod. The image contains CUDA/PyTorch, download tooling, Jupyter/SSH support and SageAttention. Each Pod boot pulls a fresh ComfyUI checkout, installs your custom nodes, downloads models and LoRAs from manifests, copies workflows, performs GPU checks, and starts ComfyUI.

## Why this design

- No persistent RunPod volume is required.
- Models and LoRAs are not baked into the Docker image.
- The Pod can be terminated when the session ends.
- The same image can be reused across common NVIDIA GPU generations.
- ComfyUI can update independently from the Docker image.
- A separate config repo can control models, LoRAs, nodes and workflows without rebuilding the image.

## Build

```bash
docker build -t runpod-comfy-stateless:latest .
```

To skip SageAttention during a development build:

```bash
docker build --build-arg INSTALL_SAGEATTENTION=0 -t runpod-comfy-stateless:lite .
```

The included GitHub Actions workflow publishes to:

```text
ghcr.io/<your-github-username>/runpod-comfy-stateless:latest
```

## Recommended RunPod template

Use the published image and configure:

```text
Container disk: 300-600 GB depending on your model library
Volume disk:    0 GB
HTTP port:      8188
Jupyter port:   8888
SSH port:       22
```

Recommended environment values:

```text
COMFY_REF=master
ATTENTION_BACKEND=auto
VRAM_MODE=auto
ENABLE_DYNAMIC_VRAM=auto
ASSET_DOWNLOAD_WORKERS=4
CONFIG_REPO=https://github.com/YOU/your-private-comfy-config.git
CONFIG_REF=main
```

Store these as RunPod Secrets where applicable:

```text
HF_TOKEN
CIVITAI_TOKEN
GITHUB_TOKEN
JUPYTER_TOKEN
PUBLIC_KEY
```

## Config repo layout

The image accepts a `CONFIG_REPO` with this layout:

```text
config-repo/
  models.json
  loras.json
  custom_nodes.json
  workflows/
    workflow-1.json
    workflow-2.json
```

If `CONFIG_REPO` is blank, the config baked into this repository is used instead.

## Model manifest

Hugging Face:

```json
{
  "name": "Wan model",
  "source": "huggingface",
  "repo_id": "OWNER/REPO",
  "filename": "wan.safetensors",
  "destination": "models/diffusion_models/wan.safetensors",
  "enabled": true,
  "required": true
}
```

CivitAI:

```json
{
  "name": "My LoRA",
  "source": "civitai",
  "model_version_id": 123456,
  "destination": "models/loras/my_lora.safetensors",
  "enabled": true,
  "required": false
}
```

Direct URL:

```json
{
  "name": "VAE",
  "source": "url",
  "url": "https://example.com/vae.safetensors",
  "destination": "models/vae/vae.safetensors",
  "enabled": true,
  "required": true
}
```

Optional `sha256` can be added to any asset to verify the download.

## Attention settings

`ATTENTION_BACKEND` accepts:

- `auto`: use SageAttention when its import and GPU architecture check pass, otherwise PyTorch attention.
- `sage`: force ComfyUI `--use-sage-attention`.
- `pytorch`: force ComfyUI `--use-pytorch-cross-attention`.
- `default`: let ComfyUI choose.

If a workflow or custom node is incompatible with SageAttention, set `ATTENTION_BACKEND=pytorch` in the RunPod template without rebuilding the image.

## Startup sequence

```text
GPU/CUDA check
    -> clone config
    -> clone current ComfyUI
    -> install ComfyUI requirements
    -> clone custom nodes
    -> install node requirements
    -> download models in parallel
    -> download LoRAs in parallel
    -> copy workflows
    -> select attention backend
    -> start ComfyUI
```

## Important

The Pod's container disk is disposable. Anything generated under `/workspace` is lost when the Pod is terminated. Export outputs you want to keep before termination.

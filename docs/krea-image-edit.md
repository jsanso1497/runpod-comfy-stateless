# K01: Krea 2 image edit, multiple references and LoRAs

Source integration added October 12, 2026. CPU contract tests and graph validation are distinct from a successful Docker build or GPU image-quality test. This integration has not been GPU-certified.

## Launch

In this repository, open **Actions > Build HQ workspace > Run workflow** and select **krea**. After the build succeeds, use its reported image digest in RunPod. Do not assume the `krea-hq` tag exists before that build succeeds.

Use the configurator's new **Krea** workspace and **K01** task. To update the public configurator, manually run **Actions > Publish configurator to Pages**, checking **publish_public_catalog**. That workflow regenerates the configurator from the merged catalog. The build also regenerates it for the container. Do not reuse a Qwen workspace image for this graph.

Planning allocation: **80 GB GPU VRAM, 128 GB system RAM, and 250 GB model/data storage**. These are conservative planning targets, not benchmarked minima. The default bundle downloads both Krea Raw and Turbo BF16, the correct encoder and VAE, the full Identity Edit adapter, and Qwen 3.8 27B BF16. Together they are approximately 120 GB before private libraries and output files. For persistent storage, the configurator also allocates an 80 GB container disk.

Keep HTTP port **8188**, no public Ollama port, and a blank container start command. Essential settings:

```text
WB_WORKSPACE=krea
WB_TASKS=K01
WB_TOOLBOX=ollama
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

Existing authentication-mode behavior is unchanged. The password variable above applies to password-protected mode. Login username: `workbench`.

After preparation finishes, open **Workbench > Tasks > K01**. A factory copy is also available under **Workbench Factory / 0.1.0 / K01**.

## Use the workflow

1. Upload the **SOURCE** photo.
2. Upload references into the optional slots, enable only the slots you need, and describe what each provides. Examples: facial identity, body proportions, clothing details, or another view of the same person.
3. Write the requested change in **Edit instructions + Qwen 3.8 27B**. Enhancement is ON. Exact LoRA trigger words go in `trigger_words`; those words are appended unchanged after rewriting.
4. Select the checkpoint and optional LoRAs, then queue. Inspect the final image, the reference-sheet preview, and the final-prompt preview.

The source photo is always the image being edited. Disabled reference slots do not require an uploaded image. Changing a role or enabled reference invalidates the dependent prompt and generation inputs.

## What multiple references means here

The upstream Krea2 Identity Edit node pack supports **two native image inputs**. This graph does not pretend otherwise.

- With no extra reference, it performs source-only editing.
- With one extra reference, that image is passed directly as Krea's second input.
- With two to four extra references, they are fitted into a labeled reference sheet, passed as the second input. The source remains the first input.

Qwen sees the source and every enabled reference separately when rewriting. Krea sees the source plus the sheet. Active references are numbered R1 onward in slot order. The preview shows exactly which sheet Krea receives. Multiple unrelated identities can still blend; role labels are guidance, not a guarantee of identity isolation. This is an integration strategy, not a claim of independent five-image conditioning.

## Checkpoints and LoRAs

The default is **Krea 2 Raw BF16**. Auto sampling uses **20 steps / CFG 3** for the supplied Raw checkpoint and **8 steps / CFG 1** for the supplied Turbo checkpoint. Both use Euler / simple, denoise 1. Custom checkpoints require an explicit Raw, Turbo or Custom sampling preset; filenames alone do not establish the correct schedule.

The checkpoint selector accepts registered **Krea 2-compatible diffusion models only**. It is not a universal SD/FLUX/Qwen checkpoint selector. Krea's **Qwen3-VL 4B BF16 encoder is always loaded with type `krea2`**, avoiding the previous conditioning-width mismatch. That encoder is separate from the 27B prompt enhancer.

The **Identity Edit v1.2 adapter** stays enabled, default strength 1.0. Two additional Krea-compatible LoRA slots are optional, each with its own strength. `none` or strength zero skips an optional slot. Duplicate active LoRAs are rejected rather than applied twice.

To expose private assets, use the existing runtime-only library secrets. No private URL belongs in a committed workflow. Add `WB_LORA_URLS={{ RUNPOD_SECRET_comfy_loras }}` and/or `WB_CHECKPOINTS={{ RUNPOD_SECRET_comfy_checkpoints }}` to the Pod when using those secrets.

An entry inside the private LoRA secret can use:

```text
krea | https://huggingface.co/YOUR_ACCOUNT/YOUR_REPO/resolve/main/YOUR_LORA.safetensors
```

A private checkpoint entry needs an explicit architecture declaration:

```json
[
  {
    "url": "https://huggingface.co/YOUR_ACCOUNT/YOUR_REPO/resolve/main/YOUR_MODEL.safetensors",
    "family": "krea",
    "architecture": "krea2",
    "kind": "diffusion_models",
    "precision": "bf16",
    "name": "my-krea-checkpoint"
  }
]
```

Prepare the private library and refresh the workflow's selectors. Registration is a compatibility declaration, not a guarantee that an arbitrary third-party LoRA will perform well. Private weights are downloaded at runtime, not baked into the image.

## Prompt enhancement and memory

The default local Ollama model is **`qwen3.8:27b-bf16`**, approximately 56 GB. There is no fallback to the quantized default tag. Ollama is private to the container, and images are not sent to an external API by this node.

Enhancement unloads ComfyUI-managed models first. The Ollama request uses `think: false` and `keep_alive: 0`. The model-loading node depends on the finished prompt, so Krea loading follows enhancement rather than racing it. A clear error is returned when the helper is unavailable or its output is truncated; it does not silently use an unrelated prompt. Toggle `enhance` OFF for manual prompting. The helper is still part of this workspace's default download bundle.

The enhanced prompt appears in a read-only preview on the prompt node after execution. Its trigger-word suffix is preserved exactly.

## Resolution and preservation

The canvas follows the source aspect ratio, rounded down to multiples of 16, with a default cap of **1.9 MP** and a maximum of **2 MP**. Small sources are not automatically enlarged. FIT geometry avoids cropping references to match the output aspect ratio. The same target latent is supplied to pre-encoding and sampling to avoid loading the VAE for the first time inside the sampling loop.

Both positive and empty negative conditioning are grounded on the same image inputs. Starting reference strength is 4, scene strength 1, grounding resolution 768. These controls remain editable and are not claimed as universally optimal.

This is a **full-frame generative edit**, not masked crop/stitch compositing. Unrequested pixels are not guaranteed identical. Do not use it as an exact-pixel overlay tool.

## Source and verification

- Native Krea components: https://huggingface.co/Comfy-Org/Krea-2
- Full edit adapter: https://huggingface.co/conradlocke/krea2-identity-edit
- Editing nodes and usage: https://github.com/lbouaraba/comfyui-krea2edit
- Official Qwen model: https://huggingface.co/Qwen/Qwen3.8-27B
- Ollama tags: https://ollama.com/library/qwen3.8/tags
- Ollama chat API: https://docs.ollama.com/api/chat
- Qwen 3.8 support introduced in Ollama 0.32.12: https://github.com/ollama/ollama/releases/tag/v0.32.12

The repository already pins Ollama 0.35.1, newer than the release that introduced Qwen 3.8 support. No Ollama binary change is included.

ComfyUI and the editor node pack are commit-pinned. Downloaded Krea assets are SHA256-checked using the public metadata recorded in `catalog/krea.json`. The runtime records the resolved Ollama digest. No model weights were downloaded as part of source authoring.

Tests cover reference toggles, sheet geometry, model presets, LoRA application order, literal trigger words, enhancer payload/unloading controls, source/reference grounding, shared target-latent wiring, catalog isolation, and UI/API graph links. Final remaining gates are repository-wide CI, the Krea Docker build, actual custom-node imports on that image, and a representative GPU edit.

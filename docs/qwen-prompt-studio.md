# U11: Smart Visual Prompt Studio

## What it does

A separate, prompt-only workflow for 1-8 images. Give it a short request; it inspects the images, assigns their roles, builds a factual visual brief, and writes a final reusable prompt. It does not run the Qwen image generator, VAE, sampler, SAM, an image upscaler or a LoRA. It exports only the final text, not the images, visual brief or raw model reasoning.

The workflow has its own visual-planning and writing instructions. It does not call the old automatic edit-rewrite function. Existing editing workflows, the dual-enhancer selector and Krea remain unchanged.

The vision model is selected by the existing `WB_QWEN_PROMPT_MODEL`. Default: `huihui_ai/Qwen3.8-abliterated:27b-bf16`. The already-supported `huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16` is also usable through that variable. The official PE-I2I 9B checkpoint is not a backend for this workflow: this tool also needs general description and text-only recreation capability. No additional model download is needed when your selected Ollama model is already installed.

## The same workflow handles several tasks

| Task dropdown | Result |
|---|---|
| Auto - understand my request | The model chooses edit, new scene, recreation or description from your request. |
| Edit / combine images | An instruction referring to the attached images with Qwen image tags. |
| Recreate as a text-to-image prompt | A self-contained prompt describing the picture to generate, without image tags or assumed references. |
| Describe the image(s) | A factual description of what is visible, not an instruction to modify it. |
| New scene using references | A prompt for a new composition, with explicit reference-image contributions. |

Auto is a best-effort inference, not a guarantee. A role hint or explicit mode is useful when a request has more than one plausible interpretation. The actual interpreted image roles appear below the prompt for review but are not part of the exported text.

## Basic use

Open **Workbench > Toolbox > U11** or load `U11_Qwen_Smart_Prompt_Studio.json` directly after installing the code.

1. Upload/select Image 1. Add Image 2, 3, etc. as needed. All eight slots are already connected. Leave unused slots set to `None`; fill in order without gaps.
2. Type your request in the writer node. Leave Task on Auto, or select one of the explicit tasks.
3. Queue the workflow. It first builds a visual brief, then writes the final prompt. The same vision model stays loaded between those two calls and unloads afterward. The user-interface prompt field stays empty until the run succeeds.
4. Click **Copy final prompt** or **Download prompt .txt**. With `save_txt` enabled, a copy is also saved under `/workspace/output/prompt_studio/` for the standard template. The file contains only the prompt.

The final node has one STRING output named `prompt`, so it can be connected to a text consumer in a custom graph. The supplied workflow contains no image-generation consumer.

The Workbench Files tab may not list `.txt` files in your existing build. The writer's Download button works directly and does not depend on that file list.

## Dog replacement example

Load:

- Image 1: the scene containing dog A.
- Image 2: a close-up of dog B's face.
- Image 3: a full view of dog B.

Request:

> Replace the dog in image 1 with the dog from images 2 and 3.

Optional role hints: `Scene with dog to replace`, `Replacement dog face`, `Same replacement dog full body`.

The instructions ask the model to identify the scene dog by visible position/action, bind image 2 to head identity and image 3 to the body's appearance, and specify integration details such as pose, perspective, coat appearance under scene lighting, contact with the ground, shadows, occluding objects and any background revealed by the replacement. It is told not to invent a breed or make a generic lookalike from a verbal checklist. Actual output depends on the model and image clarity.

An illustrative structure, not analysis of unseen photographs:

> Replace the dog in <image1> with the dog shown in <image2> and <image3>. Use <image2> for the head's identity and visible facial markings and <image3> for the same dog's body proportions, coat and tail. Fit that dog to the position and action of the original scene dog, matching the scene's perspective, ground contacts, lighting and occlusion. Preserve the other subjects and setting.

A real run should replace generic spatial wording with relevant details it can actually see. More text is not automatically a better prompt.

## Single-image examples

**Recreation:** load one photograph and request `Write a text-to-image prompt to recreate this photograph as closely as possible.` The result describes the pictured subject, pose, composition, setting, framing, light, texture and focus without relying on an attached image. Text alone cannot guarantee the exact same individual or pixel-identical reconstruction.

**Description:** select Describe the image(s), then request `Describe what is actually visible, including the subject, expression or action, framing and lighting.` The result describes the current picture, not an idealized variation.

## Using the prompt elsewhere

For edit/new-scene prompts, attach the same pictures in the same numbered order in the receiving Qwen workflow. The exported text does not carry the images with it. Pay attention to workflows that put a blank canvas or a crop in image_1: their numbering/content must agree with the exported prompt. Turn OFF that workflow's automatic enhancer when you want to use the exported prompt exactly as reviewed.

For standalone recreation prompts, references are not required downstream. Set the receiving workflow's output size separately. Description mode may be useful as input material, but it is intentionally not an editing directive.

## Controls

`writing_guidance` is editable on the node. It adds user writing preferences for this run. `Detailed` aims to include useful implications, not an exhaustive inventory of things that should remain unchanged. `Concise` is shorter. The task and explicit request outrank default length preferences.

`keep_exact`: one exact text string or LoRA trigger per line. These are checked in the final prompt; no LoRA is loaded. Do not paste provider tokens or credentials here.

`vision_edge`: 1024 by default; 512-1536 available. These are read-only, aspect-preserving PNG copies sent individually to the local vision model. They are not a collage and the original input tensors are not changed. There is also a total vision-copy budget of 8 million pixels. Fine print and tiny details can be missed; crop relevant evidence or increase the edge conservatively.

`max_tokens`: 4096 per stage by default, adjustable from 1024 to 8192. Increase it after a truncation error for a complex request, rather than assuming an invalid response is useful. The context setting is 32768. This can use more memory than the older enhancer's 16384 setting.

`seed`: fixed by default for reproducibility. Change it to request a fresh variation. ComfyUI can cache identical completed node inputs.

## Failure behavior and memory

Each stage gets at most one retry for malformed, incomplete or structurally invalid output. A failed stage reports its specific validation reason and exports nothing. This tool never labels your untouched request as a successful enhancement and never switches models on failure.

Before inference, it asks ComfyUI to unload cached models. It retains only the same local vision model between the analysis and writing calls, then uses Ollama unload and residency reporting before returning. There is no simultaneous loading of an image-generation model by this graph. Cleanup also runs after API failures or cancelled processing; long in-flight non-streaming requests can still take time to return.

The graph only uses the loopback Ollama service. The existing client disables environment proxies and HTTP redirects. No source images or model reasoning are saved by the studio. ComfyUI's own workflow/history storage remains in effect, so saved workflows and prompt text can still contain private information. Treat your Pod and its outputs as private.

## Installation and RunPod

Apply the separate additive update package to the existing repository using its `APPLY_UPDATE.py`. Do not replace or delete repository directories. It verifies expected source hashes, backs up replaced files outside the repository, regenerates the site and runs both static validator commands. Unexpected file changes stop the install instead of being overwritten.

Commit and push, then run **Validate repository** and **Build HQ workspace**, selecting **qwen**. Use the successfully published image reference/digest for your Pod. Existing running containers do not receive code just because GitHub changed. Save your workflow and back up important outputs first. To update the current Pod, use its three-dot menu > Edit Pod, set the newly published image/digest and the environment values, then Save. Saving an edit resets the container and discards files outside /workspace. Confirm /workspace is actually your volume mount. Do not terminate/delete a Pod to perform this update: ordinary Volume Disk is deleted with the Pod. A Network Volume is independent and can be reused by a new Pod.

For the existing combined Qwen workspace:

```text
WB_WORKSPACE=qwen
WB_TASKS=SQ,Q01,U11
WB_DATA_ROOT=/workspace
WB_TOOLBOX=ollama
WB_QWEN_PROMPT_MODEL=huihui_ai/Qwen3.8-abliterated:27b-bf16
WB_QWEN_PROMPT_FAILURE=original
WB_AUTH_MODE=basic
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
WB_SAVE_METADATA=0
```

`WB_QWEN_PROMPT_FAILURE` is kept for your older enhancer; U11 deliberately does not use its original-prompt fallback. Keep any existing private-library variables. To also prefetch the old official enhancer, set `WB_TOOLBOX=ollama,qwen_pe_i2i` instead. It is not required for U11.

Expose only HTTP 8188. Leave container start command blank. Use your existing `/workspace` volume and working GPU configuration. U11 reuses the vision-model weights, but changing to this workflow does not make the 27B BF16 model itself small. No new expensive GPU requirement is imposed by the feature; actual memory/runtime must be checked on your chosen GPU. The Qwen workspace still downloads its normal image-model defaults even when U11 is selected by itself; this update is not a separate reduced-storage Docker workspace.

Optional metadata-only check from the RunPod Web Terminal:

```bash
python -m workbench.qwen_prompt_local
```

Then open U11, load a real image and try a short description before a complex multi-reference task.

## Validation and sources

CPU tests validate routing, ordering, disabled slots, both inference stages, retries, explicit failures, model metadata checks, cleanup, prompt-only exports, safe file handling, graph links and old-enhancer compatibility. Browser script behavior is tested with a JavaScript harness. These tests do not establish live visual understanding, output quality, GPU memory use or Docker compatibility. Run the full repository CI and a real Pod smoke test after installation.

Design references checked October 13, 2026:

- Qwen's official I2I prompt conventions: https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/raw/main/system_prompt.txt
- Publisher's selected vision model and precision: https://ollama.com/huihui_ai/Qwen3.8-abliterated:27b-bf16
- Ollama structured chat: https://docs.ollama.com/api/chat
- Ollama unload requests: https://docs.ollama.com/api/generate
- Ollama residency reporting: https://docs.ollama.com/api/ps
- RunPod updates: https://docs.runpod.io/pods/manage-pods
- RunPod volume persistence: https://docs.runpod.io/pods/storage/types

This is an independent Workbench integration informed by those conventions, not an official Qwen workflow or benchmark-certified optimizer.

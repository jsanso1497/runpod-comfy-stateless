# Qwen image workflows: local vision prompt enhancer

## What changed

The previous implementation could stop generation with "Enhancer omitted a reference tag used by your original prompt." The image order was unchanged, but a missing `<imageN>` token in the generated rewrite failed strict validation. The checked GitHub version still contained that failure path.

The shared `enhance_edit_prompt` function supports a local Ollama vision model and the official native BF16 prompt enhancer. It asks for structured JSON, validates reference tags and exact terms, retries one invalid rewrite from the original instruction, and defaults to the original prompt unchanged after two invalid answers. The preview/report and console explicitly identify this fallback. Invalid output is never fixed by blindly appending image tags. The rejected answer and private model reasoning are not forwarded to image generation or logged.

A missing service, wrong model, missing vision capability, incorrect precision, or failure to release model residency still stops execution with an actionable error. These are not silently treated as successful enhancement. Invalid reference numbers in the ORIGINAL instruction also stop before inference.

This change does not guarantee semantic correctness or perfect identity preservation. Review the final rewrite, particularly for multiple subjects and exclusions.

## Choose the model

There is no verified `Qwen3.8-VL-32B` tag in this update. The two verified publisher releases are different models:

| Setting | Exact Ollama model | Weight precision | Approximate model download |
|---|---|---|---|
| Default: newest requested vision variant | `huihui_ai/Qwen3.8-abliterated:27b-bf16` | BF16 | 56 GB |
| Alternate 32B VLM via RunPod environment | `huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16` | FP16 | 67 GB |
| Specialized official prompt enhancer | `Qwen/Qwen-Image-2.1-PE-I2I` | BF16 | about 19 GB |

Both are abliterated variants. The package does not label them Heretic releases. Only the selected Ollama model is prepared, not both. The official PE-I2I BF16 model is a separate optional 19 GB download. Full-precision GGUF storage is allowed; a quantized `latest`, Q4, Q8, or text-only replacement is not silently selected.

Default RunPod environment:

```text
WB_QWEN_PROMPT_MODEL=huihui_ai/Qwen3.8-abliterated:27b-bf16
WB_QWEN_PROMPT_FAILURE=original
```

To select the older 32B vision model instead, replace only the model value:

```text
WB_QWEN_PROMPT_MODEL=huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16
```

Keep `WB_WORKSPACE=qwen`, your existing `WB_TASKS`, toolbox selections, authentication settings and private-library bindings. Restart the Qwen Pod after changing its environment. Preparation downloads the selected model. There is no checkpoint/encoder replacement in the Qwen image generator itself.

### Choose the enhancer from ComfyUI

The original `WBQwenPromptControl` now has an **optional backend dropdown after its existing three fields**:

- **Qwen3.8 27B (Ollama vision)** (default). Runs the full BF16 27B model against the actual source image and ordered reference photos. This group is included with Qwen at startup.
- **Official Qwen Image 2.1 PE-I2I (BF16)**. Loads the purpose-trained four-shard native BF16 checkpoint and uses the official editing system prompt. Select `qwen_pe_i2i` under Workbench Toolbox, or add `WB_TOOLBOX=qwen_pe_i2i` in RunPod to prefetch it. Existing cached official weights are reused after verification.

The dropdown is optional to protect existing saved workflow JSON. The three original control fields and output socket names remain unchanged. U10 previews either choice before a Qwen image is generated. The output report identifies the actual backend, precision, and final prompt. Neither mode silently substitutes the other on an inference failure. Changing this dropdown invalidates the cached rewrite.


## Existing workflows

No rewiring is needed for workflows already connected to `WBQwenPromptControl`. Its three widgets, output type and input socket names are unchanged. Existing factory/saved OFF switches remain OFF. Turn the existing enhancer switch ON to use the new backend. The same choice applies to all connected edit passes.

Use **Workbench > Toolbox > U10** to prepare and preview an enhanced prompt without generating an image. The preview identifies the actual model and any retry/fallback. Prompts enter through the same fields as before; exact LoRA trigger terms remain in `keep_exact`.

The enhancer receives the pass's source/crop and references as individual images in their original order, up to ten images. It uses bounded 768-pixel vision copies, not a collage. Original generator tensors, masks, crops, geometry, negative prompts and sampler settings are not changed. This does not add an enhancer to unrelated graphs that never had the shared control.

For strict handling instead of original-prompt fallback:

```text
WB_QWEN_PROMPT_FAILURE=error
```

## Memory, setup and privacy

Ollama listens on loopback. The client ignores proxy environment variables and rejects HTTP redirects. Reference images are not sent to an external inference service. Only model downloads contact the publisher.

ComfyUI residency is released before prompt inference. Ollama is told to unload after inference, explicitly unloaded on errors as well, and queried to verify that the selected prompt model is no longer resident before the image workflow proceeds. There is no simultaneous-residency guarantee for unrelated software that a user starts on the same GPU.

The catalog retains the 80 GB VRAM / 128 GB RAM planning target and budgets at least 250 GB of model/data storage. These are planning allowances, not measured minima. The official 19 GB enhancer is downloaded only when qwen_pe_i2i is requested from Toolbox; its existing cached files are not deleted. The Krea and H3 enhancer selections are unchanged.

After rebuilding and launching the Qwen image, this optional terminal command checks local model metadata without generating anything:

```bash
python -m workbench.qwen_prompt_local
```

Then preview a real source plus reference through U10 before running a full edit. Docker build, local Ollama inference, VRAM behavior and output quality have not been tested in the package-authoring environment.

## Sources checked October 12, 2026

- Publisher's 32B model: https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16
- Publisher's 32B vision model card: https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated
- Publisher's Qwen3.8 model: https://ollama.com/huihui_ai/Qwen3.8-abliterated:27b-bf16
- Ollama structured chat and keep_alive: https://docs.ollama.com/api/chat
- Ollama residency reporting: https://docs.ollama.com/api/ps

Publisher metadata was verified. Model weights were not downloaded. No superiority of either model is claimed without the same-image A/B quality test.

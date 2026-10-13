# Optional Qwen I2I prompt enhancement

## What is included

All 35 Qwen generation workflows, including `SQ / Standard HQ`, have one **Qwen I2I Prompt Enhancer ON / OFF (BF16)** control. All 40 edit passes are connected. The 34 corresponding existing API exports have the same control.

The helper is **Qwen/Qwen-Image-2.1-PE-I2I**, the Qwen team's image-aware editing prompt rewriter, using the four original BF16 shards. It is not a generic text-only prompt expander, the text-to-image enhancer, a quantized repack, or an API model. It runs locally through the existing pinned ComfyUI Qwen3.5 backend. No Ollama service or new Python inference package is added.

This is the most directly matched official option selected for this integration. No comparative benchmark or GPU image-quality result is claimed.

## Normal use

1. Open a fresh workflow from the Workbench catalog. Continue entering the edit in its existing prompt field.
2. Set the master **enabled** control to ON to enhance, or leave it OFF to use the original prompt. All templates default to OFF so the Standard HQ baseline and previous behavior remain reproducible.
3. Queue the workflow. Each Qwen encoder shows its original prompt and the final prompt actually sent to the image model. The preview is read-only; the original prompt field remains editable.

With enhancement OFF, the enhancer does not inspect images, load weights, or run inference. Existing Photo Edit normalization and its user-provided reference-role suffix behave exactly as before. Turning enhancement off is not a promise of bit-identical GPU image output across machines or software versions.

With enhancement ON, each pass receives the same ordered, preprocessed images as the downstream Qwen image encoder. Portrait and landscape references are not stretched into a common batch. Crop, mask, original-pixel protection, negative prompt, resolution, LoRA strengths, generation seed and sampler settings are not changed. The helper is instructed to preserve explicit reference assignments and edit constraints, but model-authored wording still needs review.

The encoder receives image crops and references, not an additional mask image. The real edit/protection masks remain in the original processing path. The helper cannot inspect scene context outside those crops or guarantee semantic preservation on its own.

## Multiple passes and seed comparisons

The one master switch controls all passes. Sequential body/head or two-person edits are enhanced separately, using each pass's images and instruction, including intermediate results when applicable. A fixed enhancer seed is independent of the image-generation seed. A small in-memory, content-keyed prompt cache reuses an identical rewrite for matching inputs, helping keep a three-seed image comparison consistent. Changing images, prompt, enhancer seed or keep-exact terms invalidates that cache.

## Preview without generating an image

Open **Toolbox > U10: Preview an enhanced Qwen edit prompt**. Add the source and optional references, preserve the same image order as the target workflow, enter the instruction, turn the helper ON, then queue. This runs the enhancer only, not the Qwen image generator.

For maximum correspondence with a cropped edit, connect the same working crop and prepared references. A preview using a full source photograph is not necessarily the same rewrite as a subsequent crop-based edit. The U10 preview does not automatically populate another workflow; copy the reviewed text into that workflow and leave its enhancer OFF to use that exact text.

## Private triggers and output validation

The optional **keep_exact** field accepts one trigger or literal phrase per line. Matching terms are protected only when present in that pass's original prompt. It does not activate or load a LoRA. Double-quoted text and explicit `<lora:...>` strings are also checked for exact retention.

Only the final `rewritten_prompt` is passed to the generator. The helper's `wh_ratio` and `ratio_follow` suggestions are recorded but never applied. Raw model thinking is not displayed or retained in the report. Invalid JSON, truncated reasoning, nonexistent image tags, omitted named reference tags, or missing protected literals stop that pass with an actionable error. There is no silent fallback to another model or an unreviewed prompt.

The runtime never opens your private LoRA list to construct these instructions. However, prompts and manually entered trigger text remain part of a workflow's normal metadata. Keep personal workflows private when those strings are sensitive.

## Assets, deployment and storage

The Qwen workspace automatically prepares approximately **18.82 GB (decimal)** of original BF16 enhancer weights plus two small official companion files. This startup download happens even while the graph's switch remains OFF, so later switching does not require environment-variable changes. Other workspaces do not request this group by default. Reusing a persistent, verified model cache avoids downloading those shards again.

The enhanced configurator includes this storage estimate. No additional secret is required. Existing `hf_token`, `civit_token`, and `comfy_password` references are unchanged.

The image must be rebuilt for this code update. Upload the overlay into the current repository without deleting it; run **Validate repository**, then **Build HQ workspace** with `qwen`. Deploy the newly reported image digest, not an older cached tag. No changes to `.github` files are included in the overlay.

Inference releases ComfyUI model residency before and after running the helper. The model is loaded in full precision, runs sequentially, and is not kept on the GPU alongside the image generator. It still needs extra system memory and adds prompt-generation latency. Neither peak memory nor generation time has been measured on a GPU for this release. The existing hardware targets are planning estimates only.

Missing assets do not make OFF unusable. For download issues, open U10 in Workbench and choose **Prepare assets**, then review download status. A checksum mismatch must be repaired, not ignored.

## Existing saved workflows

The catalog's fresh templates include the switch. Personal saved copies are deliberately not overwritten. To upgrade a personal copy, add the **Qwen | I2I Prompt Enhancer ON / OFF (BF16)** node and connect its output to `prompt_enhancer` on each Qwen reference encoder. A copy without that connection continues using the original prompt.

## Provenance and verification limits

Model revision: `72927bc08afc99b7888ceb7d7d51a12db3700bbd`.

ComfyUI remains pinned to `52f98af2e2e42c421070a3e147c161c47cdeaf22`. The integration merges the four original shard dictionaries in memory, verifies the tensor index and model family, and invokes the native Qwen3.5 text-generation path. It does not save a second merged 19 GB checkpoint.

Weight downloads enforce the official LFS SHA256 values. Companion files are restricted to the official model's system prompt and tensor index at the pinned revision, and verified against Git-blob checksums. This does not relax the private-library downloader's safetensors-only rules.

Primary references:
- Official model and BF16 inference example: https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I
- Pinned official assets: https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/tree/72927bc08afc99b7888ceb7d7d51a12db3700bbd
- Native loader: https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy/sd.py
- Native image-aware tokenizer/generation: https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy/text_encoders/qwen35.py
- Model license: https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE

The model card identifies the Qwen Research License; review the actual terms for the intended use. Model licensing is not changed by packaging the integration.

Offline tests use synthetic images and mocked inference. They establish routing, cache behavior, disabled-path equivalence, graph coverage, response validation and download integrity behavior, not live GPU compatibility or improved images. A real Docker build, model download and GPU smoke test remain required.

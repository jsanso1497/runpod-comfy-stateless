# Krea parity with the current Qwen build

## Scope

This is an additive Krea integration, not a replacement of the Qwen, FLUX, H3 or restoration builds. Existing model hashes, upstream software pins, registered private LoRAs/checkpoints, secret aliases, shared gateway and Civitai Red support are retained.

Three editable K01 graphs and matching API exports are supplied. The original two filenames remain stable. The new graph is `Krea_Masked_Photo_HQ.json`.

## Prompt enhancement

The local model selection is shared with Qwen through `WB_QWEN_PROMPT_MODEL`:

- Default: `huihui_ai/Qwen3.8-abliterated:27b-bf16`.
- Alternate: `huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16`.

New Krea graphs use `ollama_model=auto`. The former `qwen3.8:27b-bf16` setting is a migration alias to the selected shared model. An explicit supported tag remains allowed, but must be separately installed when it differs from the workspace selection. There is no quantized or text-only substitute.

The enhancer remains ON by default in Krea; Qwen's existing OFF defaults are unchanged. OFF does not resolve a model, contact Ollama, or alter the written instruction. Identity Edit still adds its deterministic source/reference-role preamble; trigger words are appended as before. Standard returns only the written standalone prompt and trigger words.

Local ON checks vision capability and precision, sends source plus individual references to the loopback-only service, and verifies Ollama unload before returning to the image model. It reuses Qwen's proxy-bypassing, redirect-rejecting transport. Invalid, truncated, or reference/trigger-dropping responses retry once, then visibly retain the original instruction. `WB_QWEN_PROMPT_FAILURE=error` stops instead. Missing models, transport problems, precision mismatches and unverified unloading always stop explicitly.

Krea has its own rewrite instructions because its generator has two native image inputs. Extra references are a labeled sheet, not additional native inputs. Enabled slots are relabeled R1 onward in order. Check the sheet and roles when disabling intermediate slots.

The optional official PE-I2I selector is available only in Identity Edit and Masked Photo. Prepare `qwen_pe_i2i` in Toolbox first. It uses the shared official engine and sees exactly the source and reference sheet, not the individual extra references. This is an integration candidate, not a benchmarked quality claim. Standard stays local-vision/text-only and rejects an edit-only backend.

U11 Smart Visual Prompt Studio is available separately in Krea and Qwen. It is a prompt authoring tool, not an image generation or identity-conditioning path.

## Masked editing and color

The masked graph uses the existing Qwen photo mask/geometry implementation privately. It does not register or load Qwen generation models.

Paint and save a source mask, or prepare the optional `sam` assets and enable native SAM 3.1. Selection always uses source A. Empty masks, mismatched coordinates and arbitrary generated-crop resizing are rejected. Expand and outward feather are explicit source-pixel controls. Optional protection on the crop node is applied after expansion, so protected pixels cannot be grown back into the edit.

The default working crop is below 1.9 MP. Every supported setting remains below Krea's 2 MP cap, using padded dimensions. Qwen's larger area budget and model-specific cache are not copied into Krea. Raw remains 20 steps / CFG 3; Turbo remains 8 / CFG 1. There is no automatic upscaling of small crops.

`run_edit=OFF` previews the final mask without evaluating the generation, reference-loading or prompt-enhancement branch. Turn it ON only after reviewing the boundary. The decoded crop is unpadded, restored to source coordinates and pasted through the reviewed mask. The saver independently verifies exact preservation outside that mask and writes a full-size PNG plus a verification JSON.

Input ICC profiles are converted to sRGB. Untagged files are assumed to be sRGB. Preservation is relative to that normalized source tensor, not the original encoded channel numbers of a non-sRGB file. All saved PNGs carry an sRGB profile. Optional conservative boundary color matching is OFF by default; it cannot guarantee elimination of every visible seam.

SeedVR2 is installed as an optional, separate finishing tool. Its weights are not added to default downloads and no restoration pass is inserted into basic Krea graphs. Applying a full-frame restoration pass later does not inherit masked-stitch pixel preservation.

## Use the update

Run the existing **Build HQ workspace** Action for workspace `krea`, then use the resulting immutable image digest for a new/recreated Pod. Keep the existing private library registrations and RunPod secret bindings. The shared runtime still uses `/opt/ComfyUI`, the Workbench gateway on 8188, and its internal ComfyUI port; do not start a second process on the gateway port.

Docker builds regenerate the Krea graphs and local configurator from source. The public GitHub Pages configurator is a separate manual **Publish configurator to Pages** Action. The checked-in generated `site` snapshot is not a deployment status indicator.

Open the fresh graphs from the Workbench task sidebar. Persistent factory files and personal saved graphs are deliberately not overwritten. Personal older graphs gain updated node behavior after rebuilding, but do not acquire new source loaders, save nodes or a masked branch automatically.

## Validation boundary

Focused CPU contracts cover reference-sheet routing, correct encoder/sampling presets, enhancer OFF, retry and explicit-error paths, unload-on-error, text-only Standard isolation, graph/API parity, crop limits, protected holes, source-coordinate preservation and sRGB export. Service/model inference is mocked. No image weights were downloaded and no Docker build, SAM inference, Krea GPU inference or visual likeness test was performed in the authoring environment. Full repository CI and a representative live Pod test are still required before treating this as production-certified.

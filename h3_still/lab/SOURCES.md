# Reviewed sources and integration boundaries

The manifest locks exact Git commits, Hugging Face revisions and asset SHA-256 values. This file records the source of each integration decision. Provider code is fetched unmodified during image build and retained with its own licenses.

## Existing user workspace
- Repository baseline: https://github.com/jsanso1497/runpod-comfy-stateless/tree/c8ad356ff6f7d2ca327c0395b21ec608062b900c
- ComfyUI stable core: https://github.com/Comfy-Org/ComfyUI/tree/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
- Original Fizgig implementation: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/10d5171f6e45b0757a03df977f53a660e060c33d
The successful diagnostic JSON is included unmodified. No experimental provider is applied when the experiment controls are off.

## Spectrum
- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/tree/806fe498d574a726fecb8818917f00a4b993fd77
The wrapper invokes the real SpectrumApplyMiniMaxH3 class. It uses the documented native ER-SDE full/single-pass configuration with degree 1, RAM history and audio blending disabled. Forecasting is approximate. The package does not state a speedup or identity advantage before measurement.

## RefMod
- https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/517e8c419a8584fbf69fceb487bc39d83e6a6ec8
Save invokes the real extractor and saves one original per file. Load invokes the actual loader and RefModTextEncode path. The latter already attaches the references; no second Apply node is added. Full references are not LoRAs. Compression is optional, pooling-only in this first test, and can lose information. Full-reference identity parity is not assumed.

## Spark
- https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.2.0
- https://github.com/zechengtang/Spark-H3/blob/comfyui-v0.2.0/comfyui_nodes.py
- https://github.com/zechengtang/Spark-H3/blob/comfyui-v0.2.0/comfyui/standalone/README.md
The reviewed schema uses mode=fidelity, not preset=fidelity. The step count follows the actual sampling plan. The published SM86 Linux archive is SHA-256 locked; the isolated 0.2.37 backend does not replace official comfy-kitchen. A100 is intentionally rejected because its SM80 architecture does not match the released package. GPU kernel probing happens on the Pod, not in CPU-only CI.

## Character Swap
- https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
V1.5 is a video-oriented adapter. Applying it to a one-frame REF2VA workflow is explicitly an experiment; no unrelated Turbo mixture is silently copied.

## P-H3
- https://huggingface.co/PrunaAI/P-H3-Image-Comfy
- https://github.com/Comfy-Org/ComfyUI/blob/fca60fe70e5b84c5c6b058263653d15cb3401e44/comfy_extras/nodes_minimax_h3.py
The optional ph3 build overlays the pinned newer native core in a separate image profile. The graph uses the full INT8 trio, native PH3ImageConditioning, shifts 1/1, the published 8-step grid, Euler and native VAE decoding. No prompt upsampler is installed. Native short-edit prefix behavior is disclosed in the run report. The image does not replace the stable default core.

## SAM
- https://huggingface.co/Comfy-Org/sam3.1/tree/5febd4769e8802cdfdb75e1f733abd8c68434a85
Native SAM3_Detect is already present in the stable core. The checkpoint download is optional. It supplies only the final protection mask; it does not make H3 perform native masked inpainting.

## Source-inspired interface adaptations
The new role/crop/report/contact-sheet controls are local implementation choices inspired by the reviewed posts, not copied quality claims:
- https://www.reddit.com/r/comfyui/comments/1xraqls/h3_multi_character_swap_workfow_14/
- https://www.reddit.com/r/StableDiffusion/comments/1wsb4h3/minimax_h3_ref_fizgig_nodes_as_sota_ref2img/
- https://www.reddit.com/r/StableDiffusion/comments/1xqx0jt/minimax_h3_single_frame_edit_excels_at_character/
- https://www.reddit.com/r/comfyui/comments/1xqqhxh/minimax_h3_best_remix_tool_friendly_ultrafast/
- https://www.reddit.com/r/comfyui/comments/1xn9ufc/pruna_ph3image_is_now_available_in_comfyui/
- https://www.reddit.com/r/comfyui/comments/1wxixua/minimax_h3_image_generation_create_consistent/

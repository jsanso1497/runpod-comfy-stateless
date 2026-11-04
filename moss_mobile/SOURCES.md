# Sources and selection record

Checked November 3, 2026. Source citations below document upstream behavior and the rationale; they are not claims that this delivered package has been GPU-tested.

## Primary references

- Native ComfyUI App Mode documentation: https://docs.comfy.org/interface/app-mode
- ComfyUI release pinned here: https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0
- Native edit node and format implementation: https://github.com/Comfy-Org/ComfyUI/blob/v0.43.0/comfy_extras/nodes_moss_image.py
- Native text generation node: https://github.com/Comfy-Org/ComfyUI/blob/v0.43.0/comfy_extras/nodes_textgen.py
- Backend requirements with frontend 1.57.4: https://github.com/Comfy-Org/ComfyUI/blob/v0.43.0/requirements.txt
- Frontend App Mode data and legacy numeric-node-ID migration: https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.57.1/src/stores/appModeStore.ts
- Frontend linearData types: https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.57.1/src/platform/workflow/management/stores/comfyWorkflow.ts
- MOSS project: https://github.com/OpenMOSS/MOSS-Image
- OpenMOSS Base sampling guide: https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
- OpenMOSS native Comfy guidance: https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md
- Official Comfy MOSS model packaging: https://huggingface.co/Comfy-Org/MOSS-Image
- HQ edit adapter author's recommendation, armV5 versus armV6 and eight-step limitations: https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ
- Official Comfy SFT generation template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json
- Official Comfy native chinoll edit template, for comparison only: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_edit.json
- RunPod prices: https://www.runpod.io/pricing
- RunPod stop/start behavior: https://docs.runpod.io/pods/manage-pods
- RunPod network volume persistence, pricing and availability: https://docs.runpod.io/storage/network-volumes

## Pinned model provenance

Every actual download uses the immutable revision and SHA-256 digest in `models.lock.json`, not a moving `main` URL. File metadata was inspected at these paths:

- https://huggingface.co/Comfy-Org/MOSS-Image/blob/530a4c94f92c35ef6818b1b035a983fe9dcf5f54/diffusion_models/moss_image_sft_bf16.safetensors
- https://huggingface.co/Comfy-Org/MOSS-Image/blob/530a4c94f92c35ef6818b1b035a983fe9dcf5f54/text_encoders/qwen3.5_9b_bf16.safetensors
- https://huggingface.co/Comfy-Org/MOSS-Image/blob/530a4c94f92c35ef6818b1b035a983fe9dcf5f54/vae/flux2-vae.safetensors
- https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ/blob/7c4ba5407f012362761edfb04f0de87c904efcae/weights/armV5/armV5.safetensors

The HQ adapter is a community fine-tune, not an OpenMOSS official instruction-edit model. Native ComfyUI support is official infrastructure; that does not make every supported adapter official. No controlled cross-model benchmark was performed for this delivery.

## Deliberate tradeoffs

Full BF16 keeps precision conservative, but a visible advantage over every INT8 result is not established. The full VAE avoids substituting the smaller version; it is not a claim that the small VAE is visibly defective. At approximately 30.3 GB total downloads, the workspace is lean relative to the existing multi-model workbench, not a tiny model.

The high-fidelity edit defaults follow the armV5 author's full-step setup. Generation follows the official SFT template's 25-step er_sde/simple, CFG2.5, SLG-off recipe, with full-precision model components. The Base guide's 50-step recommendation is not misrepresented as proof that SFT generation needs twice as many steps. These are supported baselines, not an independently benchmarked universal optimum. Turbo, 8-step distillation, automatic face/detail replacement, learned upscaling and approximate attention are omitted rather than silently presented as lossless optimizations.

Reference geometry, padding, original-size restoration, hard mask compositing, deterministic test coverage, download integrity checks and the additive local installer are original package implementation choices. No upstream JSON is falsely described as copied byte-for-byte.

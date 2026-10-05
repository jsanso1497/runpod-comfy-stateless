# Primary sources used for this recipe

Krea-specific source inspected for this correction; other pinned sources retained from the prior package. Claims about comparative quality are NOT inferred solely
from file precision, parameter counts, node names, or the publisher's marketing.
The delivered GPU paths have not been benchmarked on the user's references.

## RefMod

- https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/f9462081e28794389b5a6c5067eb327412ad8ee7
- https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/f9462081e28794389b5a6c5067eb327412ad8ee7/nodes.py
- https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/f9462081e28794389b5a6c5067eb327412ad8ee7/prompt.py

The source defines Full Reference/Compressed Reference, token-budget policy,
retention and scrambling. Apply appends visual reference blocks. RefMod Text
Encode already attaches them. This package avoids double injection in both paths.

## Rebalance: Krea only

- https://github.com/nova452/Rebalance-Pack/tree/53c147c72c2fbd9444765af87118caee81a26d01
- https://github.com/nova452/Rebalance-Pack/blob/53c147c72c2fbd9444765af87118caee81a26d01/krea2.py
- https://github.com/nova452/Rebalance-Pack/blob/53c147c72c2fbd9444765af87118caee81a26d01/conditioning_rebalance.py

`ConditioningKrea2Rebalance` accepts existing CONDITIONING, global multiplier and
12 per-layer weights. The selected layer weights are the upstream defaults, with
our global multiplier reduced to 1.0 rather than its 4.0 default. The source
scales the embedding tensor and copies its metadata. Applying the same transform
to both existing image-grounded branches retains the Identity Edit reference path
and avoids substituting another chat template or the pack's editing-time schedule.
This integration is not an upstream-endorsed or benchmarked likeness improvement.

## Krea

- https://huggingface.co/conradlocke/krea2-identity-edit
- https://github.com/lbouaraba/comfyui-krea2edit/tree/86f886dac23013d88996e3a2e99093ba44d322fb
- https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors

Raw + Identity Edit is a supported guided-edit path, particularly for removals.
The author also recommends Turbo for many ordinary edits. Raw is selected here
because the user requested no Turbo absent a demonstrated quality advantage,
not because we established Raw is universally better.

## H3 and prompting

- https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py

## Ollama and capacity

- https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/api/ps
- https://docs.ollama.com/faq
- https://github.com/ollama/ollama/releases/tag/v0.35.1
- https://www.nvidia.com/en-us/data-center/h200/

The model page lists a 67 GB, F16, vision-capable 32B variant. Abliteration is a
refusal-behavior modification. No better identity-understanding score is claimed.

## Deployment and licensing

- https://docs.runpod.io/pods/templates/create-custom-template
- https://docs.runpod.io/pods/templates/environment-variables
- https://docs.runpod.io/pods/templates/secrets
- https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
- https://huggingface.co/krea/Krea-2-Raw

Check the current originating model licenses and any enterprise authorization.
Code installation does not grant a model license or remove a geographic,
commercial, consent, or disclosure condition.


## Multi-reference 3.2 implementation

- Pinned RefMod creator `nodes.py`, lines 1810-2090: multiple-image Full Reference extraction normally uses a common spatial canvas and stacks time. This update instead makes one extraction per photograph.
- Pinned RefMod `bundle.py`: version-5 save/load preserves independent image tensors and metadata in order.
- Pinned native H3 `MiniMaxH3ReferenceToVideo`: accepts numbered image references and can condition the text/vision encoder with no video VAE attached.
- MiniMax's full-reference prompt-writing guide explicitly allows one subject to be defined by multiple source assets, with each asset's role stated.

Source URLs:
https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/f9462081e28794389b5a6c5067eb327412ad8ee7/bundle.py
https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/f9462081e28794389b5a6c5067eb327412ad8ee7/nodes.py
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

# Green Suit to GPT 2.5 via ComfyUI Partner Nodes (v1.1)

## Installation
Merge the ZIP contents into your existing GitHub repository root. This update does not contain API credentials. Rebuild the Qwen workspace, and make sure your pinned ComfyUI version includes `OpenAIGPTImageNodeV2`. Sign into your **Comfy account** in ComfyUI Settings > User, and fund Comfy Credits if necessary. The existing Workbench gateway login is separate from Comfy account authentication. The Comfy Partner node may require a supported secure origin; verify that login and credits work on your RunPod proxy before a paid run.

## Controls
- `WBGSBodyMask`: `person_prompt`, `hair_prompt`, `threshold`, `person_index`, `hair_guard_pixels`, `edit_expand_pixels` (0-128), and `edit_feather_pixels` (0-128). Manual protection mask can exclude hair or other pixels.
- `WBGSQwenSuit`: editable multiline `prompt`, plus Qwen stitch `feather_pixels`.
- Native `OpenAIGPTImageNodeV2`: editable multiline `prompt`, model `gpt-image-2.5-sunburst`, `quality=max`, Custom size 2336x3504, two ordered references (image 1 original, image 2 green reference). Do not attach a mask to this two-image Partner edit; ComfyUI's Partner implementation rejects masks when multiple reference images are supplied.
- `WBGSFinalize`: adjustable `edge_feather_pixels` for final compositing.

## Execution
At the final review node, select mask preview, Qwen+4K reference, or full Partner GPT edit. The full mode may spend Comfy Credits. There is no custom OpenAI HTTP client and no `OPENAI_API_KEY` requirement. The native Partner node is directly responsible for account authentication, credits and requests.

## Compatibility warning
The Partner node's dynamic `model` input and nested `images` schema depend on ComfyUI frontend/backend version. The JSON and API export are structurally validated, but **must be opened in the deployed ComfyUI frontend and checked for correct two-reference connections and nested model settings** before spending credits. If the node is unavailable, update ComfyUI or use a supported Comfy deployment. No live RunPod/Comfy login or GPU inference was tested here.

## Limitations
Pixel-perfect anatomical alignment cannot be guaranteed by generative editing. The final matte locks the external composite boundary; inspect raw GPT and overlay diagnostics for pose/identity differences.

# 1.5.3 local Reference Video addition

The new Full and Lite image builds install the pinned Hearmeman24 MiniMax Reference Pack 0.3.5 source at `7012734eabf6f98063d6eaf8ce1f9264ee803664` and patch it to local-only prompting before it is copied into ComfyUI. The Reference Pack is wired to the existing Ollama OpenAI-compatible endpoint. Hosted OpenRouter model discovery and hosted prompt execution are disabled.

New installed workflow: `H3_Reference_Video_Swap_Local_<Full|Lite>_v1_5_3`. See the repository-root `H3_REFERENCE_VIDEO_LOCAL_1_5_2.md` for usage.

# H3 Portrait 1.5

Read `../START_HERE.md` for upload and RunPod instructions.

Lite installs:

- `H3_Portrait_Lite_v1_5` - Ollama-directed H3 video with role-aware refs.
- `H3_Ref2VA_Standard_Lite_v1_5` - manual native H3 Ref2VA baseline.
- `H3_Portrait_Image_Lite_v1_5` - Ollama-directed **MiniMax H3 Ref2VA** single-image workflow. It reuses the Lite H3 model stack, renders the native five-frame minimum packet, and saves one selected frame.

Full installs the first two Full variants only. No separate still-image model assets are required.

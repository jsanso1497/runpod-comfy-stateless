# Approved workflow register

## 00 | SAM-Guided Local Edit

**This is the only user-approved workflow so far.** Approval records the user's selected workflow; it is not an independent GPU or output-quality certification.

- Task ID: `QLOCAL`
- Workflow: `workflows/qwen/00_Approved/00_SAM_Guided_Local_Edit.json`
- Source: the supplied `Qwen21_SAM_Watermark_Mask_Guided_v2.json`
- Approved on: October 19, 2026
- Stack: Qwen Image 2.1 BF16 model, matching encoder and VAE, plus SAM 3.1.
- Inference settings and all 31 connections are preserved. The public upload filename is blank and the output folder is `SAM_Guided_Local_Edit`.

The initial target and instruction still use watermark removal, preserving the selected preset. For another local cleanup or reconstruction, change the SAM target and the edit instruction together. This workflow masks the selected area in Qwen's input, so it is primarily for removal/reconstruction, not unrestricted full-image editing.

The dedicated image lists this alone in the Approved sidebar. Its native ComfyUI factory copy is placed in `00_Approved/00_SAM_Guided_Local_Edit.json` for alphabetical priority. Existing saved workflows and old factory directories on a reused volume are not deleted or reordered. User-defined Favorites or custom sort settings may still take priority in the native browser.

No prompt enhancer, Ollama service, SeedVR2 model, MOSS model, Krea/Flux model, private LoRA library, or external custom-node pack is prepared by the dedicated image. The existing general-purpose workspace images remain separate and unchanged.

## Deployment

After applying the local update, commit and push from GitHub Desktop. In GitHub Actions, run **Build SAM-Guided Local Edit**. Use `ghcr.io/jsanso1497/runpod-comfy-stateless:qwen-local-edit-v1` only after that workflow reports success. The build also records an exact digest and commit-specific tag. This update package does not publish an image itself.

Create a fresh RunPod template. NVIDIA GPU; one A40 with 48 GB VRAM is the suggested starting point. Select 64 GB or more system RAM where available. This is a planning recommendation, not a certified resource minimum. Container disk: 60 GB. Volume disk: 100 GB, mounted at `/workspace`. Expose HTTP 8188 only; no TCP ports; leave the start command empty.

Add only these environment rows:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

The `comfy_password` secret must contain at least 16 characters. Username: `workbench`. Other profile settings are baked into this image: `WB_WORKSPACE=qwen`, `WB_APPROVED_ONLY=1`, `WB_QWEN_ONLY=1`, `WB_TASKS=QLOCAL`, no optional toolbox, compiler disabled, standard metadata off, and password authentication on. Do not copy extra environment rows from an older template.

The source image still needs uploading in ComfyUI. Keep the approved crop defaults while testing. SAM mask detection and final generated quality require visual inspection. Output stays at the original canvas dimensions; this is not an AI upscaling workflow. Model files and outputs stored on the volume survive stop/start but not Pod deletion.

Public reference: https://docs.runpod.io/pods/templates/manage-templates
Storage reference: https://docs.runpod.io/pods/storage/types

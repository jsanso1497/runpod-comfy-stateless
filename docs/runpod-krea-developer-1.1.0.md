> Superseded where noted by `docs/krea-audit-1.1.1.md`. Apply the 1.1.1 ZIP, not the older ZIP.

# RunPod: Krea 2 HQ plus developer identity workflows

Use this only after applying update 1.1.0, pushing from GitHub Desktop, and
successfully running GitHub Actions > Build HQ workspace > krea. Rebuilding the
old commit would not include this update.

## Template fields

| Field | Value |
|---|---|
| Name | Krea 2 HQ + Developer Identity Edit |
| Container image | Actual new `ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:...` from the successful build summary |
| Alternative image tag | `ghcr.io/jsanso1497/runpod-comfy-stateless:krea-hq`, after the new build succeeds |
| Container start command | Leave blank |
| Container disk | 100 GB planning allocation |
| Persistent data | 500 GB network volume mounted at `/workspace` |
| Volume disk | Use the attached network volume in place of a separate local volume disk |
| Exposed HTTP ports | `8188` only |
| Exposed TCP ports | None required |
| GPU | One A100 80 GB, or another compatible 80 GB GPU |
| System RAM | At least 128 GB planning target |

Hardware/storage numbers are conservative planning choices, not a measured
minimum or a guarantee against GPU errors. The shared full-precision vision model,
Raw/Turbo bases, optional restoration, private checkpoints and outputs all need
persistent storage. Choose more capacity for a large private library.

Network volumes must be attached when creating the Pod, and their region affects
available GPUs. They persist independently of the Pod. Do not store the model
cache only on the container disk. Copy any existing nonpersistent models/outputs
to persistent storage before replacing or terminating their original Pod.

If the GHCR package is private, configure RunPod container registry credentials.
A public repository does not by itself guarantee that its image package is public.

## Environment variables

Enter each name and value as a RunPod template environment row. This block is
reference text, not a shell script.

```text
WB_WORKSPACE=krea
WB_TASKS=K01,KDEV
WB_TOOLBOX=sam,seedvr2
WB_DATA_ROOT=/workspace
WB_AUTH_MODE=none
WB_DISABLE_COMFY_COMPILER=1
WB_QWEN_PROMPT_MODEL=huihui_ai/Qwen3.8-abliterated:27b-bf16
WB_QWEN_PROMPT_FAILURE=original
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
```

This retains the requested password-free access. Do not add WB_PASSWORD. Anyone
with the Pod URL can access it; do not share that URL. Use RunPod's HTTPS proxy.
The internal ComfyUI and Ollama ports are not separately exposed.

Add these two rows ONLY if the corresponding private-library secrets already
exist and contain the desired lists:

```text
WB_LORA_URLS={{ RUNPOD_SECRET_comfy_loras }}
WB_CHECKPOINTS={{ RUNPOD_SECRET_comfy_checkpoints }}
```

Otherwise omit the unused row. An unresolved secret placeholder intentionally
stops startup. Keep actual token values and private links out of GitHub and the
public configurator. Private link lists can contain civitai.red, civitai.com or
Hugging Face provider links. Put checkpoints in comfy_checkpoints and LoRAs in
comfy_loras, one link per line. See identity-release-1.1.0.md for compatibility
rules and the explicit JSON format when metadata does not identify the base.

Ollama is already a default Krea model group. The selected Qwen helper runs before
Krea and unloads, rather than being used concurrently. KDEV never invokes it;
K01 retains its existing optional enhancement toggle. SAM and SeedVR2 are
available with WB_TOOLBOX above, but are not inserted into the developer graphs.
SAM provider access may need approval on its model page.

## Workflow locations and use

Open Workbench in ComfyUI:

- KDEV: four simple developer-derived Identity Edit recipes.
- K01: the existing multi-reference, masked photo and text-only Standard variants.

The normal ComfyUI workflow browser also receives factory copies under:

```text
Workbench Factory/0.1.0/KDEV
Workbench Factory/0.1.0/K01
```

Use the Workbench task browser for the current immutable graph if an older saved
factory file exists. Saved personal workflows are not overwritten. When startup
model preparation completes, refresh the model lists and select your own input
images. Example pictures are not included.

For KDEV scene/person edits, image 1 is the scene/base and image 2 is the person.
Start with 02 Turbo Scene Person for the basic workflow; use Raw for deletions or
when the model's developer requires real CFG guidance. Set ResolutionSelector to
the desired aspect ratio and keep output at or below 2 MP.

For a private Krea checkpoint: choose it in the model loader and use its matching
Raw/Turbo settings. K01 requires explicitly selecting Raw, Turbo or Custom for a
private checkpoint. For a private LoRA in KDEV: choose the file in an optional
LoRA node, set strength, and remove its Bypass. Leave the main Identity Edit
adapter active, and do not select it again as an optional LoRA.

If the masked K01 workflow only shows a colored mask preview, set run_edit ON in
the final Preview mask / Run edit / Verify / Save node after checking selection.
That preview is a deliberate pre-generation safety step.

## Build and recovery checks

The build summary provides an actual digest, not a guessed tag. A successful
software build is not a GPU image-quality test. Run a small single-image edit
first, then scene/person, then a private checkpoint/LoRA before larger jobs.

To restart just a crashed ComfyUI child while keeping Workbench running:

```bash
python /opt/workbench/scripts/restart_comfy.py
```

The command refuses to interrupt queued/running work. If the Workbench gateway
itself is down, restart the Pod/container. Do not start a second public server.

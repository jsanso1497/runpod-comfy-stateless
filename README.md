# ComfyUI Quality 3.2: one person, many reference photographs

Read [START_HERE.md](START_HERE.md) for the exact GitHub update and first RefMod test.

New workflows:
- `08_H3_Create_Person_RefMod`: one headshot plus up to eight optional body photos, saved as one ordered bundle of independent image references.
- `24_H3_One_Person_Multi_Ref_Quality`: original photos to H3's native vision encoder plus the corresponding RefMods, applied once; manual instruction by default.
- `25_H3_One_Person_Saved_RefMod_Quality`: load that one bundle, rebuild the one-person face/body prompt map and render.
- `09_H3_One_Person_Prompt_Draft`: optional abliterated vision prompting with all selected photographs.

No new model files or template environment-variable changes are required. Existing Krea-only Rebalance and two-person H3 graphs are preserved.

The implementation uses the existing pinned RefMod creator once per photograph, then its version-5 `save_bundle` function. This avoids the standard creator's common-canvas crop and temporal stacking when many photos are passed into a single extraction. Members remain separate `image` references in saved order. Original source-image shapes are preserved until each independent VAE preprocessing step.

Below is the preceding corrected-package reference documentation. Its existing Krea and two-person instructions still apply to those older graphs. Use START_HERE for the new ONE-person multi-image graphs.

---

# Stateless ComfyUI Quality 3.2

Read **START_HERE.md** for the exact GitHub and RunPod changes.
This is the complete corrected package, including the two previously unapplied
updates. Rebalance is wired only into Krea. H3 RefMod, the abliterated prompt helper,
LoRA support, SeedVR2 and the working HTTP patch remain included.

## Krea integration

The two Krea graphs now have the following path on BOTH sampler branches:

```text
Original reference image(s) + prompt
-> Krea2EditGroundedEncode
-> ConditioningKrea2Rebalance
-> KSampler
```

The original model-side path is unchanged:

```text
Raw BF16 model -> required full Identity Edit v1.2 -> optional LoRAs
-> Krea2EditModelPatch with original source image/VAE latent -> KSampler
```

This deliberately uses Rebalance-Pack's Krea-specific per-layer scaler, not its
alternative image-edit encoder. The adapter's trained chat template, image grounding,
image order, and dual reference mechanism are retained. Both output conditionings
have identical layer scaling, with an empty image-grounded negative for CFG 3.
The 8-substep Rebalance editing schedule is not added.

The starting per-layer weights come from the pinned pack:

```text
1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

Global multiplier is **1.0**, not the pack's 4.0 default. This is a deliberate
integration choice to avoid also multiplying all conditioning by four. It is NOT
an empirically optimized identity setting. No layer is discarded. The source
function copies conditioning metadata unchanged. Both source-grounding nodes and
the VAE/model reference patch stay connected.

Krea remains Raw BF16 / 20 steps / CFG 3 / Euler-simple / 1 MP. The earlier choice
of Raw is unchanged in this correction; Raw is not established as universally
better than Turbo. Review actual likeness and edit adherence on your reference.

## What was removed

Ideogram model entries, its profile, and its generated workflow were removed.
`runtime.json` lists the prior recipe filename as retired so it cannot become
an active shared graph after a folder merge. No H3 or SeedVR2 workflow contains
Rebalance nodes. The upstream Rebalance-Pack itself may still register additional
node types; registration does not download models or apply them to a workflow.

## Workflows

| Workflow | Use |
| --- | --- |
| `00_SeedVR2_4K_Video_Restore` | Existing restoration graph, unchanged |
| `01_SeedVR2_1080p_Video_Restore` | Existing 1080p restoration, unchanged |
| `05_Vision_Prompt_Draft` | Draft a reference-aware H3 prompt without rendering |
| `06_H3_Create_Full_RefMod` | Encode and save a full subject reference |
| `07_Backup_RefMods` | Export saved references before terminating the Pod |
| `10_Krea2_Same_Subject` | One subject reference, Identity Edit and Krea Rebalance |
| `11_Krea2_Subject_into_Scene` | Scene first, subject second, Identity Edit and Krea Rebalance |
| `20_H3_Reference_BF16` | Native H3 reference baseline |
| `21_H3_Ollama_Two_Refs_BF16` | Native H3 plus reviewed/generated prompting |
| `22_H3_RefMod_Ollama_Quality` | Main H3 Full Reference RefMod workflow |
| `23_H3_Saved_RefMods_Quality` | Reuse saved H3 reference files |

## Pinned application and preserved components

- Foundation image: the existing private GHCR digest beginning `56b51cf5`.
- ComfyUI: `65787d668397d230bf5839d69a0a7239e2dad378`.
- Krea2Edit: `86f886dac23013d88996e3a2e99093ba44d322fb`.
- MiniMaxH3Mod: `f9462081e28794389b5a6c5067eb327412ad8ee7`.
- Rebalance-Pack: `53c147c72c2fbd9444765af87118caee81a26d01`.
- SeedVR2: retained from the foundation.
- Ollama executable: 0.35.1 with the archive SHA-256 retained from the last package.
- Helper: `huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16`.

Weights download at Pod startup, not during Docker build. Application code and
node dependencies are prepared in the image. The existing Torch 2.9 / CUDA 12.9
constraint set, HTTP patch, API-key handling and Ollama sequential unloading stay
in place. No GPU inference or comparative identity test was executed here.

The full selected catalog's estimated weights total **178.1 GB**, plus roughly
67 GB for the unchanged helper, image/code layers and outputs. Keep the specified
500 GB workspace for this setup. GPU VRAM and system RAM are separate from disk;
the H3 full-BF16 memory caveat remains unchanged. The larger helper is unloaded
before H3 loads, but that does not establish that every H3 job fits every GPU.

## Add LoRAs

Retain the required Identity Edit adapter. Append compatible optional LoRAs to
`config/loras.json`; select and enable them separately in the workflow. These are
model-only LoRA slots. A download profile does not establish model compatibility.

The catalogs are JSON arrays. A protected Civitai file uses this format:

```json
{
  "name": "Your Krea 2 LoRA",
  "enabled": true,
  "required": false,
  "profile": "krea2",
  "source": "url",
  "url": "https://civitai.com/api/download/models/REPLACE_VERSION_ID",
  "sha256": "REPLACE_WITH_COMPLETE_FILE_SHA256",
  "destination": "models/loras/your_krea2_lora.safetensors",
  "estimated_bytes": 2000000000
}
```

Replace the URL, hash, name and byte-size estimate with the real file's values.
Do not append a key to the URL. Existing `CIVITAI_TOKEN` and `HF_TOKEN` RunPod
Secrets are used by the downloader. Hugging Face entries use `source=\"huggingface\"`,
`repo_id`, `filename` and `revision` in place of `url`. Missing hashes fail validation.

Commit bundled catalog or workflow changes, wait for the image build and deploy
a fresh Pod. The template image name remains `:latest`.

## RefMod use and saved results

`22_H3_RefMod_Ollama_Quality` keeps original image pixels for H3's vision encoder
and independently applies each Full Reference latent once. Do not reconnect its
native VAE reference socket. H3 stays full BF16, 25 steps, 1344x768 and 124 frames.

`23_H3_Saved_RefMods_Quality` attaches saved reference data during its text/vision
encode, so there is no second Apply. Saved references remain temporary unless
exported. Run `07_Backup_RefMods`, then download the ZIP before termination.

Use the same reference images in the same order when copying a reviewed prompt
from `05_Vision_Prompt_Draft` into workflow 22. One-click `Generate with Ollama`
continues directly into rendering; the text preview does not pause it.

## Validation and rights

See VALIDATION.md for the tests actually run. A successful CPU node-schema check
is not a successful GPU rendering or a likeness benchmark. Model licenses and
reference-image permissions are unchanged by this correction. Existing source and
licensing references are in SOURCES.md. Do not treat a private container registry
as authentication for the running ComfyUI service.

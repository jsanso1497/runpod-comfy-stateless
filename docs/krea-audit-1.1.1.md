# Krea workflow audit and cumulative update 1.1.1

## Verified repository state

Audited GitHub main: `0f4ba4ab841a9861ea76f9a8153298f4428cc7d3`.
Its parent is `136539008d2c23f6b6d4c7abb0068dc067e05107`. The newer commit adds
only the manual identity installer, its guide and its tests. It does not apply
consolidated update 1.1.0. GitHub Validate repository run 38507697867 succeeded
for 0f4ba4a. That result does not validate this newer patch or GPU generation.

This release supersedes the previous 1.1.0 download and includes its Qwen/Krea
developer tasks, Civitai.red private checkpoint support, managed restart and
stability mitigations. Apply it to a clean repository before the earlier
consolidated update. It deliberately refuses an existing consolidated-update
marker or uncommitted local changes rather than overwriting them.

The recent Research Local Image Models chat was used as context. Recommendations
were separately checked against the actual developer model card, node code,
changelog and Krea's native-generation code. This is not a claim that one set of
settings gives the best image for every prompt or checkpoint.

## Findings and changes

1. K01's single-image path ignored reference_strength and passed scene_strength
   to the upstream fidelity dial. The corrected code passes reference_strength
   in both one- and two-input modes. Scene strength applies only to the first
   image when a second reference exists. Default fidelity is 4. Reduce it when
   an intentional change, such as an expression or removal, is resisted.

2. K01 factory graphs defaulted to Raw, including ordinary identity editing and
   text-only generation. Fresh graphs now default to Turbo at the existing
   8-step / CFG-1 schedule. Developer KDEV Turbo presets retain their balanced
   10-step / CFG-1 schedule. Both are within the adapter developer's 8-12 range.
   Raw remains available and recommended for Identity Edit removals at 20/3.

3. The native Standard graph inherited the Identity Edit Raw removal schedule.
   It now uses a separate WBKreaStandardModels loader. Turbo remains 8/1.
   Native Raw uses 52 steps / ComfyUI CFG 4.5, mapped from the official example's
   52 steps / guidance 3.5. The official velocity formula is
   `cond + guidance * (cond - uncond)`, so conventional CFG is guidance + 1.
   This is a starting recipe, not a claim of identical cross-engine output.
   Native Raw is not the recommended general-generation choice; start near 1 MP
   when experimenting with it. Custom sampling remains available.

4. KDEV's original core LoadImage/SaveImage path was not the explicit ICC-aware
   sRGB path used by K01. The working KDEV copies now use the existing Workbench
   sRGB loader and PNG saver. The original developer JSON is archived unchanged.
   Untagged images are assumed to be sRGB, not guessed as another color space.
   Color management prevents profile mismatches, not diffusion-driven color drift.

5. The original KDEV graph encoded full-size sources through VAEEncode and then
   encoded them again through the pixel path at the target grid. The working
   copies omit the unused first encoding. Like existing K01, they supply the
   required placeholder source_latent from the target latent, while actual
   source images remain authoritative through vae + source_image. The upstream
   patch pre-encodes those images before sampling through target_latent.
   This removes redundant work; no GPU speedup or memory reduction was measured.

6. The older bundle added KDEV/QDEV but left an existing exact task-list test
   expecting only Q01/K01. That assertion is updated to include the two additions.
   K01 UI/API files are regenerated during safe local staging and checked into
   the user's working tree. CI checks that it is not silently repairing stale
   committed graphs. CI, image builds and Pages use the same generation steps.

7. Fresh Qwen/Krea factory copies use `Workbench Factory/Identity-1.1.1`, so an
   earlier copy-once factory folder cannot hide new defaults. Old factory files
   and personal workflows are retained, not rewritten.

## Preserved and verified by source/contract inspection

- Full BF16 base diffusion models, Qwen3-VL 4B BF16 encoder and matching Krea VAE.
- Full Identity Edit v1.2, strength 1, not a reduced-rank adapter stack.
- Upstream comfyui-krea2edit HEAD 86f886dac23013d88996e3a2e99093ba44d322fb,
  whose node package version is 1.2.5. Weight and node versions are different.
- FIT pixel geometry, ordered source/scene A and person/reference B.
- Positive and empty negative both grounded on the same ordered images.
- target_latent is the same target as the sampler.
- K01's 2 MP runtime bound and masked preview-first/source-preservation checks.
- KDEV defaults to 1 MP; keep manual ResolutionSelector choices at or below 2 MP.
  Its dimension rounding is now 16 pixels, matching Krea patchification.
- KDEV's optional LoRA slots remain bypassed and at zero strength by default.
- The existing shared prompt enhancer stays in K01, not inserted into KDEV.
- More than one optional K01 reference is a visible sheet in the second native
  image input, not independent conditioning for every reference.
- Native Standard is text-only generation and has no Identity Edit adapter.

## Recommended starting table

| Workflow | Checkpoint / steps / ComfyUI CFG | Notes |
|---|---|---|
| K01 full-frame Identity Edit | Turbo / 8 / 1 | Source plus optional references; full-frame pixels may change |
| K01 masked photo | Turbo / 8 / 1 | Raw / 20 / 3 for deletion tasks; preview then enable run_edit |
| K01 native Standard | Turbo / 8 / 1 | Text-only generation, not an identity-preserving editor |
| KDEV one-image Turbo | Turbo / 10 / 1 | Simple developer-derived edit |
| KDEV scene + person Turbo | Turbo / 10 / 1 | Scene first, identity second |
| KDEV one-image Raw | Raw / 20 / 3 | Removals or edits requiring real CFG |
| KDEV scene + person Raw | Raw / 20 / 3 | Reference edit with real CFG |

For the editing workflows use Identity Edit strength 1, fidelity around 4,
scene strength 1 and grounding 768 as the starting point. Grounding 512 may
help stubborn scene changes; 1024 can help likeness but can also worsen split
or duplicate compositions. These are controlled experiments, not automatic
upgrades. Do not increase sampling or resolution blindly to improve quality.

Native generation and the Identity Edit adapter have different operating ranges.
The editor's <=2 MP recommendation must not be confused with the native Turbo
model's larger text-to-image output capabilities.

The developer's model card and node README disagree about sequential versus
simultaneous insertion of distinct people. Neither is treated as an established
winner here. Multiple-person identity separation and full-frame locality remain
limitations. For original-pixel preservation, use the masked crop/stitch workflow.

## RunPod

The previous hardware/storage guidance is unchanged. Use the NEW successful
build's image digest, not an older krea-hq image cached before this patch.

```
WB_WORKSPACE=krea
WB_TASKS=K01,KDEV
WB_TOOLBOX=sam,seedvr2
WB_DATA_ROOT=/workspace
WB_DISABLE_COMFY_COMPILER=1
WB_AUTH_MODE=none
WB_QWEN_PROMPT_MODEL=huihui_ai/Qwen3.8-abliterated:27b-bf16
WB_QWEN_PROMPT_FAILURE=original
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
```

Optional only when those secrets exist:

```
WB_LORA_URLS={{ RUNPOD_SECRET_comfy_loras }}
WB_CHECKPOINTS={{ RUNPOD_SECRET_comfy_checkpoints }}
```

HTTP 8188, blank start command, persistent /workspace. WB_AUTH_MODE=none is
public access: anyone with the URL can access the Pod. Do not set WB_PASSWORD
for that mode. The password-protected mode remains available.

Krea-compatible Civitai.red checkpoints and LoRAs remain supported by the
cumulative update. A label cannot make SDXL/FLUX/Qwen weights Krea-compatible.
For custom checkpoints choose Raw/Turbo/Custom explicitly; do not guess from
an arbitrary filename. Download authorization and model architecture are separate.

The compiler-off setting and comfy-aimdo 0.5.6 upgrade are mitigations, not a
verified cure for the reported segmentation fault. 0.5.6 is published on PyPI;
no claim is made here that its release notes prove it fixes this particular crash.

## Validation and limits

74 offline tests passed; 2 full-repository tests skipped in this isolated bundle.
Tests cover migration/rollback guards, unchanged Git HEAD/index, real conditioning
code excerpts with mocks, native versus edit recipes, developer-link fixtures,
sRGB node selection, removal of redundant VAE work, LoRA bypass, source pins,
private-metadata rules and restart contracts.

Relevant source files, workflow generators and commit metadata were inspected
through GitHub reads. A complete repository clone was not available in this
network-restricted execution environment. The entire patched repository CI,
real upstream graph generation, Docker build, real private downloads, live node
imports and GPU inference have not been run here. The reusable GitHub validation
gate must pass before the updated image is published. Existing main's green CI
is not evidence that this patch has already passed that gate.

## Primary sources

- https://huggingface.co/conradlocke/krea2-identity-edit
- https://github.com/lbouaraba/comfyui-krea2edit
- https://github.com/lbouaraba/comfyui-krea2edit/blob/86f886dac23013d88996e3a2e99093ba44d322fb/__init__.py
- https://github.com/lbouaraba/comfyui-krea2edit/blob/main/CHANGELOG.md
- https://github.com/krea-ai/krea-2
- https://github.com/krea-ai/krea-2/blob/main/sampling.py
- https://huggingface.co/krea/Krea-2-Raw
- https://huggingface.co/krea/Krea-2-Turbo
- https://pypi.org/project/comfy-aimdo/0.5.6/
- https://github.com/jsanso1497/runpod-comfy-stateless/commit/0f4ba4ab841a9861ea76f9a8153298f4428cc7d3
- https://github.com/jsanso1497/runpod-comfy-stateless/actions/runs/38507697867

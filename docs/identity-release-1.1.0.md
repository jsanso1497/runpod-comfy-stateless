> Superseded where noted by `docs/krea-audit-1.1.1.md`. Apply the 1.1.1 ZIP, not the older ZIP.

# Identity developer workflows and stability update 1.1.0

This is an additive update for `jsanso1497/runpod-comfy-stateless`, audited against
main commit `136539008d2c23f6b6d4c7abb0068dc067e05107`. It does not replace the repository,
change existing K01 workflows, or push a commit. The local bundle applicator leaves
all changes uncommitted for review in GitHub Desktop.

## What the GitHub build now does

1. Fetch four small developer JSON sources at immutable revisions.
2. Archive those originals unchanged under `vendor/identity/`.
3. Generate four Krea and four Qwen BF16 adaptations under `workflows/developer/`.
4. Add KDEV and QDEV to the existing catalog, without replacing existing tasks.
5. Run repository validation and tests before publishing a workspace image.
6. Export the eight JSON files and source provenance as the
   `identity-developer-workflows` GitHub Actions artifact.

The Docker build includes the workflows and original source snapshots. Model
weights are never put in GitHub, the ZIP, a workflow artifact, or the container
image. On the Pod, Workbench prepares the selected standard and private models
using its existing checksum-verifying downloader. There is no separate identity
installer command to run on the rebuilt Pod.

The developer sources are fetched in GitHub Actions rather than included as
unverified reconstructed files in this ZIP. Network/provider failures stop the
build. Source adaptation, missing files, graph validation, and full repository
tests must pass before image publication. The generated sources are not claimed
to have run on a GPU just because CI passes.

## Krea task KDEV

The full `krea2_identity_edit_v1_2.safetensors` adapter is used with the existing
BF16 Raw and Turbo base models, BF16 Qwen3-VL 4B encoder and matching VAE. The
existing companion pack remains pinned to
`86f886dac23013d88996e3a2e99093ba44d322fb`. This includes the target-latent pre-encode
fix. Reduced-rank adapters and FP8 base/encoder selections are not substituted.

| Preset | Inputs | Starting sampler |
|---|---|---|
| 01 Turbo Single Image | One image to edit | 10 steps, CFG 1 |
| 02 Turbo Scene Person | Image 1 base scene, image 2 person | 10 steps, CFG 1 |
| 03 Raw Single Image | One image to edit | 20 steps, CFG 3 |
| 04 Raw Scene Person | Image 1 base scene, image 2 person | 20 steps, CFG 3 |

The adaptations use the developer's current recommended settings where the old
workflow note and current README differ. FIT, ref_boost 4, scene boost 1, a
768-pixel grounded encoder, an image-grounded empty negative, and the same target
latent connected to the patch and sampler are retained. The source graph starts
at 1 MP; choose the desired aspect ratio in ResolutionSelector. This graph does
not automatically copy the input image's aspect ratio. Keep generation at or
below the developer's 2 MP recommendation.

Two optional native LoRA loaders are added after the Identity Edit adapter. They
start BYPASSED at strength 0. To use one, choose a compatible private Krea LoRA,
set the desired strength, then remove Bypass. Do not apply Identity Edit again in
an optional slot. The main Identity Edit strength remains 1.0.

The KDEV recipes have no prompt enhancer, SAM, SeedVR2 or contact-sheet helper
inserted. They are BF16 adaptations of developer recipes, not byte-identical
originals and not pixel-preserving compositors. Original developer instructions
and attribution remain in the archived source files.

## Existing Krea task K01 is retained

All three current choices stay available in the same Krea workspace:

- Multi-reference Identity Edit with the shared Qwen vision prompt enhancer.
- Masked crop/stitch photo editing with manual or SAM source selection.
- Standard native text-only generation without the Identity Edit adapter.

The masked workflow deliberately previews the mask first. In
`Krea | Preview mask / Run edit / Verify / Save`, turn `run_edit` ON after checking
the mask to generate. Preview-only output is not a failed render. The safety gate,
sRGB image handling and exact-outside-mask checks are not removed by this update.
The Standard variant remains text-only generation, not reference-based editing.
Multiple references in K01 still use a visible reference sheet within the
upstream two-input editor; this is not native independent N-image conditioning.

## Qwen task QDEV

Four AusBoss developer adaptations are added: Consistency 1500, Consistency 2000,
LoRA-free Realign, and Outfit Swap. The three small LoRA weights are prepared when
QDEV is selected. Existing BF16 Qwen models and AusBoss 2.9.0 are reused.

1500 and 2000 are alternatives, not a stack. These are framing/consistency aids,
not dedicated face-swap models. The general adapter can resist requested pose
or head changes; use the LoRA-free recipe for those edits. Sampler settings,
reference routing and resize budgets remain those in the pinned developer
sources. Speed LoRA rows stay OFF and are not downloaded. No new prompt enhancer
is injected into these developer baselines.

To prepare them on a Qwen image, include QDEV in WB_TASKS, for example
`WB_TASKS=SQ,Q01,QDEV`. Rebuild `qwen` separately. Krea and Qwen still use their
respective workspace images and are not merged into one model stack.

## Civitai.red checkpoints and LoRAs

The existing .red host support, file/version selection, SHA256 verification,
HTTPS requirements and token isolation are retained. This update adds Krea family
inference and checkpoint registration. Civitai's term Checkpoint maps to the
`diffusion_models` folder for compatible Krea 2 diffusion-model fine-tunes; it does
not imply use of ComfyUI's all-in-one CheckpointLoaderSimple.

Keep LoRA links in the `comfy_loras` RunPod secret and checkpoint links in
`comfy_checkpoints`, one URL per line. Reference them through WB_LORA_URLS and
WB_CHECKPOINTS respectively. Use version-specific pages or exact Download links
when a version has multiple files. Leave API tokens out of URLs and out of GitHub.

A plain checkpoint link can register automatically when the provider explicitly
reports a recognized Krea 2 base. If metadata is inconclusive, use a JSON array
inside the checkpoint secret, for example:

```json
[
  {
    "url": "https://civitai.red/models/REPLACE_MODEL_ID?modelVersionId=REPLACE_VERSION_ID",
    "family": "krea",
    "architecture": "krea2",
    "kind": "diffusion_models",
    "name": "my-krea-finetune"
  }
]
```

Replace the example URL with the actual selected provider link. Declaring krea2
does not make an SDXL, FLUX or Qwen model compatible. Explicitly conflicting
provider metadata is rejected. Full-precision safetensors checkpoints are
supported; pickle .ckpt and quantized substitutes are not installed by this
workbench. A private model whose safetensors layout is unsupported by native
Krea loading will still need an appropriate conversion from its developer.

Once preparation finishes, refresh ComfyUI's model lists. In KDEV, choose the
checkpoint in UNETLoader and use the matching Raw or Turbo workflow. In K01,
choose it in `Krea Checkpoint + Correct Encoder` and explicitly choose Raw, Turbo
or Custom sampling. Auto sampling is intentionally not guessed for private
checkpoints. Optional LoRAs are selected separately and are never auto-applied.

A provider permission, login or access error cannot be fixed by relabeling a
model. Supply the correct CIVITAI_TOKEN and obtain access through the provider.
No model IDs, private links or token values are included in this patch.

## Today's errors: included fixes and limits

- Preserve or repair the shared Qwen/Krea prompt-enhancer regression test that
  previously failed by assuming only Qwen uses the helper.
- Replace the audited ComfyUI requirement `comfy-aimdo==0.5.5` with `==0.5.6`
  before installing requirements for Qwen/Krea. Add an explicit install
  constraint so the older exact requirement cannot reinstall 0.5.5.
- Default `WB_DISABLE_COMFY_COMPILER=1` in Qwen/Krea. This is a conservative crash
  mitigation, not a GPU-verified resolution of the reported A100 segfault.
  Full-precision weights remain unchanged. Set 0 to opt back into the compiler
  only after testing the new image.
- Add one-instance guards and a managed restart endpoint/command. It uses the
  actual `/opt/ComfyUI` installation and internal port 8189. It refuses a restart
  while downloads or queued/running generations are active, and never starts a
  second public server on 8188.
- Inspect embedded subgraph node types in task readiness checks rather than
  falsely reporting the subgraph's UUID as a missing custom-node package.
- Copy the new build and recovery scripts through Docker's explicit allowlist.
- Gate image publication on the repository's validation/test workflow.

Recover a crashed ComfyUI child from the rebuilt Pod's web terminal:

```bash
python /opt/workbench/scripts/restart_comfy.py
```

If the whole Workbench gateway is down, restart the Pod/container. Do not run a
second `python main.py --port 8188` process. Idle and crash recovery are tested
with process/queue doubles here; a real GPU hang remains a live-system test.

## Local apply, backup and GitHub build

Use the outer ZIP's START_HERE.md and apply_update.py. The applicator checks the
repo origin and a clean working tree, applies only known source anchors, generates
the merged public catalog in a temporary copy, and saves a backup ZIP in Downloads
before applying anything. Unexpected source changes stop the operation. It does
not pull, stage, commit or push. Review/commit/push in GitHub Desktop.

Then run Actions > Build HQ workspace with workspace `krea`. The new build calls
Validate repository first and will not publish if that job fails. After success,
copy the actual image digest from the Action summary. The mutable `krea-hq` tag
also points to the new image, but a digest avoids ambiguity. Run the same action
for `qwen` when ready for QDEV. Other workspace images do not need rebuilding.

For rollback before committing, use GitHub Desktop's discard/revert controls and
remove only the newly added paths listed in the backup's BACKUP_INFO.json. The
backup contains the original bytes of changed files. Keep it until the new image
has been validated. Never delete the repository or its .git directory.

## Validation boundaries

The bundled test report states the exact offline tests executed in this session.
They cover migration anchors, staging/backup behavior, Krea graph wiring fixtures,
exclusive LoRA selection, metadata rules, checksums, path safety and managed
restart contracts. They do not test images generated by the models.

The complete repository was inspected through GitHub reads, not cloned and run
locally here. Full repository CI, pinned developer-source downloads, the Docker
build, live Civitai transfers, custom-node imports and GPU image quality are gates
for the user-triggered build and subsequent Pod test. The package explicitly
marks these unverified rather than claiming production certification.

## Source references

- Krea Identity Edit developer: https://huggingface.co/conradlocke/krea2-identity-edit
- Companion nodes and workflows: https://github.com/lbouaraba/comfyui-krea2edit
- Qwen Consistency developer: https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA
- Outfit Swap developer: https://huggingface.co/ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA
- AusBoss nodes: https://github.com/ausboss/ComfyUI-AusBoss
- Offloader release: https://pypi.org/project/comfy-aimdo/0.5.6/
- Persistent storage: https://docs.runpod.io/pods/storage/types

Source/model licenses still apply. This update does not grant new model or
likeness rights, and it does not remove provider access restrictions.

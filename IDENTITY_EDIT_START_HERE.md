# Qwen 2.1 and Krea 2 identity-edit developer presets

Research checked October 13, 2026. This is an optional add-on for the existing
Qwen and Krea workspaces. It does not replace any saved workflows or change K01,
SAM, Rebalance, private LoRAs, prompt enhancement, gateway ports or default tasks.

## Apply this add-on first

The GitHub connection rejected the push with HTTP 403 (Resource not accessible
by integration). This package has NOT been pushed or installed on your Pod.

Extract the add-on ZIP and upload its contents to the root of the existing
jsanso1497/runpod-comfy-stateless repository, preserving the scripts/ and tests/
folders. Do not upload the ZIP itself as a source file. Do not delete or replace
other repository files. This adds three files; it changes no existing files.

The commands below work AFTER those files are uploaded. No Docker rebuild is
needed for this manual setup path on a compatible current Pod.

## Install on a running Workbench Pod

Use Python 3.11 or newer and the matching workspace. For Krea:

```bash
set -e
curl -fL --retry 3 https://raw.githubusercontent.com/jsanso1497/runpod-comfy-stateless/main/scripts/setup_identity_edits.py -o /workspace/setup_identity_edits.py
python /workspace/setup_identity_edits.py krea
```

For Qwen, use the same download command on the Qwen Pod, then:

```bash
python /workspace/setup_identity_edits.py qwen
```

The normal workspace base models must already be prepared. No base model or
lower-precision substitute is downloaded by this installer. It verifies and
reuses existing LoRA files, so Krea's full v1.2 file is not downloaded again when
its hash matches. New LoRAs go into the active ComfyUI models/loras directory,
which Workbench links to /workspace/models/loras.

Open ComfyUI's normal **Workflows** browser, then:

```text
Identity Developer Presets / krea / 1.0.0
Identity Developer Presets / qwen / 1.0.0
```

These are not new task IDs in the Workbench sidebar. No image rebuild is needed
for this command on a compatible current Pod. It does not start or restart a
server. When it installs a missing node pack, restart the existing service and
refresh the browser. Never launch another server on port 8188.

For non-Workbench installations use --comfyui and --workflow-dir. --dry-run
prints the pinned plan without changing files. --require-live additionally
requires every graph node type to be available at Workbench's internal port
8189. Node-type checking is not a full schema validation or inference test.

## Krea: one current full-rank LoRA, four presets

Selected: conradlocke/krea2-identity-edit, **krea2_identity_edit_v1_2.safetensors**,
about 1.83 GB. The developer still recommends v1.2. The model page showed 827 likes
and the upstream node repository 701 stars when checked. These are popularity
signals, not a measured quality ranking; Hugging Face does not track downloads
for this model. No newer developer-recommended full release was found.

The full model is used, not r128 or r64 reductions. Do not stack v1, v1.1 and v1.2.
Raw and Turbo are different base checkpoints using the SAME Identity Edit LoRA.
Your repository already included this LoRA and the current upstream node pin.

| Preset | Defaults | Input roles |
|---|---|---|
| 01 Krea Turbo Single Image | 10 steps, CFG 1 | One source photo |
| 02 Krea Turbo Scene Person | 10 steps, CFG 1 | Image 1 scene/base; image 2 identity |
| 03 Krea Raw Single Image | 20 steps, CFG 3 | One source; preferred starting point for removals |
| 04 Krea Raw Scene Person | 20 steps, CFG 3 | Scene/base plus identity reference |

All four retain the developer's reference boost 4, scene boost 1, FIT geometry,
768 grounding baseline, Euler/simple sampling, and 1 MP default canvas. Change
the aspect ratio and resolution in ResolutionSelector. Stay at or below 2 MP.
For likeness, 1024 grounding is a developer-suggested experiment, not a universal
improvement; reduce it if compositions duplicate. The existing custom K01
settings are untouched.

**Required pack:** lbouaraba/comfyui-krea2edit, pinned at
86f886dac23013d88996e3a2e99093ba44d322fb (current source; changelog v1.2.5).
The two essential nodes are Krea2EditModelPatch and Krea2EditGroundedEncode.
Both are required. A stock CLIPTextEncode does not supply the training-matched
image-grounded conditioning.

The pixel path, VAE and source_image connections are retained. target_latent is
connected to the exact same latent output as the sampler, preventing the known
mid-sampling VAE encode/offload problem. Both positive and empty negative are
image-grounded, including both references, which matters for Raw at CFG 3.

The current GitHub repository ships ONE unified workflow, despite the model
card mentioning two. The installer archives that exact JSON and creates four
separate BF16-adapted copies. The old embedded note suggests 40 Raw steps; the
Raw presets follow the current README's 20-step/CFG-3 recommendation and label
that change. These copies are not claimed to be unmodified official workflows.

The developer's model card and node README disagree about whether sequential or
simultaneous insertion is better for two distinct people. Neither is presented
here as established best practice. These scene/person presets use the documented
ordered two-input path; they do not promise reliable multi-person face separation.

## Qwen: consistency aids, not a dedicated face-swap model

| Preset | LoRA |
|---|---|
| 01 Qwen Consistency 1500 | qwen-image-2.1-consistency.safetensors, strength 1 |
| 02 Qwen Consistency 2000 | qwen-image-2.1-consistency-2000.safetensors, strength 1 |
| 03 Qwen Realign No LoRA | None; experimental post-edit alignment |
| 04 Qwen Outfit Swap | qwen-image-2.1-outfit-swap.safetensors, step 1000, strength 1 |

Approximately 477 MB for the three additional LoRAs. The 1500/2000 consistency
files are alternatives, not a stack. 1500 is the recommended starting point;
2000 holds framing more tightly but can make stylized edits paler. General
consistency can suppress intentional pose/head changes. Try the LoRA-free
realignment recipe for those, while recognizing alignment is not pixel-perfect.
The outfit adapter preserves the source person while changing clothing; it is
not a general identity transfer adapter.

The companion pack is ComfyUI-AusBoss at the existing Workbench 2.9.0 pin.
The outfit developer workflow requires ComfyUI 0.38+ and AusBoss 2.5.1+.
Qwen's BF16 diffusion, BF16 Qwen3-VL 8B encoder and BF16 VAE are reused. All
embedded subgraphs are traversed; LoRA positional/named controls stay in sync.
The original sampler settings, resize budgets, links, alpha handling and prompt
behavior are retained. Optional speed LoRAs are disabled, not downloaded.

## What is preserved and verified

Original upstream JSON is archived separately under:

```text
/workspace/workbench/identity-developer-sources/<family>/1.0.0/originals/
```

The installation report records URLs, hashes, generated files and any missing
live node types. Every LoRA is SHA256-verified before use. Upstream models and
workflows are pinned to immutable source revisions. Node packs are reused using
Git or the Workbench build-provenance file; a differing installed revision is
left untouched rather than overwritten. New node packs are installed only when
absent. Edited preset files get separate factory copies, never overwritten.

The installer uses /opt/ComfyUI, not the old /workspace/ComfyUI assumption.
For normal Workbench, workflows live under /workspace/comfy-user/default/workflows.
Existing workflow/task defaults, optional Rebalance/SAM/SeedVR2 tools and private
libraries are not changed. These baseline recipes do not insert a prompt enhancer.

## Validation status

17 offline tests passed locally: preset selection, Krea target-latent routing,
Qwen nested graphs and exclusive LoRA selection, BF16 substitution, grounded
negative preservation, download integrity, no-overwrite behavior, dry runs,
workspace separation and Docker node provenance without .git.

No live RunPod was accessed. Original JSON download/adaptation, live ComfyUI node
schemas and GPU inference have not been tested end-to-end here. The installer
checks expected upstream layouts and stops rather than silently changing an
unexpected graph. Model files were not downloaded into this editing environment.

```bash
python -m pytest -q tests/test_identity_setup.py
python scripts/setup_identity_edits.py krea --dry-run
python scripts/setup_identity_edits.py qwen --dry-run
```

## Primary sources

- https://huggingface.co/conradlocke/krea2-identity-edit
- https://github.com/lbouaraba/comfyui-krea2edit
- https://github.com/lbouaraba/comfyui-krea2edit/tree/86f886dac23013d88996e3a2e99093ba44d322fb/workflows
- https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA
- https://huggingface.co/ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA
- https://github.com/ausboss/ComfyUI-AusBoss

Model licenses remain those of their publishers. Node code licensing is separate.

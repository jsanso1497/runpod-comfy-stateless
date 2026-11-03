# H3 BF16 Economy

Release 2026-11-02.2. Applies to the reviewed repository commit
`6ac458aa6196de20ea68df421b1956d3acb33db9`. No remote repository writes are performed.

## Separate, lean image

Build **Build H3 BF16 economy**, then use its successful image digest. The tag is
`ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap-economy`.
It is not published until your build succeeds. Existing baseline and full-BF16
image tags are retained as comparison options.

Economy uses `minimax_h3_ref2va_pruned_bf16.safetensors` (40,225,724,176 bytes), the
original BF16 H3 Qwen reference encoder (51,506,295,256 bytes), PyTorch attention,
and the existing Akatz V1.5 / matched FL2V Turbo8 recipe. Both adapters stay at
1.0, eight steps, er_sde, normal scheduler and CFG 1. Original VAEs and SAM remain.
The source checkpoints remain BF16; this is not a conversion of INT8 weights.

The pruned diffusion checkpoint is locked to the original Comfy-Org model file:
- revision `3792b245ac4685dda00ebf43a33c646f7ce98064`
- SHA256 `37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe`

Pruned and full models are not claimed to give identical pixels at the same seed.
Compare your best known full-BF16 result before accepting this profile. No GPU
render, peak-memory or speed benchmark was performed while authoring this package.

There is NO Ollama binary, service or optional helper-model download in the new
image. Analysis nodes are not exposed and analyzer/BFS factory workflows are
omitted. Legacy analyzer source files remain only for other profiles sharing this
source tree. The mandatory H3 reference/text encoder is still required; it is not
the optional prompt writer. Existing Ollama weights on a reused volume are not
silently deleted. No SCAIL, face-refinement model, 1080p finishing or new LLM is added.

## First launch and files

Expose HTTP **8188 only**. Leave the container start command and entrypoint override
blank. Keep persistent storage mounted at `/workspace` and use the existing
`workbench` / `comfy_password` gateway login.

Open the SAME HTTP-8188 HTTPS address with **`/files/`** added. This is the new
**H3 Files** page, not the classic FileBrowser application or its old RunPod 8080
card. It is served directly by the authenticated gateway: no second daemon, port,
state database, startup script or model readiness requirement.

The gateway binds before model preparation, GPU preflight and ComfyUI startup.
You can browse, upload and download while the model weights are being prepared.
The page cannot appear before RunPod has pulled/started the container, mounted its
volume, and supplied valid authentication configuration. No promise of availability
during image pulls or a stopped Pod is made.

Features: folder listing, parent navigation, downloads, streaming uploads, new
folders and copyable file paths. It deliberately does not offer shell commands,
public sharing, deletion or overwriting existing files. Hidden files and symlinks
are not exposed. Downloads use attachment disposition. Uploads pass through the
same password and same-origin checks and the configured upload-size ceiling.
The old `.h3-filebrowser` database is neither used nor modified.

If you see RunPod's separate File Browser card still 'initializing', stop using
that obsolete 8080 link. Open **HTTP 8188**, then **`/files/`**. The startup status
page is `/_swap/status`. The page reports the file backend as `builtin`.

## First native quality test

Wait for `ready: true` and `model_profile: bf16_economy` at `/_swap/status` before
generating. Refresh ComfyUI, then open:

`H3_Character_Replacement_2026-11-02.2_ECONOMY`

Start with **02B_Akatz_Native_Simple_Preview**, the same source section, character
reference, SAM selection and seed as your successful full-BF16 render. Keep the
768 preset and reference size `match`. The native test is approximately 5.17 s.
The 480-preset variant **02D** is retained, but does not have to be the default.

Use the included simple prompt, without running any analysis:

```text
Replace the selected person with the reference character. Preserve source performance, scene and audio timing.
```

The tiny live decoder gives an approximate early preview. Judge actual face, skin
and wardrobe using the fully decoded output. A successful precision/header check
is not proof of visual quality or adequate memory for a long clip.

## Automatic independent chunks

Open **15_H3_Economy_Auto_Independent**. Upload/select the original source and one
original reference. Start with:

| Control | First test |
|---|---|
| action | Start new run |
| run_name | economy_test10 |
| duration_seconds | 10 |
| start_seconds | 0 |
| resolution | 768 |
| window_frames | 124 |
| overlap_frames | 12 |
| review_policy | Approve first chunk |
| cleanup_policy | Stage offload |
| prompt | Included simple prompt |

Set the target person's original appearance and mask-selection settings. Queue
ONCE, then open **`/_swap/economy`** on the same 8188 service. The launcher returns
while the external controller queues the short generation jobs.

The first complete candidate pauses without another generation running. Watch it
and choose **Approve and continue** or **Retry candidate**. After approval, the
remaining clips render and join automatically. Review every chunk and Automatic
modes are also available. Automatic mode is not identity or seam verification.
No preview hold is imposed inside the automatic worker; review happens between
jobs. A paused GPU Pod still incurs rental charges, so do not leave it waiting.

Every clip receives the same original reference and its own original source
interval. There is NO generated-history input. The 12-frame overlap is duplicate
source footage for editing, not previous-generated-frame conditioning. The final
join trims that duplicate interval and retains the source soundtrack. A silent
source produces a silent output. Short final tails are retained.

Use a NEW run name to change source/reference/prompt/mask/settings or precision.
Resume preserves accepted clips and verifies their hashes. A failed pending job
is not silently resampled; inspect the error and Resume explicitly. A source split
interrupted before its manifest completes needs a new run name, rather than an
unsafe overwrite. Do not queue unrelated work while the controller is running.
The old disk-chain controller and the economy controller share an exclusive lock.

### Memory policy

Stage offload keeps the existing Relay encode/sampling stage-release helpers and
Comfy's normal caching. It avoids the old controller's duplicate full-cache clear
and cleanup jobs before each successive render. Unchanged preprocessing/encoding
can be reused on retries when Comfy still has the matching cache entry. This is
not an additional persistent feature-cache system or a measured speedup.

Full cleanup between chunks is the more conservative fallback if a host shows
memory growth. Start a new run with that setting. Neither mode guarantees a fit
on every GPU. Reduce to 56-frame windows or the 480 preset only as an explicit
tradeoff after reviewing the failure and quality. Do not enable GPU-only/high-VRAM
residency for this offloading-oriented profile.

The existing **10_Akatz_Simple_Single_Reference_Chain** is still available for
history-based continuation and uses `/_swap/chain`, not the economy review page.
Manual split/render/join/recovery workflows 11, 12/12B, 13 and 14 are retained.

### Output and review

Batch records: `/workspace/output/h3_character_swap/economy/<run_name>/`.
Individual native outputs: `/workspace/output/h3_character_swap/independent/`.
Final assembly: `/workspace/output/h3_character_swap/joined/`.

Each completed clip appears as a normal generated output in ComfyUI. The final
assembly also appears as a separate completed job. Final output has both a lossless
master and a viewing MP4. No color correction, face blending, interpolation or
crossfade is silently applied. Independent chunks can differ at their joins.

## Right-click a generation: Open workflow

This update addresses the job record, not just embedded file tags. The pinned
frontend looks up the output's job ID and extracts:

`workflow.extra_data.extra_pnginfo.workflow`

Every economy render submission supplies the actual editable generation canvas,
including exact source clip, staged original reference, prompt, seed, selection,
models and sampler settings. In **ComfyUI History / generated outputs**, right-click
the ACTUAL completed clip and select **Open workflow**. It should open that native
clip-generation canvas, not just the batch launcher. The final assembled video's
Open workflow opens its assembly canvas, listing the actual selected masters.
To inspect the generation itself, open the corresponding chunk's output.

Completed H3 job records are archived under `/workspace/h3-swap/job-history/`.
The backend restores the 300 most recent valid records (bounded by 128 MiB) into
normal Comfy history on startup; older archived jobs can be retrieved individually
by their original job ID through the same native API. Nothing is automatically
queued or regenerated. Native outputs that carry a real canvas benefit as well.
Sidecars and embedded metadata remain as backup, not as the primary user procedure.

The pre-restart `h3-swap/history-backups/comfy-history-*.json` files are imported
when they contain genuine editable canvases. Files with only missing metadata or
API graphs are not converted into an invented editable workflow. Old videos with
no surviving records cannot reveal the original settings. The manual recovery
workflow 14 remains available for explicit recovery from sidecars/media records.

The existing NaN exporter fix is retained: only runtime cache markers are removed
from the saved copy, never from live execution. The economy runtime enforces
metadata saving, even if an old `H3_SWAP_SAVE_METADATA=0` workaround remains set.
Saved prompts and filenames may contain private information; review before sharing.

## Hardware recommendation and qualification

First test: one A100 SXM **80 GB**, not 40 GB. The currently listed Secure Cloud
allocation is 125 GB host RAM / 16 vCPUs at $1.79/hour before storage and taxes.
This is a planning candidate, not a measured minimum or cheapest-per-result claim.
The pruned diffusion plus mandatory BF16 encoder still total about 91.73 GB on
disk, before VAEs and adapters. They are used in stages, not held on GPU together.
The 48 GB RTX 6000 Ada remains a future test, not this package's guaranteed target.

Use 80 GB container disk and 300 GB network storage for a CLEAN dedicated setup,
or keep 500 GB+ if retaining the older models and many lossless outputs. Check free
space rather than deleting old weights automatically. Mount the network volume at
`/workspace`; do not terminate the prior Pod until all unique files are secured.

## Validation limits

The accompanying report records tests actually run in the authoring environment.
Real FFmpeg tests use synthetic passthrough frames, NOT H3 inference. The new files
page is tested over real local HTTP while model-ready is false. Queue/history tests
exercise the exact expected shapes with a test queue. The Docker action additionally
runs a REAL Comfy CPU job, checks its image metadata and `/api/jobs/<id>` canvas,
restarts Comfy, then checks that same API response without requeueing anything.

The full Docker build, real Comfy CPU restart gate, browser context-menu click,
GPU generation, peak memory and visual quality were NOT executed during authoring.
They must pass at build/deployment and the first short render before acceptance.
No seamless-motion, identity-locking, identical-to-full-BF16 or speed guarantee is made.

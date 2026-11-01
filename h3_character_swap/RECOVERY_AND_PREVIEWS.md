# Workflow recovery, native previews, and pending integration fixes

Release **2026-11-01.1**, based on repository commit
`0a0f8b0fc29e88a37ef027e4474dc16cf1a6b743`.

## What to open

After a successful new **Build H3 character replacement** run, deploy the NEW
image digest. A refreshed browser or imported JSON does not install new Python
nodes or the tiny preview decoder into an old Pod. Keep your persistent media
volume at `/workspace`; do not terminate a Pod with the only copy of its files.

Factory folder: `H3_Character_Replacement_2026-11-01.1`.

- **02B_Akatz_Native_Simple_Preview**: native developer loop and sampling, one
  reference, short default prompt, sampling preview, decoded preview, and saved
  workflow metadata. Default duration remains approximately 5.17 seconds.
- **10_Akatz_Simple_Single_Reference_Chain**: the previously supplied simple
  single-reference disk-chain preset is now in the build. Ten-second test,
  124-frame windows, Review every chunk, no automatic seed retries. It still uses
  the disk controller's source preparation and continuation; this is NOT a claim
  of pixel-identical native/chain generation or a skin-color correction.
- Existing 00/01/02/04/06/09 and other personal workflows remain available.
  Two-image prompting is optional, not injected into 02B or 10.

The short prompt in 02B and 10 is exactly:

```
Replace the selected person with the reference character. Preserve source performance, scene and audio timing.
```

No Ollama call is required for either workflow. Your selected helper model is
not changed, aliased, downloaded anew under a different name, or forced back to
8B by this update. Keep the H3_SWAP_PROMPT_MODEL value you actually intend to use.

## Native preview and cancellation

There are two DIFFERENT previews:

1. **During sampling:** the normal native `SamplerCustomAdvanced` uses the
   pinned ComfyUI TAEH3 preview path, enabled at startup. Its node is labeled
   `LIVE SAMPLING PREVIEW: approximate first frame, not final color`.
   The tiny decoder is a separate, pinned 22,709,752-byte preview-only asset.
   It does not replace your full H3 VAE, adapters or sampler. No custom sampler
   monkeypatch is installed.
2. **After the current window is decoded:** `H3 Decoded Window Preview + Cancel
   Hold` displays the actual first, middle and last decoded frames. It passes
   the SAME original frame tensor to native assembly. The default hold is ten
   seconds after the first window only. Set hold_seconds to 0 to remove this
   delay, or disable hold_first_window_only to pause after every window.

Use ComfyUI's **Cancel/Interrupt** control while sampling or during the hold.
There is no automatic identity detector and no claim that the reference is
correct just because a preview appears. This does not interrupt model-loading,
reference-encoding or an individual GPU kernel instantaneously.

The tiny preview is approximate. Early denoising frames can look wrong and
skin colors are NOT a reliable final-color reference. The core preview uses the
first latent frame; on a continuation that can be the already accepted boundary,
not a newly generated frame. Use the decoded first/middle/last preview for the
actual result. The decoded preview cannot save the compute already spent on
that window, but can prevent subsequent windows in a longer native run.

Keep long work in the disk chain: the native loop still prepares the selected
source in memory. This update does not remove that native memory limitation.
Start with the same short source, one reference, mask and seed as the native
result you liked. Sampler, CFG, steps, LoRA strengths, VAE, resolution and native
history are unchanged. Live browser placement, GPU cost and preview likeness
still require a short real-Pod test.

## Recovering future generations

Metadata is ON by default. Native ComfyUI output nodes can again save standard
`workflow` and `prompt` metadata. The new 02B save node also writes adjacent:

```
<video>.mp4
<video>.mp4.workflow.json
<video>.mp4.api.json
```

Drag the MP4 or its `.workflow.json` into ComfyUI to reopen it. Video editors and
platforms may strip embedded metadata; the JSON sidecar remains the fallback.
Metadata contains prompts, settings and filenames, NOT source/reference image
bytes, model weights, RunPod passwords or environment dumps. A workflow with
user-entered secrets in its own nodes/notes would still contain those values:
inspect metadata before sharing the video or JSON.

Every submitted H3 graph is additionally journaled BEFORE generation under:

```
/workspace/output/h3_character_swap/workflow_history/<UTC_timestamp>_<id>/
  submitted.workflow.json     # when the client sent an editable canvas
  submitted.api.json         # execution graph
  receipt.json               # submitted, not proof of a successful render
```

This includes canceled/failed submissions. A caller sending only an API graph
has no original UI layout to recover. The journal does not invent that layout.
The observer does not modify the queued graph or launch another job.

Open **`/_swap/recovery`** on the SAME authenticated HTTP-8188 service to download
recent records. The page lists up to 100 submissions; older files remain on disk.

## Disk-chain recovery

Each new run now saves:

```
/workspace/output/h3_character_swap/disk_chain_v2/<run_name>/
  repeat.workflow.json       # same settings, a new run name
  resume.workflow.json       # original run name, action = Resume
  repeat.api.json
  resume.api.json
  recovery.manifest.json
  final.mp4                  # appears only after accepted chunks are assembled
  final.mp4.workflow.json
  final.mp4.api.json
```

`final.mp4` embeds the repeat-controller workflow and its corresponding public
API prompt during the existing final encode. There is no extra re-encode.
Importing it restores the user-facing CHAIN CONTROLLER, not a pretend one-pass
native graph. Original exact internal job graphs remain in each attempt's
`job.api.json`. Worker history includes the resume-controller canvas too.

- Use **repeat** to rerender with a new run name. Confirm the current files are
  still the intended inputs. The workflow does not contain their bytes.
- Use **resume** to continue an interrupted or waiting run, keeping its accepted
  chunks. Loading it does not run anything until you queue it. A completed run
  must be repeated, not resumed.
- Checkpoints, input hashes, prompts, original seeds, 18-frame history and
  existing resume checks are preserved. Original config/state/masters are not
  rewritten by a workflow-export request.

## Older runs

On `/_swap/chain`, click **Export workflow files** for an older run that still
has its `config.json`. It writes repeat/resume sidecars, including face/body
roles where recorded. It does not regenerate a frame or re-encode the old MP4.
Download one of the new workflow links and import that JSON.

It cannot recover the exact layout/settings of an old native generation from a
video that never contained them. A manually saved canvas, surviving ComfyUI
history or another source of recorded settings is necessary in that case.
Do not queue an old internal `job.api.json` into its original attempt folder:
that is a diagnostic graph and could target existing output/checkpoint paths.

## Automatic FileBrowser is included now

This consolidates the previously delivered but uncommitted FileBrowser add-on.
It starts automatically, uses the current gateway login (`workbench` and your
`WB_PASSWORD`), and opens at **`/files/`** on HTTP **8188**. Its backend listens
only on 127.0.0.1:8081. Expose **8188 only**, not 8080/8081/8189/11434.
Remove the old public 8080 card from the new Pod/template after switching.
Leave start-command/entrypoint overrides empty. Existing manual FileBrowser
files and databases are not deleted or imported.

The original FileBrowser is archived, with no further security fixes. Its
command runner is disabled and the authenticated gateway protects every request;
that does NOT turn it into maintained software or an OS sandbox. It retains the
container's existing user to manage root-created files. Use this on your own
single-owner Pod, not as a multi-user security boundary. Set
H3_SWAP_FILE_BROWSER=0 to disable it. See FILE_BROWSER.md for the full restrictions.

## Environment and fallbacks

Keep the existing image secrets, data paths, Akatz selection and chosen analysis
model. These are the additional settings; all three have these defaults:

```
H3_SWAP_SAVE_METADATA=1
H3_SWAP_PREVIEW_METHOD=taesd
H3_SWAP_FILE_BROWSER=1
```

- `H3_SWAP_SAVE_METADATA=0`: disable standard embedded metadata and new recovery
  exports/journaling. Internal chain config/checkpoint records still exist.
- `H3_SWAP_PREVIEW_METHOD=latent2rgb`: cheaper, very rough preview without tiny
  decoding. `none` disables sampling previews. Neither setting removes the
  actual decoded-window preview node.
- A missing/conflicting tiny decoder is reported under `preview` at
  `/_swap/status` and falls back to rough latent2rgb; no conflicting file is
  overwritten. Build-time checks still require the pinned asset to verify.
- Token context, model choice and H3 sampling are not changed. Long-analysis
  logging now counts answer characters, reasoning-channel characters and stream
  messages separately. It never stores or displays reasoning text. Zero answer
  characters alone is not described as proof of a stalled model.

FileBrowser status: `/_swap/status` under `filebrowser`.
FileBrowser log: `/workspace/h3-swap/logs/filebrowser.log`.

## Scope and test boundaries

This is a focused H3 integration update. Qwen, SAM Local Edit, old h3-hq images,
GPU selection, model/sampler versions and 1080p finishing are not changed. It
adds no unverified color-correction step or new image-identity algorithm.

Offline tests use CPU tensors, mocked ComfyUI/Ollama processes, actual local HTTP
for authentication/routing and real FFmpeg files for metadata round trips and
chain media tests. The complete new Docker build, downloaded FileBrowser binary,
actual TAEH3 model decode, browser import and GPU rendering cannot be run in the
authoring environment. GitHub build gates include real FileBrowser startup/auth,
ComfyUI CPU node imports, and a real tiny-decoder CPU decode before publication.

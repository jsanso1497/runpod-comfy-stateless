# Independent H3 chunks, previews and joining

Release 2026-11-02.1. Applies to BOTH the `baseline` quantized image and the
`bf16_full` image. The model profile is not the output resolution: both profiles
have 768 and 480 pixel-budget choices. Existing models, adapters, sampler,
eight-step schedule and reference-size defaults are unchanged.

## Before restarting the current Pod: preserve older history

Older native files may have no embedded workflow. BEFORE changing/restarting the
current Pod, run this in its RunPod terminal to save up to the latest 300 execution
records to persistent storage:

```bash
(
  set -e
  mkdir -p /workspace/h3-swap/history-backups
  FILE="/workspace/h3-swap/history-backups/comfy-history-$(date +%Y%m%d-%H%M%S).json"
  curl --fail --silent --show-error --max-time 60 \
    'http://127.0.0.1:8189/history?max_items=300' -o "$FILE.partial"
  mv "$FILE.partial" "$FILE"
  echo "Saved history: $FILE"
)
```

This does not restart ComfyUI, change the queue or recover records already lost.
Retain the same /workspace storage after deploying. Workflow 14 checks these saved
backups as well as live history. It checks up to 20 recent JSON backups, each up
to 64 MiB. Malformed or oversized backups are not treated as proof of a workflow.
The backup includes prompts/filenames and should remain private.

## Build and deploy

Apply the downloaded package locally, review/commit in GitHub Desktop and manually
Push origin. Run the action for each image you intend to use:

| Profile | GitHub action | Container tag after successful publication |
| --- | --- | --- |
| Baseline INT8/NVFP4 | Build H3 character replacement | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap` |
| Original BF16 | Build H3 BF16 high precision | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap-bf16` |

Use each successful action's immutable digest. Rebuild BOTH images to update both;
rebuilding the BF16 tag does not update the baseline tag. Importing a new JSON into
an old image does not install the new nodes. No new generation weights are added.
Keep `/workspace` and existing secrets, image-specific precision and analysis
model settings. Do not terminate a Pod before preserving its files.

After deploying, wait for `ready: true` at `/_swap/status`. New factory folders:

- `H3_Character_Replacement_2026-11-02.1` for baseline.
- `H3_Character_Replacement_2026-11-02.1_BF16` for BF16.

Do not resume an existing chained run after changing its precision profile.
The manual independent path uses new outputs, not existing chain checkpoints.

## Preview workflows in BOTH images

- `02B_Akatz_Native_Simple_Preview`: existing native recipe, 768 preset.
- `02D_Akatz_Native_Simple_Preview_480`: explicit lower-resolution preset.
- `12_Akatz_Independent_Chunk_Preview`: one independent short clip, 768 preset.
- `12B_Akatz_Independent_Chunk_Preview_480`: the same independent workflow at 480.

The baseline image already had source support for 02B; this release makes the
lower-resolution choice explicit and documents that the baseline must be rebuilt.
The 480 preset changes pixels, not model weight precision.

Live sampling preview uses the existing TAEH3 preview decoder. It is approximate,
not a faithful skin-color or identity judge. The full-decoded pane shows first,
middle and last frames and a ten-second interruptible hold. The final saver also
shows a playable MP4. Use Cancel/Interrupt during a bad render or the hold. A
post-decode preview cannot refund the compute already spent on that clip.

Keep these values in the actual Pod configuration after deploying the fix:

```text
H3_SWAP_SAVE_METADATA=1
H3_SWAP_PREVIEW_METHOD=taesd
H3_SWAP_FILE_BROWSER=1
```

The existing `/_swap/chain`, `/_swap/recovery` and `/files/` pages remain on the
same authenticated HTTP 8188 service. There is no additional public port.

## Independent clip procedure

This is the alternative to propagating generated history across a long take.
It may help diagnose accumulated appearance drift; it does not promise identical
faces between independent clips or seamless motion at their joins.

### 11: Prepare independent source chunks

Open `11_H3_Prepare_Independent_Chunks`. Select your ORIGINAL full source video,
not a previous generated pass unless you deliberately intend a second edit pass.
For a first 48-second shot, use:

```text
set_name: shot48_independent
duration_seconds: 48
start_seconds: 0
window_frames: 124
overlap_frames: 12
```

This creates eleven clips for exactly 48 seconds at 24 fps. Each full clip is
124 frames (5.1667 seconds), with twelve source frames shared for editing between
neighbors. The final clip has 32 frames. No tail is silently skipped. Set overlap
to zero for simple non-overlapping chunks; a 48-second source then makes ten clips.
The clip count is calculated from the actual conformed source interval. Duration
0 uses the remaining source, subject to available disk space.

Output files are under:

```text
/workspace/input/h3_independent/shot48_independent/
```

`chunks.json` records the exact frame ranges, original-source hash and audio
origin; `ORDER.txt` lists the source clips. The `_source_24fps.mkv` is a full
conformed source/audio copy, not a short clip to select for generation. The source
files themselves are not changed. Lossless intermediate files can use substantial
disk space. No GPU sampling takes place in workflow 11.

HDR is rejected rather than silently tone-mapped. Source rotation and pixel
aspect ratio are conformed. This operation resamples timing to 24 fps but does
not add frame interpolation. The native generation preparation still chooses
its normal output canvas. Keep at least the free disk allowance in your current
template; actual needs depend on footage, source resolution and rejected takes.

### 12: Render each clip independently

Refresh ComfyUI file choices after splitting. Open workflow 12 (or 12B for 480),
select `clip_0001.mkv`, then select the SAME original reference image that worked
best in native. Keep the successful seed, mask selection, target-person description
and expansion. With several people, explicitly select the original performer.
The short default prompt is deliberately unchanged:

```text
Replace the selected person with the reference character. Preserve source performance, scene and audio timing.
```

Queue once. This workflow has NO StartLoop/EndLoop and no generated-history input.
It uses one native sampling window, not the disk chain controller. The source
loader refuses clips longer than 124 frames, preventing accidental whole-shot
preparation. Short final sections receive only native model padding, then finish
at their actual input frame count. Do not change the preparation duration or wire
in the preceding output.

Inspect the decoded preview and final video. For a rejected take, rerun that clip
with your chosen seed. Accepted and rejected attempts have separate folders;
this workflow does not silently pick the latest attempt as the best one.

The saver writes:

```text
/workspace/output/h3_character_swap/independent/<take_name>/<clip_and_take_id>/
    master.mkv
    preview.mp4
    master.mkv.workflow.json
    preview.mp4.workflow.json
    receipt.json
```

The master is 16-bit RGB FFV1 with the short source audio. It is lossless relative
to the saved 16-bit samples, not the original float tensor. Use it for the join;
`preview.mp4` is the smaller lossy viewing copy. The saver displays the exact
`output/.../master.mkv` filename for easy copying. Workflow/API sidecars appear
when metadata is enabled. The receipt is retained even when metadata is disabled.

Choose the next SOURCE clip and repeat with the original reference. No image from
the prior generated clip enters the next render. Masks are freshly tracked within
each independent section, so verify the selected person after difficult occlusions.

### 13: Join chosen videos

Open `13_H3_Join_Rendered_Chunks`. Paste one chosen master filename per line in
chronological order. Pick only ONE take for each source clip:

```text
output/h3_character_swap/independent/shot48_take1/clip_0001_<takeid>/master.mkv
output/h3_character_swap/independent/shot48_take1/clip_0002_<takeid>/master.mkv
output/h3_character_swap/independent/shot48_take1/clip_0003_<takeid>/master.mkv
```

Those are illustrative names; copy the actual filenames from workflow 12 or
FileBrowser. Use `overlap_frames=12` only for clips prepared with that overlap.
For existing non-overlapping outputs, use zero. A join cannot infer whether two
independently supplied files depict duplicate source moments.

Keep `cut_placement=Middle of overlap` initially. It keeps half the duplicate
interval from the earlier take and half from the later take. `Keep earlier take`
and `Keep later take` move the hard cut to the corresponding overlap edge.
There is no dissolve, generated transition, optical flow or face morphing. The
join preserves source timing by removing duplicate frames, not playing them twice.

Choose `audio_mode=Original source`, select the original FULL source file and use
the same `source_start_seconds` used in workflow 11. This restores a continuous
soundtrack rather than concatenating independently encoded short audio files.
Choose `No audio` for silent sources. A subset beginning later in a source set
requires its actual original timestamp; the receipt validator will report it.

All clips must have the same dimensions and 24 fps. With new independent-render
receipts, the join checks source identity, order, exact frame ranges, overlap,
reference hashes and precision profile. It rejects missing/reordered source
sections and mixed tracked/untracked takes rather than guessing. With older MP4s
without receipts, order and timing are user-declared. The original files are never
modified or deleted. There is no automatic resize, creative color correction or
frame-rate conversion during joining.

Outputs:

```text
/workspace/output/h3_character_swap/joined/<join_name>/final.mkv
/workspace/output/h3_character_swap/joined/<join_name>/final.mp4
/workspace/output/h3_character_swap/joined/<join_name>/join.json
```

The MKV is a lossless assembly master; the MP4 is H.264 CRF16 by default with
explicit SDR color tags, encoded once from the assembly master. New workflow/API
metadata and sidecars describe the JOIN operation. Each original chunk retains
its own generation workflow. Reopening a joined video does not pretend that its
join graph generated the component character renders.

Review the actual joins. If faces differ, select another take or rerender the
problematic section. Correct frame counts do not guarantee an invisible cut.

## Missing workflows and the NaN fix

The exporter now omits only execution-time `is_changed` cache fingerprints from
its SAVED copy. The runtime cache markers and actual node inputs are not changed.
Strict JSON still rejects a genuinely nonfinite user setting or invalid pixels.
This fixes the verified NaN exporter conflict without changing generation.

Metadata is enabled by default. The current standard native saver, native preview
saver, independent saver and joiner record reopenable workflow data and adjacent
JSONs. A pre-execution journal also preserves submitted native canvases even if
you cancel them later. Use `/_swap/recovery` to access it.

Open `14_H3_Recover_Past_Generation_Workflow` to recover existing records:

1. Existing adjacent `.workflow.json`.
2. For old disk-chain runs, saved `config.json` reconstructs repeat/resume controls.
3. Embedded workflow data in MP4/MKV, including case-insensitive metadata keys.
4. Matching still-available ComfyUI execution history, or a pre-restart backup above, for that exact output file.

Only an API prompt may survive in some records; the tool reports that the original
canvas layout is missing instead of mislabeling it as an editable workflow.
Native history is normally transient across restarts. Files created without any
saved metadata, sidecar, run config or surviving history cannot recover an exact
workflow from pixels alone. Rebuilding will not invent those missing records.
Editors and social uploads can strip metadata; keep the JSON sidecars.
Metadata contains prompts and filenames, so review it before publicly sharing.

## Validation boundaries

Offline checks exercise graphs, both precision transforms, paths, overlap policies,
frame alignment, original audio, cancellation, metadata round trips and recovery.
Media tests use real FFmpeg with synthetic/pass-through frames, not H3 inference.
No complete Docker build, live ComfyUI browser session, TAEH3 GPU preview, BF16
memory benchmark or identity/seam-quality evaluation was run in authoring.
The existing GitHub build-time CPU/schema and decoder checks remain enabled.

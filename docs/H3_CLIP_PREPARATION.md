# H3 long-video clip preparation

Release: 2026-10-22.2. CPU media-tested; live ComfyUI and rebuilt Docker image not tested here.

## What it does

**HCLIPS** adds two source-preparation workflows to the existing H3 workspace:

- `01_H3_Extract_Selected_124_Frame_Clip.json`: export one selected section or one numbered chunk.
- `02_H3_Split_Long_Video_Into_124_Frame_Clips.json`: export a consecutive range or every complete chunk.

Three-minute and longer inputs are supported. This is file-backed FFmpeg processing, not loading the entire video into an IMAGE tensor. Only one short interval is decoded/encoded at a time, using two encoder threads. The source file is never changed. This workflow calls no H3, pose-estimation, Ollama, or upscaling model. It adds no model weights. Normal H3 workspace startup can still prepare models for your other selected tasks.

This prepares input videos. It does NOT render every chunk through H3, join generated clips, or guarantee visual continuity between separate H3 generations.

## Exact timing

Every completed MP4 is checked with ffprobe and must have exactly **124 frames at 24 fps**, or it is not published. The exact duration is **31/6 seconds = 5.1666666667 seconds**. "5.17" is only a display approximation.

All chunk boundaries are computed with rational arithmetic, not by repeatedly adding 5.17. With source-grid origin S, chunk N starts at `S + (N - 1) * 124/24`. There is no accumulated rounding drift in requested source time. Frame selection is necessarily sampled from the input's frame timestamps when converting from another frame rate. No optical-flow interpolation or speed change is introduced.

A 180-second source at start zero yields **34 complete clips** and a **4.333333-second remainder**. The default skips that remainder. Choosing **Hold last frame to 124** exports a 35th clip with an explicit **0.833333-second final-frame hold**. The report and CSV identify padded clips. It does not invent additional movement; do not treat a padded tail as a genuine full performance.

## Controls

| Control | Meaning |
|---|---|
| `mode` | Selected chunk, or Consecutive chunks |
| `start_seconds` | Source-time origin of the chunk grid, relative to any input trim |
| `first_chunk` | 1-based chunk number within that grid |
| `chunk_count` | Consecutive mode: 0 = all remaining; otherwise export up to this many |
| `final_chunk` | Skip incomplete tail, or explicitly hold its last picture to fill 124 frames |
| `audio` | Keep source audio, or Silent |
| `make_loadable` | Make each export available in the normal Load Video dropdown |
| `make_zip` | Create a ZIP for runs containing two or more clips |
| `crf` | H.264 quality; default 16, not lossless |

Examples:

- A section starting 1 minute 30 seconds in: Selected chunk; start_seconds=90; first_chunk=1.
- Exactly chunk 7 from the start: Selected chunk; start_seconds=0; first_chunk=7. This begins at 31 seconds.
- Ten chunks beginning with chunk 7: Consecutive chunks; start_seconds=0; first_chunk=7; chunk_count=10.
- All complete chunks: Consecutive chunks; start_seconds=0; first_chunk=1; chunk_count=0.

There is a safety limit of 1000 exports per run. A longer source can still be used: request a finite count, then continue from a later first_chunk. Source positions and durations are limited to 30 days by validation. This is a safety bound, not a tested practical source-length guarantee.

## How to use the result in the motion-control workflow

1. Open Workbench > HCLIPS, upload your longer video, set the controls, and Run.
2. Inspect the exports/report. Only the first four MP4 previews are shown on the export node to avoid overwhelming the browser; all outputs are saved.
3. Refresh ComfyUI to refresh Load Video's file choices.
4. Open HMOTION > Motion Preview and select the desired `h3clip_...mp4` file.
5. Set HMOTION source `start_seconds=0`. Leave `window_seconds=5.25`; the loader will use the actual 124-frame file length.
6. Check the pose map, then use that same prepared clip and start/window settings in the character-swap render graph.

The default `make_loadable=true` publishes unique top-level names in `/workspace/input`. This matches native Load Video's top-level file picker. Files are hardlinked to outputs on the same filesystem, or copied when input/output are on different filesystems. No symlink is used, and an existing input is never overwritten. Copies use extra storage. Hardlinked files remain until both directory entries are removed.

No manual download and upload is required between preparation and HMOTION. Saved source chunks are NOT automatically queued for generation.

## Output files and downloading

Outputs are under:

`/workspace/output/h3_clips/<prefix_timestamp_uniqueid>/`

Each run contains numbered MP4s, `clip_manifest.csv`, and `run_status.json`. A run of two or more clips also includes a ZIP by default. MP4 files, ZIPs, and CSV manifests are available through Workbench > Files after the runtime update.

The CSV records the source start/end time, exact rational start, frame count, rate, padding flag, and reusable input filename. Source audio is re-encoded to AAC at 48 kHz when present, or explicitly omitted by Silent. The video duration is exact; AAC/container duration fields can differ by a small rounding/codec-padding amount.

Each run uses a new unique directory. Cancellation or failure stops the current FFmpeg process, removes incomplete temporary files, and retains completed clips plus an incomplete status report. At least 1 GiB is reserved on output storage. Creating a ZIP needs approximately another copy of the completed MP4 bytes; the tool does not delete valid clips when ZIP storage is insufficient.

## Image quality, geometry, and supported sources

The default export is H.264 CRF 16, medium preset, 8-bit YUV420 SDR, at 24 fps. This is a high-quality re-encode, not an original-bitstream/lossless cut. Direct stream copy is deliberately not used because arbitrary cut positions must not snap to source keyframes. CRF 0 is available, but only makes encoding lossless relative to the normalized 8-bit YUV420 frames, not relative to all original formats or higher-bit-depth footage.

Original display resolution is retained, with rotation baked in, square pixels, and dimensions rounded to even integers when necessary. The tool does not upscale to claim additional detail. Anamorphic pixels can require a wider square-pixel canvas. HDR/BT.2020 inputs are rejected with an explicit request to convert to SDR; they are not silently rendered with incorrect colors. H3's own later generation resolution is a separate setting.

Only a local, file-backed VIDEO within ComfyUI input storage is accepted. Network URLs, in-memory image batches, symlinked paths, and directory traversal are rejected. Normal MP4/MOV sources with valid durations are expected. A corrupt or unusual file can require remuxing first.

## Long-video upload allowance

The pinned ComfyUI CLI defaults to a 100 MiB upload allowance. This package adds `--max-upload-size 1024` for the H3 workspace, leaving other workspace launch settings unchanged. The Workbench gateway remains capped at 2 GiB per request.

Optional environment override:

`WB_H3_MAX_UPLOAD_MB=1024`

Allowed values are 1 through 1900 MiB. This only sets the application-side limit; a reverse proxy or browser may impose a smaller independent limit. The clip engine itself does not depend on a browser upload and can read a compatible source already placed in `/workspace/input`.

## Installation and scope

Use the combined **H3_Motion_and_Clip_Prep_Update_2026-10-22.zip**. It includes the previous motion-control update unchanged plus HCLIPS. It can be installed over the existing H3 video/stills build, with or without the previous motion patch already applied.

Run the included macOS Ruby installer from the repository root. It performs no network calls, commits, or pushes. Backups live in the extracted package, not in the repository. Existing conflicting files stop installation before changes. Review and manually push using GitHub Desktop, then rebuild the h3 image.

Suggested selection:

```text
WB_WORKSPACE=h3
WB_TASKS=HCLIPS,HMOTION,HSTILL,HFL,HFL4K
WB_TOOLBOX=standard
WB_DISABLE_COMFY_COMPILER=1
```

No additional environment variable is required for clip preparation. Retain authentication, HF token, port 8188 and the /workspace mount. The combined package's motion task still requires its separately documented control/pose weights; clip preparation itself adds none.

## Validation boundaries

See the clip-preparation CPU test log, installer log, and combined validation report in `reports/`. Tests include a real 180-second 60 fps source, exact output-frame counting, later-source extraction, audio, silence, tail padding, variable frame rates, rotation, anamorphic pixels, graph/API consistency, and installation over the previous motion patch. No user's private video was used for tests.

These tests do not certify the actual ComfyUI browser session, Docker build, the full repository CI suite, GPU rendering, or H3 motion/identity quality.

## Sources inspected

- FFmpeg accurate seeking and frame-count options: https://ffmpeg.org/ffmpeg-all.html
- FPS, trim, tpad and audio resampling filters: https://ffmpeg.org/ffmpeg-filters.html
- Why stream-copy segment cuts depend on keyframes: https://ffmpeg.org/ffmpeg-formats.html
- Pinned native LoadVideo and disk-backed VIDEO behavior: https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy_extras/nodes_video.py
- Pinned upload flag: https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/cli_args.py

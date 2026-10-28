# H3 low-memory chain and continuity review

For the optional TWO-reference variant added in 2026-10-27.2, use workflow 09 and
[FACE_BODY_CHAIN.md](FACE_BODY_CHAIN.md). This document describes the original
single-reference workflow 06 and the shared disk/history/review engine.

Release: **2026-10-26.2**. This is an integration update for the existing
`h3-character-swap` image, not a new model or an H3 quality certification.

## What to open

After rebuilding **Build H3 character replacement**, use its new image digest
on your Pod. Keep the existing persistent `/workspace` volume and environment.
In ComfyUI's Workflows sidebar, open:

`H3_Character_Replacement_2026-10-26.2/06_H3_LowMemory_Chain_Control`

Upload the source video and ONE character image into `/workspace/input` using
FileBrowser or an existing ComfyUI upload node, then refresh the file choices.
The complete intended face/body/outfit in one image is the simplest starting
reference. The sheet helper is a layout utility, not a proven identity solution.

Choose a unique run name, such as `kitchen_test_01`. Start with **19 seconds** and
**Review every chunk**. Queue the control workflow once. It returns quickly
because a separate process on the Pod handles the short ComfyUI jobs.

Open **`/_swap/chain` on the SAME authenticated HTTP-8188 service** to see progress,
previews, the mask, seam reports and Resume/Approve/Retry/Stop buttons. The main
landing page at `/_swap` also links to it. No additional port is required.
Keep the Pod running. The browser or web terminal can close after launch.
Do not submit other generation work while the chain is active.

## Defaults and controls

| Field | Default | Meaning |
|---|---|---|
| action | Start new run | Use Status/Resume for an existing name. |
| run_name | shot_01 | Unique name for a new shot; existing names are not overwritten. |
| duration_seconds | 19 | Delivered length, NOT generation time. Try 120 after reviewing a short run. |
| start_seconds | 0 | Starting timestamp in the original source. |
| resolution | 768 | Same Relay pixel-budget preset as before. 480 is an explicit quality/memory tradeoff. |
| window_frames | 124 | About 5.17 seconds for the first job. 56 is a smaller, approximately 2.33-second fallback. |
| target_person | person | Describe the correct performer; copy your previously working selection. |
| selection/index | largest / 0 | Same SAM selection policies as the original workflow. |
| mask_expansion | 64 | Same outward expansion; raw masks, not expanded masks, are carried between jobs. |
| seed | 904239 | Preserved for first attempts. Only rejected-section retries use alternate seeds. |
| review_mode | Auto check + retry | Heuristic checks and bounded retries. Review every chunk pauses after each candidate. |
| max_retries | 2 | Up to two additional attempts for flagged joins, then pause. |
| prompt | Original character-replacement instructions | Keep it general enough for the entire selected shot. |

Changing controls does not silently change an existing run. Resume uses that
run's recorded settings. To change the reference, seed, duration, resolution,
prompt or window size, create a new run name.

## The memory fix

1. FFmpeg normalizes the selected source to a 24 fps file on disk. The controller
   never decodes the entire long source into a Python/ComfyUI image batch.
2. A fresh ComfyUI prompt sees only one short source interval. At the default
   size, a continuation input contains at most **141 source frames**: the 18
   previous context frames plus up to 123 new frames. This is about 5.88 seconds.
3. The existing Akatz model, SAM tracking, native H3 conditioning and sampler
   generate that short section. No growing full-shot image accumulator survives
   across prompts.
4. The new output node saves a 16-bit RGB FFV1 master and a float32 history
   checkpoint. The controller asks ComfyUI to unload models and clear execution
   caches, then passes two lightweight queue fences before starting another job.
5. Completed masters remain on disk. Final assembly reads the masters as a
   stream and encodes the final H.264 video once, with the continuous original
   soundtrack. No full two-minute IMAGE tensor is assembled.

The model weights and one generation window still have to fit. Memory clearing
is not a guarantee that the allocator returns every byte to the OS. If an
individual 124-frame job fails, use a NEW run with 56 frames, or explicitly choose
480. The memory log records before/after system stats, not a measured peak.
Model reloads trade speed for lower residency. Disk use increases: source
normalization, full-precision history, 16-bit masters and rejected attempts are
retained. For a two-minute 768-preset trial, budget roughly **50 GB free** beyond
existing weights as a planning allowance, not a measured or guaranteed maximum.
Very noisy footage and multiple retries may need more. Free-space guards stop
before continuing media writes when the remaining space is too small.

## What actually crosses each boundary

The pinned Relay implementation uses **17 preceding generated frames plus one
boundary frame**, for 18 frames total. It re-encodes those pixels as history and
a first-position guide. This package preserves that method, rather than
substituting an incompatible 22/39-frame latent-prefix continuation algorithm.

The checkpoint now preserves float32 pixel values exactly; it does not pass
history through an 8-bit PNG or the final H.264 output. The next source section
begins at the matching original-video frame index. Extraction seeks to an exact
whole second, then trims by frame count to avoid millisecond-container timestamp
rounding at fractional-frame seeks. Its tracker is initialized
with the corresponding saved **raw** person mask. Expansion is applied once in
the new job, not repeatedly accumulated.

Relay still regenerates a shared boundary frame. Assembly keeps the previously
accepted portion, discards the duplicate boundary, and appends only the new
frames. A 120-second target is 2,880 delivered frames: 24 jobs with the final
one trimmed to the exact remainder. The first full job delivers 124 frames;
subsequent full jobs deliver 123 new frames.

This is **not** a hard pixel/latent lock on future H3 output. Saved history can
be exact while the next novel frame changes brightness, identity, speed or pose.
Preserving one matching frame does not establish matching motion velocity.
Earlier statements implying that a guide alone guarantees a seamless join were
too strong.

## Seam checks, retries and review

The controller examines small subject-region thumbnails on both sides of the
REAL delivered join and compares them with the same source interval. It looks
for conspicuous appearance/brightness jumps, a possibly frozen continuation,
coarse block-motion discontinuity, a missing subject mask and a possible source
cut. The coarse motion calculation is NOT a trained optical-flow model.

These are uncalibrated heuristic starting thresholds. False positives and false
negatives are possible. A `pass` does not certify facial identity, body shape,
anatomy, lip sync, exact choreography or a visually seamless join.

In **Auto check + retry**, a flagged join retries only the new section with a
recorded alternative seed. Accepted prior sections are never regenerated. If
all allowed attempts remain suspicious, the chain stops at `needs_review`.
Possible source cuts pause immediately rather than spending retries on an
intentional edit. Source-cut detection is also heuristic: use one continuous
shot and split real shot changes yourself.

In **Review every chunk**, EVERY candidate waits for approval, including the
first section. This is the recommended first-use mode. The review page shows:

- `preview.mp4`: the candidate section.
- `join_review.mp4`: about one second either side of the join, repeated three
  times at normal speed. No crossfade or interpolation conceals the transition.
- `first_frame.png` and `source_mask.png`: initial output and raw source mask.
- Numerical checks and their limitations.

**Approve candidate and continue** accepts that exact saved candidate, including
any visible defect you have chosen to accept. **Retry candidate** preserves the
accepted prefix and samples a new attempt. There is no infinite automatic retry
loop. **Stop** requests cancellation of this controller's own prompt, not other
users' jobs. After an interruption, choose **Resume** with the SAME run name.

## Outputs

`/workspace/output/h3_character_swap/disk_chain_v2/<run_name>/`

Each `part_0001/attempt_00/` includes:

- `master.mkv`: lossless 16-bit RGB master, relative to its encoded 16-bit values.
- `history.npz`: the last up to 18 generated frames and corresponding raw masks
  as float32 arrays, loaded with pickle disabled and checksum/size validation.
- `qc.npz`, `quality.json`, `receipt.json`, `memory_cleanup.json` and `job.api.json`.
- Preview MP4 and review images. Failed/rejected attempts are retained separately.

The run root contains `config.json`, `state.json`, logs, and **`final.mp4`** only
after every section is accepted and final frame-count checks pass.
Full float history is not copied into the final output; only its small tail is
loaded for the next job. The final H.264 MP4 is lossy. It must not be described
as pixel-identical to the float checkpoint or FFV1 master.

Resume verifies inputs, settings, accepted receipts and checkpoint hashes, plus
build identity and actual weight-file size/mtime receipts. It refuses changed
inputs/models rather than mixing versions. The weight receipts are not a new
adversarial model-integrity guarantee; normal image startup still verifies the
published model hashes. Existing v1 runner runs are NOT silently migrated to v2.

## Other included utilities

- `05_Face_And_Body_Style_Sheet.json`: the previously supplied two-photo sheet.
  It arranges photos; it does not synthesize one combined full-body character.
- `07_Akatz_LowMemory_Manual_Chunk.api.json`: prior no-runner manual fallback.
  It has no cross-job history or automatic joining.
- `08_H3_Prepare_5s_Clips.json`: the existing verified clip splitter now wired
  into the isolated character-swap image. It only prepares source clips.

The automatic chain does NOT require pre-splitting with 08. Feed it the full
source and choose a duration. The splitter is a separate utility for manual jobs.

## Scope and deployment

No model versions, weights, attention backend, sampler, eight-step recipe,
Ollama settings, public ports, authentication or BFS flags are changed. Qwen,
SAM Local Edit, the original h3-hq workspace and their image builds are untouched.
No automatic FileBrowser installation/startup is added by this update.

The new control/checkpoint nodes DO require rebuilding the image. Dropping only
its JSON onto the old Pod will show missing nodes. Existing images and personal
saved workflows remain available; the new factory folder is versioned separately.

Local verification covers media mechanics, state transitions, graph connections,
security and installation. **Docker build, interactive ComfyUI import, real H3
GPU execution, peak GPU/RAM use and visual seam quality have not been tested in
the authoring environment.** GitHub's existing built-image CPU startup check is
retained and expanded to require the internal checkpoint nodes.

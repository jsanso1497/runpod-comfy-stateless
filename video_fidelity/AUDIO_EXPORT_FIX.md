# Faithful Video: preserve audio when the frame sequence ends

Reviewed repository base: `b6a9d3d0b3d3ef890c28e9661b345bd85b2bfb52`.
Input evidence: `logs_113185325775.zip` supplied in this conversation.

## Observed failure

The in-container preflight correctly selected `-vsync 0` on
`ffmpeg version 4.4.2-0ubuntu0.22.04.1`. The integer-rate media case passed.
The next case, a 30000/1001 FPS clip trimmed from 0.2 seconds for 0.5 seconds,
failed with `RuntimeError: Audio was lost during encoding.`

This is separate from the earlier unsupported `-fps_mode` failure. That
compatibility fix is retained. The failed Docker build did not reach publishing.

## Mechanism and narrow correction

In the FFmpeg 4.4.2 source, `need_output()` closes every stream in an output
file when one stream reaches its `max_frames`. The current final export sets
`-frames:v` while also encoding audio, so a short PNG stream can end all output
before audio has been fully processed. This mechanism matches the supplied
failure; an actual 4.4.2 reproduction was not run in this environment.

Reference source:
https://github.com/FFmpeg/FFmpeg/blob/n4.4.2/fftools/ffmpeg.c#L3608-L3629

The patch:

- Removes the video-frame cap from the combined video/audio export only.
- Validates every expected numbered PNG header, dimensions, index and total
  count first. The finite sequence then ends naturally at EOF.
- Uses the installed FFmpeg's timestamp-preserving mode in final encoding too.
- Requires the audio mapping when the plan expects audio. It never inserts
  silence, marks required audio optional, or disables missing-audio checks.
- Checks for decodable audio frames after preparation and final encoding.
- Retains exact output frame count, frame rate, geometry, duration, overwrite
  protection and partial-output behavior.
- Expands the build preflight to decode test audio and check sample duration,
  audible tone and start offset, including the original failing case.
- Prints failed-case FFmpeg logs and media probes before temporary files are
  removed, so a remaining issue is diagnosable from the Actions log.

No second image encode, generative pass or additional model is introduced.
Model files, inference settings, strength, resolution choices, temporal window,
color matching, H.264 slow/CRF 14, AAC 256 kb/s, Dockerfile, image tag, RunPod
settings and existing workflow JSON are unchanged.

## Applying locally

Download `fix_faithful_video_audio.py` to Downloads and run from your local
`runpod-comfy-stateless` repository root:

```sh
python3 -c "from pathlib import Path; import runpy; runpy.run_path(str(Path.home() / 'Downloads' / 'fix_faithful_video_audio.py'), run_name='__main__')"
```

The installer validates reviewed old-file hashes and its embedded payload,
refuses conflicting edits, and backs up originals under
`.git/faithful-video-fix-backups/audio-export-*`. It does not fetch, stage,
commit, push, trigger Actions or deploy. Review/commit/push in GitHub Desktop,
then start a new **Build Faithful Video Minimal** run on main. Re-running the
old failed run would use its old commit.

## Verification boundaries

Local tests used FFmpeg 7.1.5. They cover real integer/fractional frame-rate
exports, single-frame output, trimmed tone audio, intentionally short audio,
video-only input, missing/empty audio refusal, finite sequence checks and the
existing timestamp, streaming, graph and authentication contracts. Legacy
`-vsync 0` cases are explicitly tests of those flags on the installed binary,
not a claim that FFmpeg 4.4 was run. Test fixtures permit up to two AAC frames
of decoded end padding; there is no claim of sample-identical AAC output.

The full Docker image, its FFmpeg 4.4 executable, and GPU restoration were not
run here. The updated preflight remains inside the actual image build, before
large dependency installation. The existing real ComfyUI CPU execution gate
also remains. Publish/deploy only after those checks pass.

# Faithful Video: FFmpeg compatibility fix

## Confirmed failure

The uploaded `logs_113179768876.zip` is from Build Faithful Video Minimal run
41767293920, commit `95f922fe718193418c4c864027fce68a065c8e73`.
The Docker build completed, but the actual ComfyUI CPU video test failed:

```text
Unrecognized option 'fps_mode'.
Error splitting the argument list: Option not found
```

The image installed Ubuntu's FFmpeg 4.4.2. The preparation function used the
newer `-fps_mode passthrough` option unconditionally. The later publish step
was skipped, so this run did not publish its container image.

## Changes

- `media.py` selects frame-timing options through `ffmpeg_compat.py`.
- New FFmpeg builds use `-fps_mode passthrough`.
- Older builds advertising `-vsync` use `-vsync 0`, the equivalent passthrough
  policy for this pipeline's single video output.
- If neither interface can be verified, processing fails clearly. It never
  drops the flag or silently converts variable timing to constant timing.
- The existing irregular-timestamp fixture also uses compatible flags.
- `media_preflight.py` runs actual synthetic video and audio through the real
  preparation and encoding functions inside the Docker image, before heavy
  dependency installation. It checks integer/fractional FPS, trimming,
  dimensions, frame count, audio presence and irregular-timestamp rejection.
- The existing final ComfyUI CPU execution gate remains unchanged.

The model, precision, VAE, source blend, color matching, dimensions, temporal
batch, noise settings, workflow JSON, GPU memory behavior, authentication,
container tag and RunPod settings are unchanged.

## Local validation

108 package tests passed on FFmpeg 7.1.5. These include real media execution
through both the modern option and the legacy `-vsync 0` preparation path,
plus unit tests for old/new capability detection and failure handling.
This is not a claim that the FFmpeg 4.4.2 binary was executed locally. The next
GitHub build exercises that binary through the new in-image preflight and
the unchanged actual ComfyUI CPU gate.

No Docker rebuild, GPU inference or restored-footage quality test was run
in the assistant's environment. A local test pass does not certify GPU fit
or model image quality.

## Apply and rebuild

Save `fix_faithful_video_ffmpeg.py` in Downloads. From your local repository
root, run:

```sh
python3 -c "from pathlib import Path; import runpy; runpy.run_path(str(Path.home() / 'Downloads' / 'fix_faithful_video_ffmpeg.py'), run_name='__main__')"
```

The standalone installer verifies its payload and reviewed source files,
backs up changed files under the repository's Git directory, and refuses
conflicts or symlinked paths. Re-running the same installer is safe when
installed files match. It does not fetch, stage, commit or push anything.
Use `--check` when invoking the script directly for a no-write preview.

Review and commit the changes in GitHub Desktop, then manually Push origin.
In GitHub Actions, start a NEW **Build Faithful Video Minimal** run on main.
Do not simply re-run the old failed run: that run still references the old
commit. Deploy only after the new run has successfully published.

The existing tag remains:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:faithful-video-v1
```

## Primary references

- FFmpeg 4.4.2 documentation, `-vsync` and passthrough mode:
  https://github.com/FFmpeg/FFmpeg/blob/n4.4.2/doc/ffmpeg.texi
- FFmpeg commit introducing `-fps_mode`, including its relationship to `-vsync`:
  https://ffmpeg.org/pipermail/ffmpeg-cvslog/2022-June/132700.html
- Current FFmpeg documentation:
  https://ffmpeg.org/ffmpeg.html
- Failed workflow run:
  https://github.com/jsanso1497/runpod-comfy-stateless/actions/runs/41767293920

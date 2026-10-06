# H3 Portrait 1.4 verification

Release: 1.4.0. Pipeline: role-routed-reference-v4.
Snapshot: h3-role-routing-1.4.0-r1.
General application source remains 3.2.0, except shared packaging/verification documentation.

## Executed locally for this release

| Check | Result |
| --- | --- |
| Existing general ComfyUI suites | 230 passed |
| Portrait suites, including the new 1.4 regressions | 198 passed |
| Shared link-only LoRA downloader suites | 69 passed |
| Source-snapshot suites | 22 passed |
| File-manager unit suites | 19 passed |
| HTTP compatibility policy self-tests | 10 passed |
| Total | 548 tests passed, no skipped or failed tests |
| Workflow graph connections and acyclicity | All 18 source graphs passed |
| Python 3.10-compatible source syntax | Passed |
| JSON, shell syntax, Docker COPY paths/RUN syntax, and Actions YAML/shell blocks | Passed static checks |
| Both portrait build configurations | Separate full/lite arguments and non-overlapping main tags verified |
| Frontend widget migration | Executed the actual JavaScript migration in Node; valid new, legacy, and already-shifted examples passed |
| JavaScript module syntax | Passed |
| Both portrait profile source checks | Passed |
| Actual local Jupyter HTTP service | Password login, anonymous rejection, CSRF, upload, download, edit and delete passed |
| Actual MP4 final-frame decoding | Six-frame H.264 fixture with audio decoded; saved PNG equals an independent extraction of frame 5 |

The final ZIP is re-extracted and its source snapshot/tests rechecked. The update ZIP is
also applied to a fresh copy of the prior supplied package; the result must match the new
snapshot without replacing the LoRA list. Results are provided in the verification archive.

## What the new tests establish

- Explicit uploader roles override inferred model roles. Pose/camera and expression guides
  are text-only by default; their pixels do not enter either native H3 reference path.
- All original images are presented to the vision analyst. Only retained native visual
  references reach H3, in order, with original image tensors, remapped picture numbers,
  and an explicit source-to-H3 ledger. The original analysis/brief remain in the record.
- A guide performer does not automatically become a new subject. A single target is
  normalized to the native subject; unresolved multi-person assignment stops explicitly.
- An unnecessary clarification with explicit roles gets at most one bounded review of the
  original task. A repeated essential clarification still stops. Nothing is silently accepted.
- Lite reuses one 8B Instruct helper for both calls; Full uses separate 32B Instruct/Thinking
  editions. Cleanup is checked before returning an H3 job. Model HTTP calls are mocked.
- The Director's 15 saved values include the frontend-generated seed companion. Combo labels,
  numeric finiteness, LoRA defaults and Draft-only startup are checked, not just Python inputs.
- Both new graphs have an explicit completed-export dependency before the final-frame node.
- MP4 export retains audio in the fixture test. A matching PNG/JSON is written from the last
  decoded displayed frame, not a pre-export image or estimated seek time. Unsafe/missing/
  modified files, cancellation and bounded extraction failures are tested.
- The native manual graph contains no Ollama director. Its optional LoRA and portrait crop
  are local helper nodes around the pinned native H3 sampling path.

## Real checks versus mocked checks

The local video test uses the real installed FFmpeg/ffprobe to encode a short fixture and
extract the last frame. Its native ComfyUI Video.save_to object is mocked locally because
this environment does not have the pinned ComfyUI/PyAV runtime. This is not a claim that a
real H3 video or native ComfyUI export was generated here.

Both portrait Docker builds now start a real CPU ComfyUI server and execute:
LoadImage -> native CreateVideo -> H3PortraitExportVideo -> H3PortraitSaveLastFrame.
The build checks the actual MP4/PNG pair and final-frame index. It prints
`H3 EXPORT CPU SMOKE PASS` only after that succeeds. That newly added Docker smoke test
has NOT run here; it must pass on GitHub before an image is published.

The file-manager live test used local JupyterLab 4.5.3 / Jupyter Server 2.17.0. The existing
Docker pins remain 4.6.4 / 2.21.1, whose live HTTP test is repeated by the Docker build.
The frontend test runs actual migration JavaScript in Node with an app stub, not a real
ComfyUI browser canvas. The running-registry test additionally checks the generated seed
companion; actual browser loading still requires the first deployment.

## Not executed or guaranteed

- No full Docker image was built or pushed here. Docker/Podman and a GPU are unavailable.
- No real Ollama model generated a prompt in this environment.
- No Full or Lite H3 inference, quantized-LoRA merge, peak VRAM/RAM measurement or speed
  benchmark was run. Lite uses smaller weights/helper/canvas but is not guaranteed to fit
  a particular lower-VRAM GPU or be a particular number of times faster.
- The exact user's three reference photos, generated prompt, video and LoRA were not supplied
  for this incident. The pipeline mechanism was inspected, not the precise perceptual cause.
- The private LoRA text file was not read, extracted, downloaded or tested. A comments-only
  stand-in was written locally. The update ZIP excludes the list completely.
- Live GitHub inspection returned 404. This release is based on the supplied source package
  and recorded build context, not on unseen changes in the live repository.
- Pose/camera text routing removes those source pixels but is not a pose ControlNet. It can
  lose exact geometry. Face/body/wardrobe/scene references are still visual conditioning;
  their verbal role limits are not hard pixel-level identity masks.
- Saving a last frame prepares an asset for chaining; it does not establish hard first-frame
  conditioning or guarantee a seamless next clip.

## Reproduce

Run from the repository root using the project's test dependencies:

```bash
PYTHONPATH=tests python -m unittest test_update test_krea_rebalance test_ollama_h3 test_quality test_single_person test_user_directed_h3
python -m unittest discover -s h3_portrait/tests -p 'test_*.py'
python -m unittest discover -s shared_loras/tests -p 'test_*.py'
python -m unittest discover -s tools -p 'test_snapshot.py'
python -m unittest discover -s file_manager/tests -p 'test_*.py'
python scripts/comfy_http_fix.py self-test
node tools/test_widget_serialization.mjs
python tools/verify_snapshot.py
python tools/check_repository.py --target portrait
python tools/check_repository.py --target portrait-lite
python tools/check_repository.py --target general
python tools/validate_repository.py
```

These commands document validation; they are not extra local-Mac setup steps for James.
Follow START_HERE.md for browser-only GitHub upload and RunPod deployment instructions.

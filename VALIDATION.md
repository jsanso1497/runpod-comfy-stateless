# Consolidated repository validation, 2026-10-05

General package: 3.2.0. Portrait package: 1.3.0.
Portrait pipeline: split-model-reference-v3.

## Executed in this environment

| Check | Result |
| --- | --- |
| General ComfyUI package regression suites | 230 tests passed |
| Portrait regression suites, including split-model routing | 150 tests passed |
| Shared LoRA downloader tests | 69 tests passed |
| Existing HTTP compatibility policy self-tests | 10 tests passed |
| Total | 459 tests passed; no skipped or failed tests |
| Python source syntax, using Python 3.10 grammar | 38 files passed |
| JSON parsing | 27 files passed |
| Shell syntax | 4 scripts passed |
| Dockerfile directive/continuation, COPY paths and shell RUN syntax | Both passed |
| GitHub Actions YAML, build paths, source-commit arguments and embedded shell syntax | Both passed |
| Static ComfyUI graph connections and required fields | 17 graphs passed |
| Portrait uploader JavaScript syntax | Passed `node --check` |
| Both repository-source preflights | Passed |

Portrait tests cover actual model-edition selection in mocked HTTP requests,
analysis-to-director unloading order, no image resubmission to the director,
separate model digests in cache identity, complete map preservation, compact
field bounds, malformed/truncated JSON regeneration from the original task,
Thinking-to-Instruct repair routing, and final model cleanup before H3 handoff.
They reject the old assumption that Qwen3-VL Thinking must accept think=false.
One service-level mock verifies that both selected models are downloaded and
that Ollama is configured to allow only one loaded model.

Source-release tests exercise the real manifest writer and reader in subprocesses.
They confirm that missing analysis files, older client revisions, changed node
copies and altered manifests fail validation. The generated runtime manifest is
created during Docker build. It is not a file the user maintains or a guarantee
that an old image would know how to perform these checks.

Shared-LoRA download tests use small simulated safetensors files, provider
responses and explicit disk-space fixtures. The real 8 GiB download reserve is
retained. The root duplicate test file and incomplete-paste failure are guarded
by the repository preflight; the test source includes all 69 corrected tests.

## Not executed here

- Neither complete Docker image was built or pushed. Docker is unavailable here.
- The new release has not run on RunPod or on a GPU.
- Ollama inference was mocked. Neither selected 32B model generated a real prompt
  in this environment; thinking/content handling, speed, and image interpretation
  still need the first live Draft-only test.
- No authenticated download of the user's actual LoRA was performed. Its link,
  licensing, full/pruned H3 target, strength and quality remain unverified.
- Peak RAM/VRAM/disk consumption and actual render times were not measured.
- Browser rendering/upload behavior and model-level identity fidelity were not
  tested. JavaScript syntax and Python unit tests are not a browser/GPU test.
- A fresh real CPU ComfyUI server with all pinned upstream packages was not run
  locally. Both Dockerfiles retain their actual CPU startup/schema checks and
  will execute those during GitHub builds before publication.
- The private live GitHub repo returned 404. This is a reconstruction from the
  supplied archives and fixes, not an export of unseen repository content.

## Reproduce local checks

From the repository root, with the project's test dependencies installed:

```bash
PYTHONPATH=tests python -m unittest test_update test_krea_rebalance test_ollama_h3 test_quality test_single_person test_user_directed_h3
python -m unittest discover -s h3_portrait/tests -p 'test_*.py'
python -m unittest discover -s shared_loras/tests -p 'test_*.py'
python scripts/comfy_http_fix.py self-test
python tools/validate_repository.py
python tools/check_repository.py --target general
python tools/check_repository.py --target portrait
node --check h3_portrait/node/web/references.js
```

These commands are for verification, not additional RunPod setup steps. Follow
START_HERE.md for the user's upload/deployment path.

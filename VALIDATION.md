# Complete source + file manager: verification report

Snapshot: `h3-clean-filemanager-2026-10-05-r2`.
H3 Portrait code remains 1.3.0, pipeline split-model-reference-v3.
General ComfyUI code remains 3.2.0. New file-manager integration: 1.0.

## Executed locally against the packaged source

| Check | Result |
| --- | --- |
| General ComfyUI regression suites | 230 passed |
| H3 Portrait regression suites | 150 passed |
| Shared link-only LoRA downloader tests | 69 passed |
| Source-snapshot tests, including Finder metadata | 22 passed |
| File-manager auth/config/supervisor tests | 19 passed |
| Existing ComfyUI HTTP compatibility self-tests | 10 passed |
| Total local unit/self-tests | **500 passed** |
| ComfyUI graph structure / reciprocal links | **17 workflows checked** |
| JSON/Python 3.10/shell source syntax | Passed |
| Both Docker COPY paths, entrypoints and shell RUN syntax | Passed static checks |
| Both GitHub Actions build paths, source-check steps and shell syntax | Passed static checks |
| JavaScript syntax | Portrait upload and RefMod download interfaces passed |
| Exact source inventory and normalized source hashes | Passed after ZIP re-extraction |
| Editable LoRA list | Content changes accepted without changing hashes |
| Real local file-manager HTTP test | Passed, details below |

The final ZIP was re-extracted and all tests above rerun against that extracted
copy. There are no model weights, credentials, previous ZIPs, old hotfixes, virtual
environments or Python caches in the package. The image tag remains
h3-portrait-clean. The LoRA link list is comments-only for the user to populate.

## Real local Jupyter HTTP test

This was NOT a mock server. The actual packaged service.py launched JupyterLab
with a temporary root and a randomly generated test password. The smoke test
confirmed:

- Login page available; unauthenticated /api/contents access denied.
- Password login succeeds; filesystem access then succeeds.
- Missing CSRF token prevents writes after authentication.
- Upload, raw download, edit and delete succeed with valid auth/CSRF.
- Raw password is not written in the generated config; an Argon2 hash is used.
- No URL token; private config/runtime outside the file-tree root.
- The test server was stopped and the temporary files were removed.

**Versions actually available for local HTTP execution:** JupyterLab 4.5.3,
Jupyter Server 2.17.0, Python 3.13.5. These are not the new dependency pins.

**Versions configured for the actual Docker image:** JupyterLab 4.6.4 and
Jupyter Server 2.21.1, verified on their official PyPI pages as Python >=3.10.
Installing those versions locally was blocked by this environment's external
network access. Therefore their exact runtime compatibility is NOT claimed as
locally executed. BOTH Dockerfiles run the same real smoke test using those pinned
versions before publication. They fail the build if it fails. This separates the
new dependencies from ComfyUI's Torch/transformers/aiohttp environment.

The supervisor's missing-secret, port, auth-readiness, app-failure and browser-
failure paths were additionally tested with process fixtures. No production GPU
process was started or killed by these tests.

## Preserved functionality

The H3 generation pipeline, prompt-model editions, split reference/director
routing, 9:16/2:3 dimensions, quality presets, original reference-image flow,
LoRA selection and one-line download catalog are unchanged from the consolidated
1.3 source. All 17 workflow JSON files are byte-for-byte identical to that source.
The file manager is a separate password-protected service on 8888, not a new
inference path and not merely an exposed unused port.

No globally disabled authentication, wildcard CORS, or disabled CSRF has been
added. Jupyter's owner login permits editing and terminal access to the container;
it is NOT a restricted security sandbox. It does not add a login to ComfyUI 8188.
No inference-model weight is downloaded at Docker build time.

## Remaining deployment checks

- Neither complete Docker image was built or pushed here; Docker is unavailable.
- The exact new Jupyter dependency pins require the included GitHub build smoke test.
- No RunPod HTTPS-proxy session or interactive browser UI was exercised here.
- No live Ollama model inference, H3 GPU inference or LoRA compatibility test ran.
- Ollama/provider responses in application unit tests are mocked.
- User LoRA URLs have not been supplied; authorization and model compatibility
  cannot be verified in this package.
- Peak disk/RAM/VRAM, render speed and reference-identity quality are unmeasured.
- The source uses existing pinned GHCR images; they must still exist and be
  accessible with the existing repository/package credentials.
- This is the complete intended source reconstructed from the supplied packages,
  not an export of Git history or unseen private-repository changes.

A successful GitHub portrait build and a first live Draft-only/video test are still
required. Passing these tests is not a promise that arbitrary workloads will fit.

## Reproduce

```bash
python tools/verify_snapshot.py
python -m unittest discover -s tools -p 'test_snapshot.py'
PYTHONPATH=tests python -m unittest test_update test_krea_rebalance test_ollama_h3 test_quality test_single_person test_user_directed_h3
python -m unittest discover -s h3_portrait/tests -p 'test_*.py'
python -m unittest discover -s shared_loras/tests -p 'test_*.py'
python -m unittest discover -s file_manager/tests -p 'test_*.py'
python file_manager/smoke_test.py
python scripts/comfy_http_fix.py self-test
python tools/validate_repository.py
python tools/check_repository.py --target portrait
python tools/check_repository.py --target general
```

These are verification commands for the build/test environment, NOT instructions
to run Terminal on the user's work Mac. Use browser-only START_HERE.md.

# Clean repository verification

Snapshot: `h3-clean-2026-10-05-r1`.
General application: 3.2.0. Portrait application: 1.3.0.
Portrait pipeline: `split-model-reference-v3`.

## Executed for this package

| Check | Result |
| --- | --- |
| General ComfyUI regression suites | 230 tests passed |
| H3 Portrait regression suites | 150 tests passed |
| Shared link-only LoRA downloader | 69 tests passed |
| HTTP compatibility policy self-tests | 10 tests passed |
| New source-snapshot tests | 21 tests passed |
| Total | **480 tests passed**, no skipped tests |
| Python 3.10 syntax compatibility | 40 Python source files parsed |
| JSON | 28 files parsed, including the source snapshot |
| Shell | 4 scripts passed bash syntax checks |
| Dockerfile instructions, continuations, RUN syntax and COPY source paths | Both recipes passed static checks |
| GitHub Actions YAML, shell blocks, build contexts and source-commit arguments | Both build workflows passed static checks |
| ComfyUI reciprocal connections and graph structure | 17 workflows passed static checks |
| Portrait custom-node serialized widgets | All six custom-node definitions matched their actual INPUT_TYPES; Director has 13 widget values in the correct order/types |
| JavaScript syntax | Portrait upload UI and existing RefMod download UI passed node --check |
| Repository source preflight | General and portrait passed |
| ZIP re-extraction | CRC, root layout, exact manifest verification and all 480 tests passed after extraction |
| Clean replacement rehearsal | Local Git main history preserved; obsolete files removed; one replacement commit created; LoRA links remained editable |

The application source is unchanged from the complete 1.3.0 consolidation.
The changes in THIS package are source-upload verification, the additional
`h3-portrait-clean` image tag, new packaging tests, and clean reset instructions.
No new model, rendering preset, LoRA behavior, or prompt architecture was added.

## What the new source check tests

Missing hidden workflow files, stale or truncated Python files, an extra enclosing
upload folder, a duplicate root test_links.py, mixed verifier/manifest versions,
unsafe manifest paths and source symlinks are rejected before a Docker build.
LoRA-link contents can change without editing any hash. Windows line endings and
UTF-8 BOMs are normalized for source hashing. Git metadata and Python caches are
excluded. This detects incomplete/mixed uploads; it is not a digital signature.

The manifest verifies the expected complete source tree, not merely a version
number printed by a script. It does not prove a model will render correctly.

## What the application tests cover

- Different actual Ollama editions for analysis and writing.
- Sequential model unloading and no image resubmission during the text-only pass.
- Compact complete reference maps, stable image ordering and model digest caching.
- Malformed/truncated JSON regeneration from the original task; no unlimited retry.
- Thinking-to-Instruct retry routing rather than think=false on a Thinking edition.
- Image-bundle copy checks and build-generated source-commit manifests.
- Shared-LoRA downloads with explicit disk fixtures and real low-space guard tests.
- Workflow links, UI field order, and preserved portrait resolution/step settings.

Ollama and provider HTTP responses are mocked in those tests. CPU Torch tensors
are used in the applicable image tests. No actual user photos or LoRA file were
available for a fidelity or model-compatibility evaluation.

## Not executed or guaranteed

- Neither full Docker build was executed or pushed from this environment; no Docker
  or Podman engine is available here. The two GitHub recipes still perform actual
  CPU ComfyUI startup/schema checks before publishing the images.
- No live RunPod or GPU execution of this source occurred here.
- No real Ollama vision/Thinking call or authenticated user-LoRA download occurred.
- No rendering-speed, peak VRAM/RAM/disk, or identity-fidelity benchmark occurred.
- No browser end-to-end upload or Windows Explorer/GitHub Desktop GUI test occurred.
  The equivalent clean working-tree replacement was rehearsed in a local Git repo.
- The private repository read returned 404. This source is based on conversation
  packages and fixes, not an unseen live repository export.
- The pinned base-image digest was confirmed in the earlier successful build log,
  but its present availability/permissions were not authenticated here. Keep the
  existing GHCR package and repository. Deleting that package breaks the dependency.

A successful new GitHub portrait build and the first live Draft-only/video test
remain necessary. Test counts are not a guarantee that GPU inference cannot fail.

## Reproduce checks

```bash
python tools/verify_snapshot.py
python -m unittest discover -s tools -p 'test_snapshot.py'
PYTHONPATH=tests python -m unittest test_update test_krea_rebalance test_ollama_h3 test_quality test_single_person test_user_directed_h3
python -m unittest discover -s h3_portrait/tests -p 'test_*.py'
python -m unittest discover -s shared_loras/tests -p 'test_*.py'
python scripts/comfy_http_fix.py self-test
python tools/validate_repository.py
python tools/check_repository.py --target general
python tools/check_repository.py --target portrait
node --check h3_portrait/node/web/references.js
node --check scripts/local_nodes/ComfyUI-Quality/web/refmod_download.js
```

These are verification commands, not extra setup steps. Your reset/deployment
instructions are in START_HERE.md. Do not manually maintain SHA values.

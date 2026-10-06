# H3 Portrait 1.5.3 validation

## Executed in this session

| Suite | Passing tests |
| --- | ---: |
| H3 Portrait, startup, Reference Pack handoff, local HTTP protocol | 211 |
| KREA Identity | 153 |
| H3 Media | 50 |
| General template | 230 |
| Shared LoRA downloader | 69 |
| File manager | 19 |
| Snapshot verifier | 22 |
| Protected model loaders and saved-workflow repair | 20 |
| Total Python unit/integration tests | 774 |

The separate HTTP-policy self-test also passed all 10 tests. Total test cases:
784, not counting the separate frontend, schema, syntax and package checks.

Other checks executed: Full/Lite/general repository preflights; H3 source/release
verification; two Node.js frontend serialization contracts; Python 3.10 syntax
parsing; shell syntax; and Actions YAML parsing. Final packaging additionally
checks source snapshot hashes, ZIP extraction, changed-file overlay completeness,
private-LoRA-file preservation using a sentinel fixture, and graph connections.

Three tests use a real loopback HTTP server that simulates Ollama. They exercise
streamed pulls, installed-model reuse, missing-model errors and unload/confirmation
requests. This is real local HTTP, not actual Ollama inference or a model download.
Other model/ComfyUI memory operations are tested with mocks. Graph tests check
structure and serialized settings, not image quality.

## Added to the Docker build, not executed locally

The build installs the pinned actual Reference Pack, applies and verifies the
local-only/handoff patch, then executes its real none-provider class on CPU. It
also runs the actual ComfyUI registry/schema checks for the protected and native
graphs and retains the native export/final-frame CPU probe. No external prompt
inference is required for those build probes.

## Not executed or claimed

No final Docker image build or GHCR publication, live RunPod deployment, actual
Ollama 32B/8B inference, full upstream dependency installation, GPU H3/KREA/SeedVR2
generation, render-time benchmark or identity/quality evaluation ran here. This
runtime has no Docker daemon, and direct package downloads were unavailable.
Upstream source contracts were inspected through the GitHub connector.

The original VRAM error is a real allocation failure. The logs do not establish
its sole cause. Verified unloading and a non-dynamic allocator are implemented
mitigations, not evidence that every requested reference size will fit on an A100.

The first successful GitHub build, a live Draft-only run, and a short live render
remain necessary deployment checks.

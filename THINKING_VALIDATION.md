# Validation: H3 Portrait 1.2 thinking update

## Executed locally

- 106 portrait tests passed: all 57 existing tests (transport expectations updated
  from one pass to two), plus 49 additional schema, transport, cache and policy tests.
- 69 existing shared-LoRA tests passed using the complete corrected test_links.py
  from the user's last successful build-fix package. That file is NOT in this patch.
- 10 existing HTTP-policy tests passed with the HTTP patch unchanged.
- Python compilation, Python 3.10 syntax parsing, and start.sh Bash syntax passed.
- The existing graph validator passed through the portrait test suite. The workflow
  JSON itself is byte-for-byte unchanged, as are the uploader JavaScript, models.json,
  shared LoRA list and GitHub portrait build workflow.
- All H3 generation-backend code beginning at _LoraWarnings in node/__init__.py is
  byte-for-byte unchanged: LoRA key checks, model loading, native reference VAE path,
  sampler and exact-aspect crop are not altered by this prompt-helper update.
- Main/optional widget names, order, types and outputs remain unchanged.

Coverage includes both calls carrying all selected images in order; think=true;
locked ownership data carried from pass 1 into pass 2; ordinary picture mentions;
invalid, repeated and out-of-range labels; analysis/director clarification; bounded
repair; truncation; invalid stream packets; HTTP failures; cancellation; failed
unloading blocking H3; loopback-only transport; thinking text excluded from final
records/H3 input; sampling/policy/model changes invalidating cached drafts; and
video seed/quality changes retaining the existing draft.

These tests use local mock HTTP responses, not a running Ollama server. Tensor/image
checks use CPU Torch 2.10.0. Local Python is 3.13.5; all Python sources in this patch
also pass ast.parse with feature_version=(3, 10). This is not the deployment's
Torch/CUDA stack.

## Retained in the GitHub Docker build, not executed here

The existing shared-LoRA tests and real ComfyUI CPU startup/node registry check,
as well as the portrait tests including the new thinking suite. The same pinned
base image/dependencies and existing disk-cleanup workflow are kept. No LLM or H3
weight download or GPU inference is run during the build.

## Not verified here

The complete Docker build, live authenticated downloads, deployed /api/show response,
real thinking/structured-output combination, browser operation, GPU peak memory,
wall-clock latency, visual reference analysis accuracy or generated identity fidelity.
The published model tag is about 21 GB, not a measurement of total VRAM/RAM. Thinking
plus two passes is expected to take longer than one 8B Instruct call; no benchmark
or guaranteed quality gain is claimed. Temperature 0.25 is an engineering starting
point, not an official vendor recommendation or a measured optimum.

The reference map is validated structurally and cannot be replaced by pass 2's JSON.
It can still contain a semantic mistake from pass 1, or the director can misstate an
action in ordinary prose. Review the prompt and output. Existing one-target-per-image
mapping and the native nine-image limit remain; an ambiguous multi-target group photo
is not resolved by secretly duplicating/dropping sources.

Source basis: the supplied runpod-h3-portrait-links.zip plus the user's corrected
shared-LoRA test file, and primary Ollama/MiniMax documentation linked in
THINKING_START_HERE.md. The GitHub connector returned Not Found for the private
repository read. No live repository files or RunPod settings were changed here.

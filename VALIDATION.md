# H3 Portrait 1.3 validation

See [the complete repository report](../VALIDATION.md).

150 portrait tests passed locally, including compact JSON validation, model
switching, bounded retry, failure cleanup, cache identity, source-copy manifests,
geometry, selected LoRA plumbing and workflow structure. A separate 69-test
shared downloader suite and 10 HTTP-policy tests also passed.

The two prompt models are different checkpoints:
- Reference analysis: abliterated Qwen3-VL 32B Instruct Q4_K_M.
- Director: abliterated Qwen3-VL 32B Thinking Q4_K_M.

Tests confirm the request routing using mocks. They do not prove model-level
quality, complete GPU residency behavior or successful prompt generation.
The complete Docker build, real Ollama calls, browser execution and GPU video
render still require the user's GitHub/RunPod deployment. No hidden upload,
source commit, image publication or automatic background task was performed.

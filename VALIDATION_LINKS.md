# Validation: shared LoRA links / H3 Portrait 1.1

Executed locally:
- 62 new shared-library tests: URL parsing, Civitai version and file resolution,
  FP32/other selection parameters, HF repository and file URLs, checksum variants,
  token scoping, redirect guards, resumable downloads, checksum rejection,
  real small safetensors headers, failed-link isolation and index output.
- 57 current portrait tests: existing prompt/reference/memory/geometry behavior,
  selection-only LoRA loading, duplicate selection rejection and trigger metadata.
  The retired numeric-ID resolver tests were replaced by the shared resolver tests.
- 174 existing ComfyUI package tests passed against a temporary merged copy that
  includes the successful Rebalance and test-selection fixes.
- 10 unchanged HTTP-policy tests passed. Total: 303 tests across these four suites.
- Python AST/compilation, Bash syntax, YAML/JSON parsing, reciprocal workflow links,
  acyclic dependencies and the actual Director widget/schema count were checked.
- Integration checks confirm both images copy the SAME link list/module, the
  portrait action rebuilds on shared-list changes, existing private tokens are not
  baked into either image, and model files/presets are unchanged.

Provider responses and network downloads in tests are fixtures or mocks. The file
integrity tests use real tiny safetensors bytes; no generation weights are used.
No real Civitai or HF LoRA link has been supplied by the user. An attempted live HF
metadata lookup could not resolve its host from this environment, so no live
provider download is claimed.

Build-time validation remains enabled: the real ComfyUI registry/CPU smoke check,
existing runtime tests, HTTP checks and disk guards. Shared-library tests also run
inside both images during their build. No private token is required for those tests.

Not executed: complete Docker/GitHub image build, RunPod download/startup, browser
LoRA dropdown selection or GPU generation using the actual adapter. Authenticated,
gated and paid file access depends on the user's provider account. Code-level
compatibility and published model labels are not a quality/identity guarantee.

No GitHub write or RunPod account mutation was performed. Integration is based on
the supplied package files, not an inspected snapshot of the inaccessible private
repository. The package does not replace the user's recent build-check fixes or
existing model/LoRA catalogs.

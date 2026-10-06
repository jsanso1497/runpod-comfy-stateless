# 1.5.2 validation additions

Local validation for this release covers Reference Pack local-only defaults, Full/Lite model routing, aspect/quality geometry, workflow installation, Docker pin/patch instructions and source-drift checks. The complete Docker build and live GPU H3 generation are still deployment checks and are not claimed as locally executed.

**Completed locally:** 721 Python unit tests across the repository, 22 source-snapshot regression tests within that total, 2 frontend widget-serialization checks, Full and Lite repository preflight checks, source workflow graph validation, Python compilation and shell syntax checks.

# H3 Portrait 1.5 validation

See `../VALIDATION.md` for executed commands, results, and limitations.

Release-specific checks cover:

- role-aware H3 video routing;
- still safe-swap routing where pose/camera, expression, and scene guides are text-only;
- MiniMax H3 Ref2VA five-frame single-image recipe and one-frame selection;
- no separate still-image model assets in Lite;
- corrected ComfyUI seed/widget serialization;
- exact exported-video final-frame PNG;
- Full/Lite workflow installation and source/version verification.

GPU H3 inference is not performed by the local test suite.

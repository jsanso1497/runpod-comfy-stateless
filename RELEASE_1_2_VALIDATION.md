# Krea + H3 Media 1.2.0 validation

```text
KREA IDENTITY + H3 MEDIA HQ 1.2.0
Prepared: 2026-10-06
Snapshot: h3-1.5.0-krea-identity-1.2.0-media-r1

EXECUTED LOCAL TEST SUITES
PASS: general: 230 tests
PASS: portrait: 171 tests
PASS: shared_loras: 69 tests
PASS: file_manager: 19 tests
PASS: krea_identity: 153 tests
PASS: h3_media: 50 tests
PASS: snapshot_tests: 22 tests
TOTAL: 714 unique Python tests passed, with no skipped tests in the repository runs.
64 additional cases: 14 Krea described-scene tests and 50 H3 media tests.

REAL MEDIA FIXTURE TESTS
Generated a local H.264/AAC MP4 using FFmpeg and exercised the actual decoder:
- Voice-only extracts trimmed PCM without calling video-frame decode.
- Still-identity extraction, motion conversion from 30 to 24 fps and 17k+5 snapping.
- Simultaneous voice/identity/action, selected-frame position, active trim and end trimming.
- Missing audio gives a clear error; visual-only accepts clips without audio.
- Input file bytes are unchanged; no original audio passthrough is wired to output.
These are demux/conditioning-input tests, not diffusion-model or voice-likeness tests.

ISOLATED DOCKER-MODULE FILESYSTEM TESTS
PASS: Krea tests in an isolated module directory: 153 run, 1 repository-only Dockerfile assertion skipped.
PASS: H3 tests in an isolated module directory: 50 run, 1 repository-only graph-regeneration test skipped.
The actual ComfyUI node-registry build gate remains mandatory and is not mocked or skipped.
These repeated runs are not added to the unique-test total above.

STATIC REPOSITORY CHECKS
KREA IDENTITY STATIC PASS: eight graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
H3 MEDIA STATIC PASS: BF16 native conditioning, MP4 roles, generated audio only, final-frame dependency. No inference run.
{
  "python_sources": 67,
  "json_files": 49,
  "shell_scripts": 5,
  "dockerfiles_checked": 2,
  "actions_workflows_checked": 3,
  "graphs_checked": 28,
  "krea_identity_graphs_checked": 8,
  "h3_media_graphs_checked": 1
}
STATIC REPOSITORY CHECKS PASS. No Docker build or GPU inference performed.

FRONTEND CHECKS
H3 FRONTEND SERIALIZATION PASS: 15 widgets including seed control; legacy and corrupted migrations; new defaults unchanged.
H3 REF2VA STILL SERIALIZATION PASS: 15 widgets including seed control; safe-swap + Draft-only defaults; no shifted/NaN values.

REPOSITORY PREFLIGHTS
KREA IDENTITY STATIC PASS: eight graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
H3 MEDIA STATIC PASS: BF16 native conditioning, MP4 roles, generated audio only, final-frame dependency. No inference run.
REPOSITORY PREFLIGHT PASS: general
H3 PORTRAIT SOURCE VERIFIED: 1.5.0 | role-routed-reference-still-v5
H3 PORTRAIT RELEASE VERIFIED: 1.5.0 | source=local-uncommitted
KREA IDENTITY STATIC PASS: eight graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
H3 MEDIA STATIC PASS: BF16 native conditioning, MP4 roles, generated audio only, final-frame dependency. No inference run.
REPOSITORY PREFLIGHT PASS: portrait
H3 PORTRAIT SOURCE VERIFIED: 1.5.0 | role-routed-reference-still-v5
H3 PORTRAIT RELEASE VERIFIED: 1.5.0 | source=local-uncommitted
KREA IDENTITY STATIC PASS: eight graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
H3 MEDIA STATIC PASS: BF16 native conditioning, MP4 roles, generated audio only, final-frame dependency. No inference run.
REPOSITORY PREFLIGHT PASS: portrait-lite

BACKWARD COMPATIBILITY
PASS: 29 prior workflow/catalog files are byte-identical.
PASS: all seven previous Krea workflow JSONs are unchanged.
PASS: all original general-image and H3 workflow JSONs are unchanged.
PASS: SeedVR2 CPU schema-guard function is byte-identical.
PASS: copied H3 full BF16 model catalog matches the existing full catalog.
PASS: no private LoRA list content included in the update.

NOT RUN
Full Docker build/push; live ComfyUI CPU node registry or native H3 encode execution;
model downloads; RunPod startup; GPU diffusion; visual/voice identity comparisons;
H3 motion-quality evaluation; peak RAM/VRAM and throughput measurement.

The adapter targets the existing pinned native H3 API, adds a real registry build gate,
and fails clearly at runtime if the actual native signature is incompatible.
Unit tests use a mocked native encoder only to verify argument forwarding, not to prove inference.
No live GitHub changes were made. This is a cumulative file update based on the supplied ZIP.
```

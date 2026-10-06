# Krea Identity 1.1.0 validation

```text
KREA IDENTITY 1.1.0 - LABELED REFERENCES AND TWO-PERSON REPLACEMENT
Prepared: 2026-10-06
Snapshot: h3-1.5.0-krea-identity-1.1.0-r1

IMPLEMENTED
- Workflow 05: up to six labeled original references and text-directed target/replacement selection.
- Workflow 06: man/woman single-pass replacement with separate A/B reference banks and clothing choices.
- Workflow 07: separate Krea generations in selected crops with protected masked compositing.
- Existing screenshot rebalancer is available for every pass with the same layer weights.
- Source aspect is automatic. Protected final composites retain the original scene dimensions.
- Original four Krea workflows, all inherited image/video workflows and model manifests unchanged.
- Existing SeedVR2 CPU build-schema guard retained unchanged.

EXECUTED PYTHON TESTS
PASS: general: 230 tests
PASS: portrait: 171 tests
PASS: shared_loras: 69 tests
PASS: file_manager: 19 tests
PASS: krea_identity: 139 tests
PASS: snapshot: 22 tests
TOTAL: 650 tests passed. 77 new directed-reference/region/graph cases.
The new tests use CPU tensors and synthetic images. They are not diffusion-model or likeness benchmarks.

STATIC REPOSITORY CHECKS
KREA IDENTITY STATIC PASS: seven graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
{
  "python_sources": 58,
  "json_files": 43,
  "shell_scripts": 5,
  "dockerfiles_checked": 2,
  "actions_workflows_checked": 3,
  "graphs_checked": 26,
  "krea_identity_graphs_checked": 7
}
STATIC REPOSITORY CHECKS PASS. No Docker build or GPU inference performed.

EXISTING FRONTEND SERIALIZATION CHECKS
H3 FRONTEND SERIALIZATION PASS: 15 widgets including seed control; legacy and corrupted migrations; new defaults unchanged.
H3 REF2VA STILL SERIALIZATION PASS: 15 widgets including seed control; safe-swap + Draft-only defaults; no shifted/NaN values.

REPOSITORY PREFLIGHTS
KREA IDENTITY STATIC PASS: seven graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
REPOSITORY PREFLIGHT PASS: general
H3 PORTRAIT SOURCE VERIFIED: 1.5.0 | role-routed-reference-still-v5
H3 PORTRAIT RELEASE VERIFIED: 1.5.0 | source=local-uncommitted
KREA IDENTITY STATIC PASS: seven graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
REPOSITORY PREFLIGHT PASS: portrait
H3 PORTRAIT SOURCE VERIFIED: 1.5.0 | role-routed-reference-still-v5
H3 PORTRAIT RELEASE VERIFIED: 1.5.0 | source=local-uncommitted
KREA IDENTITY STATIC PASS: seven graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.
REPOSITORY PREFLIGHT PASS: portrait-lite

BACKWARD COMPATIBILITY
PASS: 28 protected workflow/model/inference files are byte-identical to the working 1.0.1 snapshot.
PASS: SeedVR2 CPU schema-guard function is unchanged.
PASS: new workflow generation is reproducible, and local node widgets/signatures match all new graphs.
PASS: tests cover exact outside-mask pixel preservation and first-person protection during the second composite, including feathered masks.

NOT RUN
Full Docker build/push, live ComfyUI server/node registry, RunPod deployment, model downloads, GPU rendering,
visual likeness/realism comparisons, GPU memory or throughput measurements.
The Docker stages still require real ComfyUI object_info validation for all seven Krea graphs.
There is no Docker/GPU runtime here; direct upstream downloads also failed DNS resolution.
No live GitHub changes were made. This is a cumulative update to the user's provided source snapshot.

KNOWN LIMITS
Reference sheets and purpose labels are a prompt/conditioning arrangement, not a newly trained multi-reference model.
Single-pass two-person results can mix faces or change unrelated scene content.
Protected editing requires correct regions/masks; rectangles are not automatic person detection.
First-person pixels take priority on mask overlap. Global upscaling voids pixel-preservation guarantees.
Six references per person is an integration limit, not a promise that six improves likeness over two.
The two-person graphs are not claimed to outperform the user's baseline without GPU image comparisons.

APPLY
Merge the ZIP contents, including .github and SOURCE_SNAPSHOT.json, into the existing repository.
Keep the existing config/lora_links.txt, which is excluded from this update.
Commit and rebuild General/Image from the new commit; deploy that successfully built image.
Keep the same RunPod environment variables and existing models.
```

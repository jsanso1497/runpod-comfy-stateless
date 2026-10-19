# QOVERLAY: selective overlay from a finished edit

Load the full original photo as **A** and the complete edited photo as **B**.
The selection says where B may replace A. Nothing in this workflow generates
another image or runs the original through a VAE.

## Use it

1. Open **Selective overlay: original A + selected parts of edited B** in the
   Workbench task browser, or import the included JSON. Do not reopen an old
   personally saved copy of the former watermark-generation graph.
2. Load A in the first image node and B in the second. Both must already be
   aligned and have identical pixel dimensions. A size mismatch is rejected
   rather than silently stretching, cropping, or misplacing your edit.
3. Choose a selection method. With `auto_mask` ON, SAM selects the named object
   on A. The existing default, `watermark`, is retained; change that text to the
   object you need. With `auto_mask` OFF, paint and save a mask on **Original A**
   using MaskEditor. This manual route does not call SAM inference.
4. Adjust expansion, feather, and inversion in node 4. Queue the workflow.
   The large **FINAL PHOTO** node saves the complete composite. The other two
   views are labeled mask previews, not the final output.

**White = edited B. Black = original A. Gray = blend.** Expansion and feather
are in original-image pixels. Feather extends outward, so its transition band
also includes some pixels from B. Set expansion and feather to zero for a hard
mask. Pixels outside the final mask remain exact copies of loaded A. With
expansion or feather enabled, the existing mask utility uses the 50% contour
as the solid selection; set both to zero to pass a hand-painted soft mask
through without that contour conversion.

The optional `protect_mask` socket on the compositor can lock pixels to A after
feathering. It is not required. Changing the reveal selection never regenerates
B. You can create B in any of your existing Qwen workflows first.

## What this replaces

The active QOVERLAY task no longer contains the native Qwen editing subgraph,
prompt enhancer, crop/stitch chain, inverted protection-mask derivation, or
transparent-overlay exporter. QNATIVE, QKEEP1, MOSS, Krea, other workflows, model
filenames, secrets, and environment settings are not changed.

The previous generator remains available in Python as `make_legacy_overlay`
for compatibility tests. Existing runtime nodes remain available for old saved graphs. The factory builder now uses
`make_overlay` to create this selective A/B graph, so a rebuild will not bring
back the broken generation-and-stitch design.

## Output and limits

The primary output is an opaque RGB PNG at A's dimensions, saved under
`Qwen21_Selective_Overlay/FINAL_A_plus_selected_B`. No transparent PNG is needed.
Load a complete edited photograph as B, not a transparent overlay cutout.

This is a pixel compositor, not an alignment or color-matching tool. It cannot
correct a pose change, shifted face, different framing, lighting change, or
color difference already introduced in B. Feather can soften a boundary but
cannot make mismatched geometry line up. Precision claims concern the loaded
image pixels, not original JPEG file bytes, metadata, or color profiles.

## Installation and deployment

The package installer modifies only local repository files and creates a
rollback backup under the repository's Git directory. It does not fetch,
pull, commit, or push. It checks the affected source files against the versions
inspected for this package and stops on unexpected differences.

The installer updates the QOVERLAY generator, task metadata, prebuilt JSON,
its two legacy test imports/catalog expectations, and adds current behavior
tests. Existing generated configurator files are refreshed for QOVERLAY only.

Commit and push using GitHub Desktop. After the repository validation passes,
rebuild the Qwen workspace for future pods. This JSON also uses nodes already
present in the inspected Qwen workspace and can be imported into an existing
pod without changing its Python node code. Live node availability and SAM
execution still require a real-pod check.

## Validation

New tests cover graph connectivity, A/B direction, selection on A, no generation
nodes, output routing, non-mutating source behavior, zero/one/soft masks,
inversion, finite feathering, size mismatch rejection, hard protection, and
RGB PNG pixel preservation. They use the repository's existing compositor and
feather functions on CPU. Live SAM inference, the ComfyUI browser, and the
complete repository test suite were not run in this delivery environment.

Run in the repository with its test dependencies available:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_qwen_selective_overlay.py
python3 scripts/build_site.py
python3 scripts/validate.py
PYTHONPATH=src python3 -m pytest -q
```

The normal site build may download its existing pinned upstream workflow
snapshots if they are not cached. The package installer itself is offline.

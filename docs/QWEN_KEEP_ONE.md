# QKEEP1: keep one person, repair the others, protect painted pixels

This is an additive Qwen workflow. QNATIVE, QOVERLAY (watermark removal), the
other Qwen workflows, private libraries and model selections are not replaced.

## What it does

1. Load the **complete original sRGB photograph**. A mask painted on this
   LoadImage node means **protect these original pixels**. It does not select
   people for removal. Leaving the source mask blank is supported.
2. Queue with `run_edit = false` for a numbered preview. Native SAM 3.1 detects
   people and orders IDs by bounding-box center, left to right, then top to bottom.
   Select `keep_person = 1` through `5`. Green is the keeper, red is the repair
   region, and blue is painted protection. Recheck IDs after changing source,
   threshold or SAM refinement.
3. Set `expected_people` to the number you can see when you want a count check;
   0 is automatic. The detector requests `person:6` internally because this core
   SAM version otherwise defaults to one instance per plain text category. The
   sixth detection is an overflow check: more than five detected people stops
   the run rather than silently dropping someone. This is not a guarantee that
   SAM found everyone; inspect the preview.
4. Turn `run_edit = true`. Qwen receives one bounded scene crop containing the
   detected group and one isolated, neutral-background reference of the keeper.
   The reference is not pasted into the output. The prompt asks Qwen to remove
   the others and reconstruct the background and newly exposed details.
5. The generated crop is returned to its exact source coordinates. Only the
   removal/repair region is blended into the original. Visible keeper pixels
   and every painted protection pixel are locked after mask expansion and
   feathering. Where another person originally hid part of the keeper, new
   detail can be reconstructed in the editable area, not inside the protection.

The defaults are SAM automatic detection, keeper 1, a **preview-only first run**,
2048 squared maximum processing area, 3072 maximum working long side, 128 source
pixels of context, 24 pixels of repair-mask growth, 2 pixels of keeper protection
growth, 0 pixels of manual-protection growth, and 12 pixels of inward feathering.
These defaults apply only to QKEEP1. QOVERLAY still has its own SAM/edit ON defaults.

## Outputs

The primary preview is `*_composite.png`: an **opaque, full-resolution RGB photo**.
The following files are saved together under `Qwen21_Keep_One`:

- `*_composite.png`: original photo with the permitted repairs applied.
- `*_overlay.png`: a full-canvas, straight-alpha RGBA edit layer. It is transparent
  over the keeper, all painted protection, and unchanged pixels.
- `*_edit_mask.png`: the actual quantized overlay alpha.
- `*_audit.json`: sizes, selection IDs, protected-pixel audit and workflow version.

Place the overlay at x=0, y=0 over the same original with Normal blending and
100% opacity. Its alpha is quantized before the matching composite is calculated,
so ordinary 8-bit sRGB compositing reproduces that output. Different working
color spaces or linear-light blending in an editor can produce different edges.
`show_overlay_preview` displays the layer after the full photo; the layer is
saved even when this option is off. The normal preview deliberately does not
show a transparent edit layer in place of the finished photo.

## Protection and repair boundaries

The original LoadImage mask is **protection**, including any nonzero soft-edge
pixels. Protection always wins over a requested removal. Painting over a person
marked REMOVE means those protected fragments remain in the original photo; the
preview and audit report warn about that overlap. A fully blocked removal stops
with an explanation rather than saving an unexpected partial result.

Person segmentation does not reliably include distant cast shadows, reflections,
objects being held, or all clothing edges. Increase `repair_grow_pixels` for nearby
edge cleanup. For a remote shadow/reflection, connect a **separate full-size
extra_repair_mask** to node 3. This optional input is an editable region, unlike
the protection mask on the original loader. Its pixels cannot override protection.

The 128-pixel context is for model understanding, not permission to replace the
rest of the source. All model resizing is limited to the working crop/reference;
the full-resolution original is retained. Native edit `resolution` must remain
0 and `custom_size` off so the crop returns with the correct geometry. Lowering
the preparation node's resolution to 1536 reduces model processing area without
resizing the final canvas, though reconstructed detail may be less fine.

## Models and installation

Uses existing Qwen 2.1 BF16 diffusion/base text encoder/VAE, native SAM 3.1, and
the existing stock ComfyUI I2I prompt enhancer. No new weight files are introduced.
The native sampler remains 25 steps, CFG 1, Euler/simple. `refine_prompt` remains
optional and off by default. The stock text-preview pass-through inside the new
subgraph is removed and its STRING is forwarded directly, so preview-only runs
have no independent prompt-generation output node. Sampling and enhancer model
settings are not changed. The new subgraph has its own ID, so existing native
Qwen graphs are not altered.

The repository updater adds the custom-node registration, task catalog, build
hook, graph, documentation and tests. `scripts/build_site.py` regenerates QKEEP1
after the existing native Qwen templates. Add `QKEEP1` to an existing `WB_TASKS`
list to pre-prepare its assets. Its requirements are `qwen,sam,qwen_stock_pe`,
which the current complete Qwen environment already prepares. Other environment
variables and private secret bindings stay unchanged.

**A JSON file alone cannot install the three new WBPeople nodes.** Use a rebuilt
Qwen image after applying the repository update, or install the accompanying
live-Pod node kit, restart the managed ComfyUI process, and import the JSON.
The live kit does not modify the GitHub repository or download models.

## Limitations and verification

A single detected person with no extra repair region is a no-op: Qwen is skipped,
the original RGB photo is saved, and the overlay is fully transparent. No detected
people, a nonexistent keeper index, mismatched mask dimensions, or a sixth
person produces an error. A generated transparent hole in the repair area is
rejected instead of silently showing the removed person underneath it.

Exact preservation is of the decoded **8-bit sRGB image representation** outside
the exported alpha and in the protected regions. It does not preserve JPEG file
bytes, EXIF metadata, HDR/16-bit values, or a non-sRGB source profile.

CPU tests cover stable numbering, native multi-instance arguments/unloading,
mask priority, same-size opaque PNG output, real PNG save/reopen, alpha
recomposition, no-op handling, input validation, native graph wiring, unchanged
sampler/enhancer settings and additive catalog behavior. SAM and Qwen inference
are mocked in those tests. No live GPU inference, detection accuracy, Docker
build or semantic image quality is certified. Background/occluded detail is
plausibly reconstructed, not recovered from evidence that was never visible.

Sources inspected:
- ComfyUI pinned SAM detection: https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy_extras/nodes_sam3.py
- SAM multi-instance prompt parsing: https://github.com/Comfy-Org/ComfyUI/blob/52f98af2e2e42c421070a3e147c161c47cdeaf22/comfy/text_encoders/sam3_clip.py
- Native Qwen workflow: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1

# Qwen 2.1 Torso Lock v1.0.0

A focused add-on for the existing Qwen 2.1 / AusBoss Pod. Change a crop top to a
sports bra, reconstruct only the newly revealed skin, and keep the existing
abdomen and navel as original scene pixels. No BFS, face restorer, extra model,
prompt enhancer, automatic body normalization, or automatic upscaler is used.

## Start with workflow 36

`36_Sports_Bra_Torso_Landmark_Lock.json`

| Input | What to supply |
|---|---|
| A: ORIGINAL SCENE / EDIT | The crop-top photograph. Paint a white MASK over the old garment, intended new garment area, and newly revealed skin. |
| B: ORIGINAL PHOTO / PROTECT | The identical original photograph. Paint a white MASK over the already-exposed abdomen, including the belly button and surrounding anatomy. |
| C: PRIMARY TORSO | The closest-angle genuine torso photograph. Keep enough ribcage, waist and navel context to identify the anatomy. |
| D/E: OPTIONAL TORSO ANGLES | One or two useful genuine supporting views. Leave `__none__` when unused. |
| F: OPTIONAL GARMENT | The sports bra design. Leave `__none__` to use the garment-description field instead. |

Three loaders must be filled, using two unique photographs: the scene in A/B
and the genuine torso photo in C. D/E/F are genuinely optional: no dummy images are encoded. Uploading a
reference into an optional loader uses ComfyUI's normal input directory.

### Paint the masks carefully

A's white mask means **may change**. B's white mask means **must stay original**.
Paint on Mask Editor's MASK layer, never the RGB Paint Pen. The source RGB
comparison deliberately rejects a different original, a recrop or painted marks.

For A, include all obsolete crop-top fabric, the new bra silhouette, and the
small skin/edge/shadow region that must change. The skin hidden under the old
crop top is part of that same edit. Allow a small margin around old cloth to
prevent retained fabric at a feathered edge. Leave face, hair, arms, trousers,
background and foreground objects out unless they genuinely need editing.

For B, protect the already-exposed abdomen broadly, not just a tiny circle around
the navel. Leave a narrow transition at the old hem editable only where a shadow
or edge must be repaired. Do not protect fabric that needs removal. Protection
wins where masks overlap. The node does not detect the navel for you.

The preview shows **orange = editable** and **cyan = locked original**. It is
for inspection only; those colors are never sent to the model. `36A` runs this
mask check without loading model weights. After checking, copy the two painted
Load Image nodes into 36, or select their saved masked PNGs in 36.

### Why the original navel cannot be moved by the final paste

1. The crop bounds include the UNION of the edit mask and protected abdomen.
   Qwen therefore sees the established navel/abdomen even though it cannot replace
   their final pixels. Keep B focused on the relevant abdomen; protecting the entire
   photo would turn this into an unnecessarily large context crop.
2. Qwen edits the unmarked context crop. It is **not** given a masked latent.
3. AusBoss places the generated crop back in the original-size scene.
4. A separate compositor caps the paste at A, feathers only INWARD, then restores
   B from the original photo. AusBoss's mask here determines context, not final
   permission. Its own blend/grow/blur controls stay at zero.
5. Exact RGB tensor checks verify that B equals the original, and that everything
   outside the actual paste equals the pre-generation base with its original lock.

This preserves the selected original pixels. It does not verify anatomical
correctness, forbid a second invented navel elsewhere, perform 3D registration,
or guarantee a perfect tone match. Inspect the **verified final**, not a raw
model crop. New details still require generation unless you use an aligned
photographic composite with workflow 38.

## Defaults

- Existing Qwen diffusion BF16, Qwen3-VL 8B BF16 (`qwen_image`), and Qwen VAE BF16.
- Native encoder, independently sized references, resolution 0, matching latent.
- 40 steps, Euler/simple, CFG 1, denoise 1, fixed seed, default/lossless KV cache.
- Approximately 2 MP context crop; original scene dimensions retained.
- References capped near 1 MP without deliberate detail enlargement.
- Inward feather: 8 source-image pixels. Protection padding: 0.
- No automatic post-generation tone shift and no final full-frame upscale.

Keep denoise 1: it is an empty-latent reference edit, not conventional low-denoise
image-to-image sampling. Test 4 MP by changing only Crop For Inpaint's target if
there is a detail-resolution problem; more resolution is not guaranteed to improve
anatomical correspondence. Do not alter the crop's Multiple=32.

The actual prompt is built from the connected reference roles. If F is supplied,
its garment design takes precedence over the generic garment-description field.
Use the extra-instructions field for a specific local correction. Avoid manually
referencing image numbers there, as optional references change the numbering.
The saved audit JSON records the exact compiled prompt and reference order.

## Optional follow-ups

### 37: Torso skin / seam repair, original lock retained

Only use after inspecting the result of 36. Load the accepted result into A and
paint only the small new-skin/seam error. Keep the original crop-top photograph
and its protection mask in B. C remains the genuine torso evidence. The workflow
restores the original protected abdomen before the refinement and again after it.
Outside the new repair mask, it preserves the accepted result, except for the
explicit restoration of original protected pixels. It does not re-create the bra.

### 38: Aligned photographic patch, no generation

This is a strict compositor for an **already aligned**, original-size candidate
made externally from real photographic pixels. Its third image must be on exactly
the same canvas as the original. It does not align a raw torso reference or solve
a viewpoint change. Use it when you have manually warped/positioned a near-angle
photographic patch and want maximum preservation without another AI generation.

## Output

The save node writes one final PNG and matching companions:

- `final_....png`: final original-size composite with workflow metadata.
- `final_....audit.json`: pixel checks, mask settings, roles, exact prompt.
- `final_....edit_mask.png`: actual paste alpha; white replaces, black preserves.
- `final_....comparison.jpg`: base vs verified final, a review image only.

The tensor checks are exact, but PNG export is 8-bit; original JPEG encoding,
EXIF, ICC profiles and high-bit-depth file metadata are not preserved. Review
color in the intended viewing environment. A later full-frame generative upscale
or color adjustment invalidates pixel-identity guarantees; do not run one just to
make the final dimensions larger. The starting scene's dimensions are retained.

## Install on an existing Pod

Extract the download under `/workspace`, then run:

```bash
python /workspace/qwen21-torso-lock/qwen21_torso_lock/INSTALL.py
```

The installer first detects the running ComfyUI process and its actual user
folder. It adds only `qwen21_torso_tools` and four workflows under
`Qwen21 Torso Lock v1.0.0`. It does not replace native/identity helpers, alter model
paths, download weights, install Python packages, or stop the server.

If auto-detection cannot determine one server, use the paths appropriate to the
running container. For the native image's defaults:

```bash
python /workspace/qwen21-torso-lock/qwen21_torso_lock/INSTALL.py \
  --comfy-dir /opt/ComfyUI \
  --user-dir /workspace/qwen21_native/user
```

The old identity image's default user path is `/workspace/qwen21_identity/user`.
An explicit `DATA_ROOT` changes the user path to `$DATA_ROOT/user`.

Restart **ComfyUI**, not the stateless Pod. Use the existing service/process
restart mechanism or reload custom nodes through your existing management setup.
Do not terminate/redeploy a stateless Pod unless your files are backed up; the
new addon must be baked into the image or reinstalled after a fresh deployment.
Refresh the browser after restarting ComfyUI.

Then validate actual node schemas without generating anything:

```bash
python /workspace/qwen21-torso-lock/qwen21_torso_lock/scripts/validate_workflows.py \
  --server http://127.0.0.1:8188
```

## GitHub / future Pods

Two routes are included. Neither is needed just to try this on a running Pod.

**Defensive script:** From the extracted download, run:

```bash
python APPLY_TORSO_TO_REPO.py --repo /path/to/runpod-comfy-stateless
```

This expects the existing `qwen21_native` suite. It copies the addon into
`qwen21_native/addons/torso_lock` and adds its startup install, runtime schema
check, and tests inside the existing Torch-equipped Docker test step. It backs
up the touched files before writing. It does not commit, push or build.

**Ready-to-upload overlay:** The separate GitHub overlay ZIP contains those same
paths already patched against the previously supplied Native Suite v1.1.0. Copy
its `qwen21_native` folder into the repository root and merge/replace those files.
It is not a replacement for the original suite and is not for the old
`qwen21_identity`-only layout. Use the defensive script for a modified native kit.

No GitHub Actions workflow, model allowlist, precision, pinned upstream revision,
image tag or environment variable changes are required. Commit the actual new
files and start a new `Build Qwen 2.1 Native Suite` run on that commit. The addon
is installed at Pod startup and checked against that real server's schemas, not
a GPU-less BuildKit inference test.

## Validation boundaries

Local validation exercises Python helpers with real Torch tensors, exact pixel
locks, inward feathering, role numbering, generated graph consistency, additive
installation and integration with the original suite. Native-encoder delegation
is tested with a stub, not a loaded Qwen checkpoint. These checks do not substitute
for a Docker build, running ComfyUI frontend, or GPU generation using your photos.

See `docs/VALIDATION.md` for the recorded tests and remaining limits.

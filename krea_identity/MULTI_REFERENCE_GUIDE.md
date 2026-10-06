# Krea Identity 1.1: labeled references and two-person replacement

## Which workflow to open

| Workflow | Best use | Krea generations |
|---|---|---:|
| `05_Labeled_References_Directed_v1_1` | More references for one person, including selecting one target in a group photo | 1 |
| `06_Man_Woman_Single_Pass_v1_1` | Replace a man and a woman together with the simplest setup | 1 |
| `07_Man_Woman_Protected_Regions_v1_1` | Separate each identity's reference conditioning and protect the first edit/background during the second | 2 |

The filenames begin with `Krea_Identity_`. Workflows 01-04 remain available and their JSON files are unchanged. All new generation passes use the existing Krea 2 Turbo BF16 and Identity Edit v1.2. There is no custom subject LoRA training, new generative model, remote inference, or LLM prompt rewriting.

The two-person workflows have editable target descriptions. They do not detect gender, infer names, or assume that a man must always be on the left. Those are only starting examples.

## Add and label reference photographs

Each `Krea Identity - Labeled Reference (chain to add)` node has an image input and three controls:

| Control | Meaning | Example |
|---|---|---|
| `label` | Your name for this reference | `Side profile` |
| `use_for` | What Krea should take from it, and what it should ignore | `Nose profile, jaw shape and ear geometry only. Do not copy this pose or lighting.` |
| `include` | Include or omit this photograph from the reference list | `true` |

The first active reference is the primary reference. Put your strongest, tightly framed original face photograph first. With three or more active references it receives the largest panel. This is panel sizing, not an invented attention-strength control.

To add a photograph:

1. Duplicate a `LoadImage` node and a `Labeled Reference` node. Upload the new original photograph into the duplicated image node and connect its IMAGE output to the new reference node's `image` input.
2. Enter a unique label and its purpose. Connect the previous reference node's `references` output to the new reference node's `references` input.
3. Connect the new node's list output to the edit node. For workflow 06, choose the correct `references_a` or `references_b` input. For workflow 07, use the appropriate person's separate edit node.

There are no unfilled optional image loaders in the supplied graphs. Only add a loader when you have an image for it. To remove the default body photograph, connect the face reference list directly to the edit node. Alternatively, keep its valid uploaded image selected and switch its `include` control off. The upstream `LoadImage` still needs a valid file when it remains connected.

Up to **six active references per person** are supported by this integration. Two-person workflows therefore support up to twelve original reference photographs. This is an application limit, not a claim about a trained six- or twelve-reference model.

### Useful labels and purposes

| Label | `use_for` text |
|---|---|
| Primary face | Facial identity, eye spacing, nose/mouth shape, jawline and natural skin detail. Do not copy clothing or background. |
| Three-quarter face | Supplemental facial geometry at this angle. Preserve the scene's head direction and expression. |
| Body proportions | Build and body proportions only. Adapt these to the scene pose. |
| Hair | Hair color, length, texture and hairline only. |
| Wardrobe | This outfit and its material details only. Do not use this photograph's face as identity. |
| Detail | The visible tattoo or accessory only. Do not copy the reference pose. |

Start with two or three complementary photographs, not all six automatically. Additional panels share finite vision-encoder and latent resolution and can introduce conflicting information. Original face crops are preferable to retouched/generated proxies when likeness is the priority.

The helper constructs a letterboxed reference sheet from your original photographs. It does not stretch or crop them. Compact A1/A2/B1/B2 identifiers appear in separate margins. Your labels and purposes are inserted verbatim into an explicit panel map in the edit prompt. A PreviewImage node shows the exact sheet sent to Krea.

This retains the working two-input architecture: scene first, reference sheet second. It does **not** add six independent native image-conditioning slots. Labels are instructions, not guaranteed attribute isolation.

## Tell Krea who replaces whom

Workflow 05 has a `Who Replaces Whom` node. Describe both the target in the scene and the replacement identity.

```text
Target description:
The man standing on the left in the gray jacket, behind the seated woman.

Replacement description:
The man shown in reference group A.

Extra instruction:
Keep both women, their clothing, the chairs and the background unchanged.
```

Use visible position and clothing to disambiguate a target, especially when there are several men or women. The prompt preserves the scene's person count and tells Krea not to alter unselected people. It does not force a group photograph to become a photograph of one person.

### Clothing is a separate decision

`Keep scene clothing` is the default. `Use labeled wardrobe references` uses only photographs you explicitly label for clothing. A body reference whose purpose says "physique only" is not automatically a wardrobe reference. Change its purpose or add a wardrobe photograph when you intend to transfer the outfit. `Follow extra instructions` lets you write a specific clothing direction.

## Workflow 06: man and woman in one generation

Keep the man in reference group A and the woman in group B. Enter the actual target descriptions, for example:

```text
Target A: The man on the right wearing the dark shirt.
Replacement A: The man shown in group A.

Target B: The woman on the left wearing the blue dress.
Replacement B: The woman shown in group B.

Extra instruction: Keep the child between them unchanged.
```

Each group has its own labels and clothing choice. Krea receives the scene plus a combined reference board, with A on its left half and B on its right half. It generates both replacements together. This is the simplest two-person option, but the model can still blend or misassign identities or change other parts of the frame. Text-only preservation is not a pixel lock.

The starting canvas is 2 megapixels, with dimensions derived from the scene. When faces start blending or a split/collage output appears, compare 1.5 megapixels and reduce both grounded encoders from 1024 to 768. Keep the seed fixed when testing one change.

## Workflow 07: protected, separate-identity editing

Use this for tighter control over each identity and untouched scene pixels. The man and woman are generated in separate passes using only their own reference groups. Both passes use the same underlying Krea model and existing editor weights, but each gets a fresh model patch. The first person's reference conditioning is not stacked into the second person's patch.

The graph:

```text
Original full-resolution scene
  -> selected man crop + man's labeled references
  -> Krea generation
  -> composite only the man's edit region into the original
  -> selected woman crop + woman's labeled references
  -> Krea generation
  -> composite only the woman's edit region
  -> one final full-resolution scene
```

An intermediate image after the first replacement is also saved for inspection. Workflow 07 uses approximately 1.5 megapixels per selected crop, rather than dividing one full-scene working canvas between both people. It costs two generations. A crop can still include the other person as scene context, so prompt the target clearly.

### Configure the regions

Each `Protected Region` node has `left`, `top`, `width`, and `height` as fractions of the original image, from 0 to 1. The defaults cover the left side for the man and the right side for the woman. **They are placeholders, not automatic detection.**

The preview shows the editable region brightly and protected context dimly. Include the original and replacement silhouettes, hair, clothing and any shadows that need changing. A mask that cuts through a jaw, garment or shadow can create a seam. A mask that includes another person's face can change that face in the first pass.

`context_padding` adds surroundings to the generation crop without expanding the actual composite mask. `feather_pixels` softens inward from the mask boundary in original-scene pixels. Text target descriptions refer to this local crop; "the man in this crop" may be clearer than a full-scene left/right description.

### Optional painted masks

For people who are close together, duplicate a `LoadImage` node, load the **same original scene**, and use its Mask Editor to paint the target area. Connect its MASK output to the relevant Region node's `edit_mask` input. A painted mask overrides that rectangle. The IMAGE output of this additional loader is not needed.

White means edit; black means protect. The mask must have exactly the original scene's dimensions. An empty default mask or a mask from a differently sized image raises an error rather than being silently resized.

The second region's `protect_mask` is already connected to the first region's effective mask. Every pixel used by the first composite, including its feathered edges, is excluded from the second. This is a hard protection rule. When regions overlap, the first region takes priority. Tight interactions and crossing limbs need careful, nonconflicting masks; broad half-frame rectangles are only starting points.

This is **crop-based image editing followed by masked compositing**, not a claim that a diffusion inpainting mask is applied inside the sampler. Krea can alter surrounding crop content internally, but the composite discards those changes outside the selected region.

### What is actually preserved

The final image keeps the original scene dimensions. Pixels outside both effective edit masks are copied from the original scene. The first composite's pixels are protected during the second composite. CPU tensor tests verify these properties bit-for-bit before saving or optional global upscaling.

Keeping a 6000 x 4000 background does not mean Krea generated 24 megapixels of new person detail. New person detail is generated at the crop's working resolution, then mapped back into the original. The original background detail is retained rather than unnecessarily regenerated.

## Quality and the screenshot rebalancer

The starting recipe remains Turbo BF16, Qwen3-VL BF16, full Identity Edit v1.2 at 1.0, 12 steps, CFG 1, Euler/simple, denoise 1 on an empty latent, scene strength 1, subject strength 4 and FIT. Both grounded encoders start at 1024. `target_latent` is wired before sampling.

Every pass includes the same actual `ConditioningKrea2Rebalance` node and the screenshot weights:

```text
multiplier = 1.0
per_layer_weights = 1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

The switch starts off. Keep the fixed seed and enable it to compare. Workflow 07 has one switch per person. Multiplier 1 does not neutralize non-unity layer weights. No per-person independent attention-strength sliders are claimed for the combined sheet; use separate passes when independent tuning is needed.

All new workflows preserve the source orientation automatically, including horizontal/landscape, vertical/portrait and square scenes. References can have different orientations. Narrow grid-rounding margins are removed after generation instead of cropping away the original scene edges. Native output aspect is subject only to integer-pixel rounding. In protected mode, the final composite has exactly the original dimensions.

Use existing workflow 04 for optional SeedVR2 finishing only after approving identity. Global generative upscaling may change pixels and facial details, so the pre-upscale pixel-preservation guarantee does not apply to that optional result. Keep the native/composited original.

## Install

Merge the **contents** of `krea-identity-integration-update-v1.1.0.zip` into your existing repository. Include `.github` and `SOURCE_SNAPSHOT.json`. Do not replace whole directories with the smaller overlay directories. Your private `config/lora_links.txt` is deliberately absent from the update and must remain in the repository.

Commit the update and build **General/Image** from the new commit. Deploy the successfully built image. Keep your current RunPod settings, including:

```text
ENABLE_KREA_IDENTITY=1
KREA_IDENTITY_UPSCALE=1
```

No additional weights or upstream custom-node repositories are introduced. The helpers use PyTorch, NumPy, Pillow and SciPy, already present in the pinned ComfyUI requirements. A new Pod may still need to download the same existing weights when storage is ephemeral.

The installer adds the new versioned workflow files and does not overwrite existing user-edited workflow files. Reload ComfyUI after deploying the new image. Just importing the JSON into the old image is insufficient because the new local helper classes must also be installed.

## Validation and limits

650 local Python tests pass, including 77 new reference, target-mapping, aspect-ratio, crop, mask, compositing and graph-contract cases. Seven Krea graphs and 26 repository graphs pass static validation. The Docker build still requires a real ComfyUI node-registry smoke test for every Krea graph; the earlier SeedVR2 CPU schema guard is retained.

No full Docker build, live ComfyUI execution, RunPod deployment, model download, GPU render, speed/VRAM measurement or visual identity comparison was performed for this release. The new arrangements are designed for control and reference fidelity, but their visual performance with your photographs is not benchmarked. No custom subject LoRA was trained, and no changes were pushed to your live GitHub repository.

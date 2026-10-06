# H3 reference investigation and 1.5 release decisions

## Reported problem

With three references, the first two represented the intended subject and the
third was intended only for pose/camera/composition. Even with an explicit brief,
the generated MiniMax H3 video could resemble the third-reference person.

The exact three images, generated H3 prompt, selected private LoRA and resulting
clip are not part of this package, so the specific failure cannot be reproduced
or attributed to one cause with certainty.

## What the pinned native H3 path actually does

The automatic workflow ultimately calls the pinned ComfyUI
`MiniMaxH3ReferenceToVideo` implementation. That implementation iterates over all
visual reference images. Each supplied reference participates in the vision/text
presentation; with the video VAE connected it also becomes native H3 reference
conditioning.

There is no native per-image control that means "use this photograph for pose but
ignore the person's appearance." Therefore a prompt saying that a photo is
"pose only" is semantic guidance, not a hard exclusion of the guide image's
identity pixels. This gives a concrete mechanism by which an unrelated pose
reference can compete with subject identity.

Source:
https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py

## 1.5 video fix

The uploader retains an explicit role for each image. In
`Role-aware (recommended)` video mode:

- identity, face, body, hair, wardrobe and scene roles remain visual H3 refs;
- pose/camera and expression roles are analyzed by Ollama but their pixels are
  removed before native H3 Ref2VA conditioning;
- the remaining visual refs are renumbered and the generated prompt is remapped;
- a routing ledger is shown before generation.

This does not create a pose ControlNet or guarantee exact pose replication. It
removes one source of identity competition while preserving the guide as text.

## Still-image request

The Lite release also needs a single-image workflow that can combine multiple
subject photos with an optional pose/camera or scene reference while keeping
MiniMax H3 Ref2VA as the actual generator.

The pinned native H3 node supports a minimum target length of five frames. The
new workflow therefore asks H3 to render that minimum packet, decodes all five
frames, selects one frame, and saves only the selected image. It does not add a
separate image-generation model.

A community H3 Image Studio project independently demonstrates the same general
idea of using H3 for still delivery from a short frame packet. This release does
not install or copy that project; it uses the pinned native ComfyUI H3 path.

Reference:
https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio

## Still safe-swap routing

For `H3_Portrait_Image_Lite_v1_5`, the default routing is stricter than video:

- identity / face / body / hair / wardrobe remain visual refs;
- pose/camera, expression and scene roles are **text-only**;
- Ollama still analyzes every upload, including the guide, and translates its
  geometry/environment into the H3 prompt;
- native H3 sees the desired subject appearance refs but not the unrelated
  pose/scene person's pixels.

`All images visual (comparison)` is retained as an explicit opt-in when direct
visual composition conditioning is more important than identity isolation.

## Still resolution decision

The default still canvas targets about one megapixel, close to the ordinary H3
native-detail area. `High-res (~2 MP, experimental)` asks H3 itself to sample a
roughly two-megapixel 32-pixel-grid canvas. It is not a post-generation upscale.
The larger canvas is intentionally labeled experimental because larger sampling
can cost substantially more VRAM and does not guarantee better identity/detail.

The release offers:

```text
High fidelity: 20 steps, reference size max
Standard:      16 steps, reference size match
Fast preview:  12 steps, reference size match
sampler:       res_multistep
scheduler:     simple
length:        5 frames
```

## Workflow-default bug

The screenshot showing `mode = 0`, `prompt_variation = NaN`, and LoRA values
shifted to `1` was caused by a serialized-widget alignment issue. ComfyUI creates
a companion `control after generate` widget for the seed. The workflow now stores
that value explicitly as `fixed`, and tests compare the serialized values against
the actual node schema plus the frontend seed companion.

## Exact final-frame save

Video workflows save the MP4 first, then decode the completed file and save its
last decoded frame as a matching `_last.png`. This gives a real exported-video
frame for future chaining work rather than assuming the pre-encode tensor is
identical to the file's final decoded frame.

## Limits

Local tests establish source consistency, graph structure, routing behavior,
widget serialization, pure still geometry/frame selection, and the exported
last-frame helper. They do not establish GPU image quality, peak VRAM, render
speed, or private-LoRA compatibility. Those require the published Docker image
and live RunPod tests.

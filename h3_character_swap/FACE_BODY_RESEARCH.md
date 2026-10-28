# Face/body low-memory update: evidence and design decisions

Reviewed 2026-10-27 for jsanso1497/runpod-comfy-stateless commit
073d7404836b5fab27ce754fcfbcb937975910c1.

## Source verification

The H3 subtree was reconstructed from the delivered baseline and prior patches,
then its Git tree was computed locally. It matched the live GitHub H3 tree exactly:
`3ff6dc96068160af6402578b9c63572d192771d7`.
The source audit is scoped to this H3 subtree, not the unrelated Qwen/FLUX workspaces.

## Upstream facts

Native ComfyUI reference guidance documents separate `<Picture N>` inputs and
explicit role assignment. The implementation pinned by this build has up to nine
native image inputs and loops through them in order. With the VAE connected, it
encodes each reference into an image-latent block as well as a text/vision entry.
Its schema exposes `match` versus `max` sizing, not individual face/body strength
sliders or hard regional assignments.

- https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
- https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy_extras/nodes_minimax_h3.py

Akatz's V1.5 guidance uses one character image in the first picture slot and retains
its matched eight-step Turbo recipe. The developer explicitly warns that identity,
expression, source timing, occlusion and shot cuts can drift. It does not validate
a separately enforced face-photo/body-photo conditioning method.

- https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA

The pinned Relay method already carries 17 preceding generated frames and one
boundary frame. This update leaves its source/history layout unchanged.

- https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/h3_relay/vendor/context_loop/sliding_context.py

## Integration choices, not upstream guarantees

Picture 1 supplies face/head/hair; Picture 2 supplies physique/proportions/wardrobe.
This preserves the user's working face-first order, while describing full-body
replacement equally explicitly and excluding competing head identity from image 2.
The runner does not accidentally turn the second photo into a second person or
assign clothing from the face close-up. These are prompt-directed semantics only.

Both references enter every separate prompt, including retries and resumed chunks.
The code keeps ordered connections to `ref_images.ref_image_0` and `ref_image_1`,
keeps the VAE and `ref_image_size=match`, and verifies immutable per-run snapshots.
It neither silently merges photos into a board nor changes sampler/model pins.

The long analyzer is extended with a new two-photo node. Its image ordering,
system instructions, context reserve and cache key all account for the body image.
The original single-photo node and graphs are retained. Its selected-clip draft
is explicitly distinguished from the chain-safe appearance draft.

No prompt wording alone is presented as a guaranteed solution. The initial test
compares 10 seconds with the same source, masks and seed, followed by a new long run
only after face, body, non-target preservation and the continuation join are reviewed.

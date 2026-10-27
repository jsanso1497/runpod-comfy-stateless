# Continuity research and integration decision

Reviewed October 26, 2026. Distinguish developer documentation, inspected source
code and this package's engineering proposal. No independent benchmark on the
user's footage was available. Search snippets claiming "seamless" were not
accepted as proof of a perceptual guarantee.

## Current repository

The user's `main` is `8cefbc9c0093ca2c0b6d85c2d2a8dd63542d7948` (October 22).
This package is built against that source, including all four prior build fixes.
The previous October 26 runner was an uncommitted standalone attachment, not an
installed GitHub feature. The older 120-second JSON still materialized the full
source/masks and accumulated output in one execution.

## 1. Preserve the method tested for this character adapter

Developer model card:
https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA

Akatz continues to recommend V1.5 and the native Relay recipe: Ref2VA INT8,
its character adapter and matching FL2V Turbo8 adapter at strength 1, eight
steps, CFG 1, er_sde/normal. The developer describes identity, timing, occlusion
and cut drift as continuing limitations. It is not a universal quality guarantee.

Pinned, inspected implementation and contracts:
https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/docs/toolkit-v2.md
https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/h3_relay/planning.py
https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/h3_relay/toolkit.py
https://github.com/akatz-ai/h3-relay/blob/e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220/h3_relay/vendor/context_loop/sliding_context.py

The inspected code explicitly describes an 18-frame handoff as 17 preceding
history frames plus a boundary anchor. The explicit segment schedule is intended
for separate app jobs with supplied history; it excludes that supplied history
from delivered output. Its decoder boundary is regenerated and trimmed once.
The source documentation describes separate segment jobs, lossless history
masters, durable records and original audio in the developer app.

Decision: reuse these pinned nodes and their segment schedule. Do not add a
second arbitrary first-frame guide on top of the existing continuation guide.
The new runner changes storage/residency and review, not model inference logic.

## 2. Native masked latent continuation is a distinct method

Primary developer sources reviewed:
https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum
https://github.com/xmarre/ComfyUI-H3-Continuum-Plus
https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite/blob/main/README.md

These projects describe direct latent-tail reuse with native zero-denoise prefix
masks, compatible Core support and explicit temporal/audio alignment. This is
stronger preservation of the reused representation than decoding/re-encoding a
reference. However, it is NOT evidence that novel generated movement after the
protected span is always smooth. Their temporal contracts differ from Relay's
18-frame sliding-history path, and some examples use FL2VA rather than this
source-masked Ref2VA character-swap recipe.

Decision: do not silently splice their 22/39-frame contracts into the user's
working Akatz graph or upgrade the entire ComfyUI runtime speculatively. A direct
latent-prefix experiment should be a separately qualified comparison. This
package does not claim to implement that alternate continuation technique.

## 3. Memory must be bounded outside the execution graph

Pinned Core implementation already inspected in this conversation:
https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/main.py
https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy_extras/nodes_loop.py

Official API routes:
https://docs.comfy.org/development/comfyui-server/comms_routes

A loop with small sampling windows can still retain a full source, masks or
accumulated outputs. /free is handled by the prompt worker between jobs, not a
hard reset in the middle of an ordinary running JSON graph. The updated control
node launches an external local process and immediately returns, avoiding the
deadlock of waiting inside a node for its own queued prompts. Cleanup uses the
existing /free API plus two lightweight worker fences before a later generation.

## 4. Preserve timing and inspect the actual delivered seam

Official FFmpeg sources:
https://ffmpeg.org/ffmpeg-formats.html#concat
https://ffmpeg.org/ffmpeg-filters.html

Use frame-index boundaries on one 24 fps source timeline. The extractor seeks to
a whole second, then trims by frame index; tests compare actual decoded frame
content, not only counts. This avoids fractional timestamp rounding. Drop the shared boundary
once, and assemble accepted masters without a growing IMAGE batch. Keep the
continuous original soundtrack rather than repeatedly stitching compressed chunk
audio. H.264 delivery is still lossy; saved float checkpoints and 16-bit masters
are separate from that delivery encoding.

## What is new engineering, not a published "best method"

The float32 checkpoint files, raw-mask carryover, source-aware thumbnail/block
motion gate, bounded automatic retries and authenticated review page are this
integration's additions. Thresholds have synthetic regression coverage but are
not calibrated on real H3 output. They can miss defects or flag good motion.
The review workflow makes uncertainty visible rather than labeling every
numerically accepted transition seamless.

The practical recommendation is: start with a short continuous shot in Review
every chunk mode, approve identity and full-body masking, inspect each join, then
try Auto check + retry. Do not claim that a successful JSON/static/media test
establishes GPU compatibility, output identity or invisible transitions.

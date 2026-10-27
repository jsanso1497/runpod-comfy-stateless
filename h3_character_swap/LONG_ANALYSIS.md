# Akatz long-form prompt analysis

Release: 2026-10-27.1. Incremental update against GitHub commit
`ef4aabdb77e35b9b5e04e233e2256df1fcdc7182`.

## What changed

The new `H3SwapAnalyzeAkatz` node replaces the analyzer on the factory
`00_Analyze_Clip_Build_Prompts` canvas. It is Akatz-only. It does not run BFS,
H3 sampling, an upscaler, or a different language model.

| Control | Default | Available |
| --- | --- | --- |
| Detail level | Exhaustive | Detailed / Very detailed / Exhaustive |
| Maximum output tokens | 16,384 | 1,024-32,768 |
| Total helper context tokens | 32,768 | 16,384 / 32,768 / 65,536 |
| Source frames sampled for analysis | 16 | 8-32, no more than frames in the selected interval |
| Analysis timeout | 1,800 seconds | 120-3,600 seconds |
| Reference layout | Single character image | Or Face-left / body-right sheet |
| Selected interval | 124 frames at 24 fps | Existing 22-362, 17n+5 frame grid |

The previous analyzer used 1,800 output tokens, asked for concise text, and
inspected eight samples. The new default gives roughly 9.1 times that response
budget and asks for approximately 3,000-5,000 evidence-supported words across
face/hair, body, wardrobe, scene, action observations, camera/light, other
people/objects, and uncertainty. This is an instruction and ceiling, NOT a
promise that the model will write exactly that amount. Shorter supported
observations are preferable to invented details or repetition.

The character reference keeps the prior 768-pixel maximum edge. Source samples
use a 576-pixel maximum edge. No face-resolution reduction is needed for the
larger output allowance.

The model remains local `qwen3-vl:8b`. The new analyzer sends its context and
output settings explicitly to Ollama. It does not reintroduce the previous
262,144-token allocation or a 32B FP16 helper. The server fallback becomes
32,768; a saved legacy node still sends its own smaller per-request setting.

For an even larger response, select **32,768 output tokens and 65,536 context**.
Use that only after the default works. Higher context uses more memory, and a
larger output allowance can make analysis take longer. The settings validator
reserves room for the sampled images and instructions; its reserve is a
conservative estimate, not an exact tokenizer calculation.

## Installation and first use

Apply the supplied local update, commit and manually push in GitHub Desktop.
Start a NEW **Build H3 character replacement** run and deploy its successful
image digest. Keep `/workspace` and the existing model files. There is no local
Python upgrade requirement.

In the new Workflows folder, open:

`H3_Character_Replacement_2026-10-27.1/00_Analyze_Clip_Build_Prompts`

Select the original video and one character reference or one combined sheet.
Specify the ORIGINAL person to replace, and choose the same selected interval
as the native render. For multiple people, be explicit about clothing/position.
Set `reference_layout=Face-left / body-right sheet` for that exact sheet layout.
Leave the larger default analysis controls for the first trial and queue once.

The old `H3SwapAnalyzeClip` node is retained for saved-workflow compatibility.
**Opening an old saved canvas does not upgrade its 1,800-token analyzer.** Use
this new factory canvas, not just the old workflow in a refreshed browser.

## Which output to use

**AKATZ SELECTED CLIP PROMPT:** detailed identity and the observed scene/action
sequence, with timestamps relative to the selected interval. Copy it to the
native Akatz prompt only when rendering that same interval. A trimmed file's
start is zero; do not analyze one section and render another.

**AKATZ CHAIN-SAFE PROMPT:** the same detailed reference identity/physique/outfit,
plus instructions to follow whichever source segment is currently supplied.
It intentionally excludes the sampled interval's timestamps, scene details,
and camera/action trajectory. Copy this to the long-chain control's prompt.
It will normally be shorter than the clip prompt because it does not pretend
to have analyzed every later section.

**ANALYSIS REPORT:** actual Ollama `eval_count` and `prompt_eval_count`, stop
reason, word/character counts, elapsed time, options, model digest, sample
timestamps, uncertainty and paths to complete saved drafts. Ollama response
token counts include the structured evidence response, not the exact token
count that H3 will encode from the assembled prompt.

**SOURCE CONTACT SHEET:** all sampled frames, not only the first eight.
Audio is not analyzed. Sparse samples cannot establish hidden movements,
exact action velocity, dialogue or content between frames.

The parser and output storage no longer impose the old 8,000-character field
limit on this new analyzer. The chain's application-side prompt limit is
160,000 characters instead of 20,000. This is a storage/validation bound,
NOT a measured GPU-safe H3 token budget. No draft is silently shortened.

## Cached drafts and incomplete responses

Drafts live in:

`/workspace/output/h3_character_swap/prompt_drafts_long/`

Each successful analysis saves the entire JSON evidence/report, `_akatz_clip.txt`,
`_akatz_chain.txt` and `_source_samples.jpg`. The cache key covers model digest,
source identity/timing, reference pixels AND dimensions, sampling, detail,
context/output limits, layout, wardrobe and analyzer version.

Changing the token budget or detail level makes a new draft. Turning off
`reuse_cached_draft` performs a new request every time the node runs, rather
than being bypassed by ComfyUI's execution cache. Identical requests with a
fixed sampling seed may still produce identical wording.

If the output reaches its token/context limit, the node raises a visible error
and retains `_partial.txt` and `_failure.json`. It does not use incomplete JSON
as an H3 prompt or silently increase the memory allocation. Raise the appropriate
budget, choose fewer sampled frames, or lower the requested detail level.
A malformed cached result is discarded and regenerated rather than blocking use.

Ollama is asked to unload after success, failure and cancellation. Cleanup errors
are reported without hiding the original error. Analysis refuses while the
low-memory controller holds its job lock, so it cannot unload models under an
active chain. A paused chain can be analyzed before resuming.

## Other fixes from the audit

- Anamorphic source samples now use display proportions instead of changing
  only the pixel aspect-ratio label after scaling. This was tested with FFmpeg.
- Cropped VIDEO wrappers are explicitly rejected by the new analyzer. Export
  the crop first or connect Load Video directly; reading the underlying raw
  file must not silently ignore an active crop.
- The resume path clears a stale error message before a new attempt, so a
  successfully resumed run is not still labeled with an old failure.
- Default Akatz-only workflow synchronization hides BFS from the new factory
  folder. Existing saved BFS graphs, nodes and model files are not deleted.
  The unchanged image build retains optional BFS metadata/templates for legacy
  compatibility; `H3_SWAP_VARIANTS=akatz` prevents BFS weight preparation.

## Environment and what is not changed

```
H3_SWAP_VARIANTS=akatz
H3_SWAP_PROMPT_ASSIST=1
H3_SWAP_PROMPT_MODEL=qwen3-vl:8b
H3_SWAP_DATA=/workspace
H3_SWAP_MAX_UPLOAD_MB=1024
WB_AUTH_MODE=basic
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Remove `H3_SWAP_BFS_LICENSE_REVIEWED` when not using BFS. Token settings are on
the NEW analysis node; no new environment variable is needed to make it longer.
Do not set `OLLAMA_CONTEXT_LENGTH=262144` or switch helper models as part of this update.

Model selections, dependency pins, Akatz sampling, frame history, mask geometry,
resolution limits, Qwen/SAM images, and the continuous source-audio path are unchanged.
No 1080p/SeedVR finishing is added. The automatic FileBrowser add-on was NOT
present at the reviewed commit and is NOT silently included in this focused
update. Your existing manual FileBrowser and port exposure are unchanged.
Do not apply an older pending package over this update without reconciling files.

Longer prompts can increase H3 conditioning/sampling memory and time. The disk
chain bounds VIDEO accumulation; it does not make unlimited text cost-free.
Test one short generation before running a long shot with a very long prompt.
Long text cannot enforce identity, full-body transfer or seamless motion by itself.

## Verification boundaries

The H3 source subtree was reconstructed from the previously delivered files and
verified against the live Git tree hash, not guessed from old documentation.
Local checks exercise request options, strict parsing, complete long-text storage,
cache invalidation, cleanup/error paths, controller exclusion, workflow wiring,
source sampling and existing disk-chain tests. Ollama responses and H3 inference
in tests are fixtures. No claim of generated quality follows from them.

No full Docker build, actual Ollama generation, interactive ComfyUI browser import,
GPU execution or peak-memory measurement was performed in this authoring session.
The existing GitHub built-image CPU startup/schema check is retained. Local edits
are not a published image until that build succeeds and you deploy its digest.

# RunPod ComfyUI: complete repository

**Start with [START_HERE.md](START_HERE.md).** This snapshot consolidates the
general ComfyUI setup, H3 Portrait, shared link-only LoRA library, and
build/runtime fixes supplied in this conversation.

| Image tag | Build workflow | Purpose |
| --- | --- | --- |
| `h3-portrait-clean` (also `h3-portrait`) | Build H3 Portrait template | H3 Portrait 1.3: natural-language direction + references + two-pass Ollama + selected LoRA + portrait video |
| `latest` | Build RunPod ComfyUI image | Existing Krea-only Rebalance, SeedVR2, H3 and optional RefMod workflows |

Both builds are independent. For portrait-only use, build only the portrait image.
The portrait Dockerfile uses a pinned, previously published foundation; it does
not fetch the current `latest` build as its base.

## What you edit

- `config/lora_links.txt`: your own Civitai/Hugging Face LoRA links, one per line.
- ComfyUI's instruction field: the video you want, written in ordinary language.
- ComfyUI's aspect, quality, seconds and LoRA dropdowns: each generation's controls.
- RunPod Secrets: tokens, never committed to GitHub.

## Current portrait implementation

H3 Portrait 1.3.0 keeps the same uploader, UI controls and render presets.
Pass 1 uses a separate abliterated 32B Instruct Q4 model for a compact reference
map. Pass 2 uses abliterated 32B Thinking Q4 for the MiniMax Full-Reference prompt,
without a second image pass. This avoids the unsupported assumption that the
Qwen3-VL Thinking edition is a hybrid with a reliable thinking-off toggle.
The original images still condition H3. JSON is validated, malformed/truncated
outputs receive at most one bounded regeneration with the Instruct model. Models
are unloaded between editions, and Ollama must unload before video generation.

| Preset | Steps | Ref sizing | 9:16 output | 2:3 output |
| --- | ---: | --- | --- | --- |
| Preview | 12 | match | 576 x 1024 | 576 x 864 |
| Standard | 20 | match | 756 x 1344 | 768 x 1152 |
| High fidelity | 25 | max | 756 x 1344 | 768 x 1152 |

No Turbo/Lightning adapter is automatically added. Standard 9:16 crops six
pixels from each side of the model's 768x1344 canvas, not a geometric stretch.
Five seconds resolves to 124 frames at 24 fps, approximately 5.17 seconds.
These settings are not a measured optimum or a guarantee of identity retention.

## Build integrity

This clean reset is snapshot `h3-clean-2026-10-05-r1`. Both Actions first run
`tools/verify_snapshot.py`, which compares the exact packaged file set and hashes
against SOURCE_SNAPSHOT.json, with `config/lora_links.txt` deliberately editable.
Use the Desktop clean-replacement instructions in START_HERE.md instead of merging
old browser-upload folders.


`tools/check_repository.py` verifies critical source paths before the Docker pull.
`h3_portrait/verify_release.py` checks the intended implementation, creates hashes
inside the build, and verifies image and running-node copies before downloads.
The build summary and startup logs expose the exact source commit. No user-edited
hashes or per-LoRA metadata fields are needed.

The GitHub runner cleanup, diagnostic reserve, live disk guard and disabled
external cache are retained. The actual CPU ComfyUI schema/startup checks remain.

## Important scope

This is the intended complete source snapshot, reconstructed from supplied files.
The private live repository could not be read. Copy YOUR existing LoRA links into
this ZIP before upload. Git history, tokens, outputs, RefMod data and model weights
are not included. Retain the existing GHCR base images and registry credentials.

See [SOURCES.md](SOURCES.md) and [VALIDATION.md](VALIDATION.md). Models, community
adapters and node packages keep their respective licenses; this repository does
not supply additional model-use rights. The general and portrait endpoints do
not gain a new login or file-browser service from this consolidation.

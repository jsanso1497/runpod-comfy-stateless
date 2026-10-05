# H3 Portrait 1.2: 32B Thinking + two-pass prompting

This is a patch for your working **H3 Portrait - Automatic** template and its
shared, link-only LoRA library. Do not install earlier ZIPs after this one.

## 1. GitHub: upload only the included h3_portrait folder

Repository: `jsanso1497/runpod-comfy-stateless`, branch `main`.

Unzip the patch. On the repository Code page, upload the included `h3_portrait`
FOLDER into the repository root, merge it with the existing folder, and replace
matching files. Do not upload the ZIP or put files alongside the root Dockerfile.

Existing files replaced:

```
h3_portrait/Dockerfile
h3_portrait/settings.json
h3_portrait/start.sh
h3_portrait/ollama_service.py
h3_portrait/node/__init__.py
h3_portrait/node/logic.py
h3_portrait/node/ollama_client.py
h3_portrait/node/system_prompt.txt
h3_portrait/tests/test_portrait.py
```

New files added:

```
h3_portrait/node/analysis_prompt.txt
h3_portrait/tests/test_thinking.py
h3_portrait/THINKING_START_HERE.md
h3_portrait/THINKING_VALIDATION.md
```

Commit to main. Open **Actions -> Build H3 Portrait template** and wait for the
run for your final commit to succeed. Do not rerun an older commit.

Keep your current root Dockerfile, both Actions workflows, .dockerignore,
shared_loras folder and its corrected 69-test file, config/lora_links.txt, all
model/LoRA catalogs, existing workflows and earlier fixes UNCHANGED. The patch
does not include any of them. No manual SHA, profile or LoRA metadata entry is added.

No new Python dependencies or inference weights are installed during the build.
The existing pinned ComfyUI/base image and real CPU startup/schema check are retained.
The existing portrait Dockerfile automatically discovers the new regression tests.

## 2. RunPod: change ONE environment-variable value

In your **H3 Portrait - Automatic** template, replace the current `OLLAMA_MODEL`
value with:

```
huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
```

This replaces the old 8B Instruct value. Do not add a second duplicate variable.
It is essential: a template environment variable overrides the image's default.
The new code rejects a model that lacks vision/thinking support instead of silently
running the old Instruct configuration.

Keep the container image exactly:

```
ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait
```

Keep your existing GPU, 300 GB container disk, 0 GB volume disk, port 8188,
GitHub GHCR registry credential, ASSET_DOWNLOAD_WORKERS, HF_TOKEN and CIVITAI_TOKEN
references unchanged. Do not add think/temperature settings as environment variables.
They are already configured in h3_portrait/settings.json. Do not expose port 11434.
Your other :latest template is not changed by this patch.

After the portrait build is successful, download any images/videos/prompts you
need from the current Pod. Deploy a NEW Pod from the updated template. Changes
in GitHub or the template do not replace code/models in an already-running Pod.
Stopping/terminating an ephemeral Pod can remove local files; save them first.

## 3. Verify startup and use your existing workflow

Look for:

```
H3 PORTRAIT 1.2 | Two-pass thinking | Shared link-only LoRAs | Native H3 | No OmniNode
H3 PORTRAIT ASSETS READY
H3 PORTRAIT SCHEMA PASS
H3 PORTRAIT OLLAMA READY: huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
```

The last line also reports thinking support and two-pass prompting. The new
helper's published download size is approximately 21 GB; allow it to finish.
Startup checks advertised capabilities, not an inference/quality benchmark.

Open **H3_Portrait_Auto** in ComfyUI. No workflow import, new node, manual system-prompt
paste or reconnection is needed. The node's input names, order and types are unchanged.

Upload/reuse your reference images, type the ordinary-language instruction, and
select your H3 LoRA as before. Use these first-run settings in **2. Tell Ollama what
you want**:

```
aspect: 9:16 (or 2:3)
quality: Standard
seconds: 5
seed: 42
prompt_variation: 0
mode: Draft only
```

Keep your selected LoRA and its intended strength. The selected LoRA's trigger
words are still provided to the writer. Run once and review **What Ollama wrote**.
This writes a prompt but blocks the H3 generation job. Then change only mode to
**Generate video** and Run. With unchanged references/instructions/LoRA triggers/
aspect/duration/prompt_variation, the node reuses the cached draft and generates
instead of repeating the two Ollama calls.

You can also start in Generate video for one-click operation. No compulsory review
or copy/paste gate was added. Changing prompt_variation requests a new draft;
changing only seed or quality reuses the draft and changes the video render.

Fresh prompt generation logs:

```
H3 PORTRAIT PASS 1/2: reference analysis with thinking.
H3 PORTRAIT PASS 2/2: MiniMax guide prompt writing with thinking.
H3 PORTRAIT: Ollama unloaded; GPU handoff clear.
```

## What changed internally

- Both calls explicitly send `think: true`. Only final JSON content is used;
  thinking-channel text is not logged, saved, or passed to H3.
- Pass 1 inspects all selected images (aspect-preserving PNG previews at up to
  1024 pixels on their longest side), maps subjects and priorities, and checks
  conflicts. Explicit user roles take priority. Essential unresolved conflicts
  stop with one clarification question rather than proceeding to H3.
- Pass 2 sees all the same images, your original brief, and the validated map.
  It writes the MiniMax narrative fields; code carries the map forward unchanged.
  No second reference array can replace the analysis map.
- The director instructions teach MiniMax's six sections, stable reference labels,
  retention terminology, temporal shot descriptions, sound separation, and its
  normal 350-500-word detailed_description guidance. This is a custom adaptation,
  not a copy of the full guide or a guarantee that the model will obey every rule.
- H3 receives the original reference image tensors, not the reduced Ollama previews.
  Its full BF16 weights, Standard/Preview/High fidelity presets, LoRA application,
  exact portrait output handling and native reference conditioning stay unchanged.
- The LLM is held for up to five minutes between the two serial calls to avoid
  reloading its weights. The final call requests immediate unload; a finally block
  explicitly unloads and checks /api/ps before the H3 job can be returned. Failure,
  cancellation, malformed final JSON or incomplete output cannot start H3.
- One JSON-repair attempt is allowed per pass, not unlimited retries. Complete
  truncation/transport errors stop instead of starting an expensive invalid render.
- The prompt cache includes both instruction files, model digest, sampling settings
  and the actual request. Editing system_prompt.txt or analysis_prompt.txt now
  invalidates the prompt on next execution; they are not frozen at Python import.
- Prompt record JSONs include the structured reference analysis and stage timing/
  completion statistics. These are not the model's private thinking text.

Configured in settings.json (already applied by this patch):

```
think: true
context_length: 32768
image_max_edge: 1024
temperature: 0.25
top_p: 0.9
top_k: 20
analysis_max_output_tokens: 8192
max_output_tokens: 12288
request_timeout_seconds: 1200
```

The two token limits budget both reasoning and the final answer for their respective
calls. They are not word counts or a guaranteed amount of deliberation. Temperature
0.25 is our reference-mapping tuning choice, not a measured optimum or the model
publisher's recommended default. More thinking/two passes can take longer. The
21 GB download is not total VRAM consumption: vision/context buffers also need memory.
Neither the model change nor the structured map guarantees identity fidelity.

## Sources checked for this patch

- Model tag/capabilities/size: https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
- Thinking request/response separation: https://docs.ollama.com/capabilities/thinking
- Chat API, formats and keep_alive: https://docs.ollama.com/api/chat
- Vision with structured output: https://docs.ollama.com/capabilities/structured-outputs
- MiniMax guide: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

See THINKING_VALIDATION.md for tests and remaining deployment checks.

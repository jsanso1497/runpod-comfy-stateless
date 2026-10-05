# Complete repository: 2026-10-05 consolidation

**Use this complete ZIP instead of all earlier update ZIPs and live-hotfix scripts.**
This is the source repository for `jsanso1497/runpod-comfy-stateless`.
It includes both Docker builds and the working general ComfyUI workflows.
The current portrait implementation is **H3 Portrait 1.3.0**.

## First: keep your own LoRA links

The private GitHub connector returned Not Found. This package was reconstructed
from the supplied source packages and fixes, not exported from your live main branch.
Its `config/lora_links.txt` has comments only. Your personal links were not available.

1. Download a backup of your current repository on GitHub: Code > Download ZIP.
2. Copy your EXISTING `config/lora_links.txt` into this extracted package at the
   SAME path, replacing the comments-only file. Do this before uploading.
3. Keep any other changes you independently made and need. They could not be
   compared with the live repository. No credentials or model weights belong here.

## Upload the complete repository

The ZIP opens directly at the repository root, with `Dockerfile`, `config/`,
`h3_portrait/`, `scripts/`, `shared_loras/`, `tests/`, `tools/` and `.github/`.
Do not upload the ZIP itself or add an enclosing folder around these paths.

On your repository's main branch, use Add file > Upload files. Upload folders
intact, merging and replacing matching files. To stay under GitHub's 100-file
per-upload limit, use these two batches:

1. `config/`, `scripts/`, `tests/`, `shared_loras/`, `tools/`, and the loose files
   at the repository root. Commit.
2. The ENTIRE `h3_portrait/` folder and the ENTIRE `.github/` folder. Commit.

If the file chooser hides `.github`, open the existing GitHub files below and
replace their complete contents using the included files. Keep both names:

```
.github/workflows/build-image.yml
.github/workflows/build-h3-portrait.yml
```

Also ensure `.dockerignore`, `.gitignore`, and `.gitattributes` exist at the root.
Use Add file > Create new file with the exact name when a hidden file will not upload.
Do not copy fragments of Python files from previews; upload the actual files.

Delete the known obsolete files below if they remain after the folder merge:

```
test_links.py
config/workflows/30_Ideogram4_Rebalance_Reference_Quality.json
```

Keep `shared_loras/tests/test_links.py`. That is the correct test file. There is
no need to delete the repository, its history, GitHub package, tags, or secrets.
Unrelated old files are not deleted automatically by a folder upload.

## Build the image you actually use

After ALL files are uploaded, go to Actions > **Build H3 Portrait template**.
Use the run associated with the final commit, or Run workflow > main after the
complete upload. Do not use Re-run on a previous commit's failed job.

Only this portrait build is required for your current use. **Build RunPod ComfyUI
image** is separate and is only needed when you also want the normal `:latest`
template updated. Neither image build depends on running the other one first;
both retain their known, immutable base-image references.

A successful portrait run must print:

```
H3 PORTRAIT SOURCE VERIFIED: 1.3.0 | split-model-reference-v3
H3 PORTRAIT SCHEMA PASS
```

Its final summary is **H3 Portrait 1.3 image published** and includes the source
commit and published image digest. No one needs to hand-edit a SHA field.

The build creates a release manifest automatically and checks that the code in
`/opt/h3-portrait/node` and `/opt/comfy-bundle/...` matches. Startup checks the
active `/workspace` copy too, before the large model downloads. Missing or older
prompt files stop the new build or startup; they are not silently accepted.

## RunPod: your portrait template

Keep these settings:

| Field | Value |
| --- | --- |
| Image | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait` |
| Container disk | 300 GB |
| Volume disk | 0 GB |
| Network volume | None |
| HTTP ports | 8188 |
| TCP ports | Blank |
| Container start command | Blank |
| Registry credential | Your existing GitHub GHCR credential |
| GPU | Keep the type you are already using for H3 |

Your complete CUSTOM environment-variable list is:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
```

Keep the exact existing secret references when you used other names for those
secrets. The actual token values stay in RunPod Secrets.

For THIS portrait template remove old `CONFIG_REPO`, `CONFIG_REF`, `COMFY_ARGS`,
`MODEL_PROFILES`, `OLLAMA_MODEL_DIGEST`, and `RESTORE_*` custom overrides if listed.
Do not remove RunPod's automatically generated environment variables. No new
thinking or temperature variable is required. Do not change your OTHER template.

Port 8188 is ComfyUI. **There is no file-browser/Jupyter service on 8888 in this
repository.** Do not add that port expecting it to provide file access. Do not
expose 11434; Ollama is local to the container. Web Terminal is still available
through RunPod, subject to RunPod's current interface.

## Deploy after the portrait build succeeds

Save outputs, prompts, and uploaded references from the current Pod before
stopping or terminating it. Deploy a NEW Pod from your portrait template after
the new image is published. Do not expect an existing running Pod to update.

The new Pod must print:

```text
H3 PORTRAIT 1.3 | Instruct analysis + Thinking director
H3 PORTRAIT RELEASE VERIFIED: 1.3.0 | source=<the new Git commit>
```

The banner has additional text after the prefix. Confirm the source commit agrees
with the successful portrait build summary. A 1.1 or 1.2 banner is NOT this release;
do not assume a successful image pull proves which source files are inside it.

Wait for:

```text
H3 PORTRAIT ASSETS READY
H3 PORTRAIT SCHEMA PASS
H3 PORTRAIT OLLAMA READY
```

Open Connect > HTTP Service 8188, then `H3_Portrait_Auto`.

## Use your existing workflow controls

1. Upload 1-9 reference photos with the uploader. Check the visible order.
2. Describe the people, image roles, action, outfit, scene and camera normally.
   No `BELONGS TO`, reference grammar, OmniNode, or RefMod is involved.
3. Choose `9:16` or `2:3`; start with `quality=Standard`, `seconds=5`, `seed=42`.
4. Select your matching H3 LoRA in `lora_1`, set its author's strength, and keep
   `use_loras=true`. Leave `lora_2=(none)` unless deliberately combining adapters.
5. Start in `mode=Draft only`, `prompt_variation=0` for this update's first test.
   Review **What Ollama wrote**, then change only `mode=Generate video` and Run.
   Direct Generate video remains the one-click option.

## Corrected prompt models: no unsupported thinking toggle

The prior advice to use `think=false` on Qwen3-VL's Thinking edition was wrong.
Ollama's maintainers explain that Qwen3-VL uses separate Instruct and Thinking
editions, not a hybrid that reliably toggles both modes. This source release fixes
that design rather than increasing the same failing request's budget.

| Stage | Model | Input |
| --- | --- | --- |
| Reference map | `huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M` | All image previews plus your full brief; compact final JSON |
| MiniMax director | `huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M` | Validated map plus your full brief; text only; `think=true` |

Both are abliterated. The service downloads both automatically, about 21 GB each
as listed by their publisher. No additional environment variable is necessary.
They are loaded SEQUENTIALLY: analysis unloads before direction; the director
unloads before H3. The two download sizes are not a measured peak VRAM figure.
Your 300 GB temporary disk setting remains the starting allocation; your own
LoRA collection and outputs consume additional space.

The prompt writer follows the included MiniMax Full-Reference instructions.
H3 still receives all ORIGINAL image tensors, not Ollama's resized previews.
Pass 1 uses temperature 0.10; the director uses 0.25. These are implementation
settings, not a demonstrated optimum for your photographs.

Malformed/truncated JSON gets at most one bounded regeneration per stage. A
failed director attempt switches back to the Instruct model for that regeneration,
never `think=false` on the Thinking model. The switch is logged. Broken JSON is
not fed back as something to continue, and missing reference entries are not
silently dropped. Final content is validated before a video job is returned.

The reference stage has a 240-second deadline across attempts; the director has
480 seconds across attempts; a repair has at most 180 seconds within the stage
budget. A blocked network read/cleanup can add its own bounded wait. Long failed
reasoning is not retried indefinitely. These limits and GPU offload behavior
have not been benchmarked on your host; slow jobs may stop rather than finish.

Ollama is unloaded and `/api/ps` checked before H3 receives a runnable job.
Abliteration, multiple passes, and the word "Thinking" do not guarantee correct
reference interpretation or better identity retention. Check the first draft.

## Add more LoRAs later

Edit ONLY `config/lora_links.txt`: one real Civitai or Hugging Face URL per line.
For a particular precision/version use that file's Download/file-page link.
Provider hashes and filenames are resolved automatically. Do not paste tokens.
Both templates make the same list available; workflows apply ONLY selected files.
An FP32 LoRA does not require an FP32 base model. It must match full H3 Ref2VA.

For portrait use, commit the list and run/wait for **Build H3 Portrait template**,
then deploy a fresh Pod. You do not also need the general build for portrait use.

## What has actually been verified

See `VALIDATION.md` for this consolidation's executed checks and limitations.
This ZIP is source/configuration, not a built Docker image, a GPU-tested release,
a copy of model weights, or an export of your Git history. A registry connection
and the existing pinned base images are still needed to build.

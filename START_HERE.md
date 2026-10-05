# Clean replacement for James's RunPod repository

Package snapshot: `h3-clean-2026-10-05-r1`.
Portrait code: **1.3.0, split-model-reference-v3**.

This is ONE complete repository, not a patch. Do not combine it with old ZIPs or
hotfixes. It includes the general template and the H3 Portrait template, but you
only need to build the portrait template for your current use.

## Do NOT delete the GitHub repository or its container package

Clear the repository's working FILES, not the repository itself. Keep its history,
settings, secrets, and GHCR images. Both Dockerfiles still depend on already
published, pinned base images in your existing container package.

Use GitHub Desktop for a single clean replacement commit rather than separate
browser upload batches. Do this on your computer, NOT the RunPod terminal.

## 1. Download and clone

1. Download this clean ZIP and extract it to a new folder in Downloads. The folder
   you will copy FROM must contain `Dockerfile`, `h3_portrait`, `config`, `tools`,
   `SOURCE_SNAPSHOT.json`, and `.github` directly inside it.
2. On GitHub, open `jsanso1497/runpod-comfy-stateless`. Use Code > Download ZIP and
   keep that old ZIP as your backup. Save any LoRA links you want to reuse.
3. Install/open GitHub Desktop, sign in, and choose File > Clone repository.
   Select `jsanso1497/runpod-comfy-stateless` and clone it to your computer.
   If already cloned, select it, use Fetch origin, and Pull origin if offered.
4. Select Current Branch > main. Then Repository > Show in Explorer opens the
   actual local checkout. Its folder is normally named `runpod-comfy-stateless`.

The extracted clean ZIP and the cloned repository must be TWO DIFFERENT folders.

## 2. Empty the checkout's old files

1. In the CLONED repository's Explorer window, enable View > Show > Hidden items.
2. Select everything with Ctrl+A, then hold Ctrl and click **.git** to DESELECT it.
3. Confirm **.git is NOT selected**, then press Delete.

The checkout must now contain **only `.git`**. Keep `.git`; it contains the clone's
history and connection to GitHub. The OLD `.github` folder DOES get deleted,
because the clean package will replace it. `.git` and `.github` are not the same.

Do not commit or push this empty state. Do not delete anything through GitHub's
Settings > Danger Zone. Do not delete the GHCR package, base image digests, tags,
or RunPod registry credentials.

## 3. Copy the entire clean package into the checkout

1. In the EXTRACTED CLEAN PACKAGE folder, keep Hidden items visible. Select all
   its contents and copy them.
2. Paste INTO the emptied CLONED repository folder, beside its preserved `.git`.
   Copy the contents, not the enclosing package folder or the ZIP file.
3. Confirm these locations exist directly inside the checkout:

   - `.github/workflows/build-h3-portrait.yml`
   - `.github/workflows/build-image.yml`
   - `h3_portrait/Dockerfile`
   - `h3_portrait/VERSION` containing `1.3.0`
   - `h3_portrait/node/analysis_prompt.txt`
   - `config/lora_links.txt`
   - `SOURCE_SNAPSHOT.json`

4. Open `config/lora_links.txt`, add your own LoRA URLs one per line, and save.
   No IDs, profiles, hashes, destination filenames, or JSON are required. Never
   paste API keys into the links. Use the desired file's Download/file-page link
   when a model offers multiple versions/precisions. No real LoRA URLs are included.
5. Keep every other package file unchanged for this first clean build. The source
   snapshot deliberately checks all files except your editable LoRA link contents.

## 4. Commit everything ONCE and push

In GitHub Desktop, keep ALL changes selected in the Changes list, including
removed files, `.github`, `.dockerignore`, and new files. Do not select only the
visible Python files.

Use this commit summary:

    Replace repository files with verified H3 clean snapshot

Click Commit to main, then Push origin. This sends the complete replacement as
one commit. You do not need to upload the ZIP through GitHub's browser.

If GitHub Desktop reports branch protection or a push conflict, do not force-push.
Resolve that prompt before building. Do not delete history to work around it.

## 5. Wait for the PORTRAIT build

On GitHub, open Actions > **Build H3 Portrait template** for the new commit.
The push should trigger it. If no run starts, use Run workflow > main.
Do not use Re-run on an older job.

The first validation step must show:

    CLEAN SOURCE SNAPSHOT VERIFIED: h3-clean-2026-10-05-r1
    H3 PORTRAIT SOURCE VERIFIED: 1.3.0 | split-model-reference-v3
    REPOSITORY PREFLIGHT PASS: portrait

The snapshot verification is read-only. It fails before Docker pulls the base
image if files are missing, truncated, stale, unexpectedly nested, or left over.
LoRA list contents are deliberately excluded from hashes, so adding links works.
Windows CRLF/LF differences are normalized for the snapshot comparison.

Inside Docker, expect the portrait's 150 tests and shared-LoRA 69 tests to pass,
followed by **H3 PORTRAIT SCHEMA PASS** from a real CPU ComfyUI startup.
The final summary must say:

    H3 Portrait 1.3 CLEAN image published

It will include the exact source commit and image digest.

Only this build matters for your portrait Pod. The general **Build RunPod ComfyUI
image** may start on the reset commit too. You can cancel/ignore that run for
portrait-only use. Neither build requires the other to finish first.

## 6. Change your portrait template's image ONCE

For this clean reset, change Container Image to:

    ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean

This is an additional tag from the portrait build. Older portrait builds never
published it. The normal `h3-portrait` tag is still published for compatibility,
but use the new clean tag for this reset. Future LoRA-list builds also republish
`h3-portrait-clean`; you do not paste a new SHA for each addition.

Keep the rest of the template at these existing values:

| Setting | Value |
| --- | --- |
| Container disk | 300 GB |
| Volume disk | 0 GB |
| Network volume | None |
| HTTP ports | 8188 |
| TCP ports | Blank |
| Container start command | Blank |
| Registry credential | Your existing GitHub GHCR credential |
| GPU | Keep the GPU type you are already using for H3 |

Custom environment variables:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
```

Preserve existing secret references if you used other secret names. Do not add
old CONFIG_REPO, CONFIG_REF, COMFY_ARGS, MODEL_PROFILES, OLLAMA_MODEL_DIGEST, or
RESTORE_* overrides to this portrait template. Do not remove RunPod-generated
variables. Do not change your OTHER/general template.

There is no file-browser service on 8888. Do not expose 11434; Ollama stays local.
Both 32B helper models are configured in the package and download automatically.
The analysis model is Instruct; the director is Thinking. They run sequentially.

## 7. Launch a fresh Pod, then test

Save outputs/references you need before terminating your current Pod. After the
clean portrait build succeeds, deploy a FRESH Pod using `h3-portrait-clean`.
No model downloads are needed to establish source identity: startup verifies the
image manifest and installed node copies first.

Confirm:

    H3 PORTRAIT 1.3 | Instruct analysis + Thinking director
    H3 PORTRAIT RELEASE VERIFIED: 1.3.0 | source=<your reset commit>

Compare that source commit with the successful build. A 1.1 or 1.2 banner is not
this source. Then wait for ASSETS READY, SCHEMA PASS, and OLLAMA READY.

Open HTTP Service 8188 > H3_Portrait_Auto. Use:

```text
aspect: 9:16 (or 2:3)
quality: Standard
seconds: 5
seed: 42
prompt_variation: 0
mode: Draft only
```

Upload 1-9 references, enter an ordinary-language instruction, and choose your
compatible full H3 Ref2VA LoRA in lora_1 with the author's strength. Leave lora_2
at (none) unless intentionally combining compatible adapters. Keep use_loras=true
when applying selections; turn it off for an explicit no-LoRA test.

Review What Ollama wrote, then change only mode to Generate video and Run again.
No OmniNode grammar, RefMod, manual prompt copying, or extra model profile setup.
The smaller Standard reference policy and the original H3 quality presets remain.

## Later additions and deliberate code changes

Add links only to config/lora_links.txt, commit/push, wait for the portrait build,
and launch a fresh Pod using the SAME `h3-portrait-clean` tag. The running Pod does
not sync GitHub changes. Other templates can build separately when needed.

The clean snapshot treats other source changes as intentional development. Do
not edit SOURCE_SNAPSHOT.json just to silence a failure: restore the actual file
from this ZIP, or regenerate a new reviewed snapshot for a deliberate code update.
This is a completeness/mixed-version guard, not a digital signature or guarantee
of model output quality.

## Verification limits

Read VALIDATION.md for executed tests. This ZIP is source, not a Docker image or
model weights. The current source was tested locally, but neither complete Docker
build nor live GPU generation was run here. GitHub must still pass the full build;
RunPod must still pass the first real prompt and video test. The private repository
could not be inspected, so no independent unseen custom edits are included.

GitHub reference documentation:
- https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop
- https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop
- https://docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages

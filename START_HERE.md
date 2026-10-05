# Complete upload: H3 Portrait + file manager

Snapshot: `h3-clean-filemanager-2026-10-05-r2`.
Portrait pipeline: 1.3.0, separate Instruct reference analysis and Thinking prompt writing.
File manager: password-protected JupyterLab, HTTP port **8888**, browsing **/workspace**.

This ZIP is the ENTIRE source repository. Your repository files have already been
cleared: upload this package only. No earlier patches, local Terminal, or GitHub
Desktop are needed. The existing GitHub repository and GHCR container package
must remain: both Dockerfiles depend on your previously published base images.

## 1. Extract, add your LoRA links, upload in the browser

1. Extract the ZIP in Finder. Open the extracted folder. You must see Dockerfile,
   config/, h3_portrait/, file_manager/, tools/, SOURCE_SNAPSHOT.json and .github/
   directly inside it, not another enclosing source folder.
2. In Finder, press **Command + Shift + .** to show hidden files. Include .github,
   .dockerignore, .gitignore and .gitattributes. These are supplied in the ZIP.
3. In your existing github.dev browser editor for
   **jsanso1497/runpod-comfy-stateless**, select **main**. Drag ALL extracted
   contents into the Explorer at the repository ROOT. Do not drop them inside a
   folder, and do not upload the ZIP file itself. No additional wipe is needed.
4. Open **config/lora_links.txt** in that browser editor. Paste your Civitai/Hugging
   Face LoRA links, one per line, and save. It currently contains comments only.
   Use the desired file's Download or file-page URL for a specific version/FP32
   variant. Do not put tokens in URLs. Do not change hashes, profiles or filenames.
5. In Source Control, stage ALL changes, including any remaining deletions. Commit
   and push with: **Install full H3 package with file manager**. Confirm the commit
   appears on the GitHub website's main branch.

The root should contain:

```text
.github/workflows/build-h3-portrait.yml
.github/workflows/build-image.yml
.dockerignore
.gitattributes
.gitignore
Dockerfile
SOURCE_SNAPSHOT.json
config/lora_links.txt
config/workflows/
file_manager/
h3_portrait/Dockerfile
h3_portrait/node/analysis_prompt.txt
h3_portrait/workflows/H3_Portrait_Auto.json
scripts/
shared_loras/
tests/
tools/
```

There are 17 ComfyUI workflow JSONs: 16 general workflows and the H3 Portrait
workflow. Both GitHub build workflows are included. The two kinds of workflow
are different and both are required in their supplied locations.

The browser editor commits files directly. It does not run Python or build Docker.
The GitHub Actions build performs those steps after the complete commit.

## 2. Wait for the portrait build

On GitHub, open **Actions > Build H3 Portrait template** for your new commit. If it
does not start automatically, select **Run workflow > main > Run workflow**.
Do not rerun a historical job; it would use the old commit.

Early source checks must include:

```text
CLEAN SOURCE SNAPSHOT VERIFIED: h3-clean-filemanager-2026-10-05-r2
H3 PORTRAIT SOURCE VERIFIED: 1.3.0 | split-model-reference-v3
```

The build installs the file manager in its OWN Python environment, leaving the
ComfyUI/Torch dependencies alone. It starts a temporary real Jupyter server and
checks password login, unauthenticated denial, CSRF, upload, download, edit and
delete before publishing. Its test line is:

```text
FILE MANAGER HTTP SMOKE PASS
```

Final successful summary:

```text
H3 Portrait 1.3 CLEAN + File Manager image published
```

For your portrait Pod, only THIS build needs to succeed. The general ComfyUI build
is independent; it can be ignored/cancelled for your current portrait-only use.
Do not deploy until the portrait build for the complete commit is green.

## 3. Create one RunPod secret

RunPod > Secrets > Create Secret:

- Secret name: **filebrowser_password**
- Secret value: your own strong password, **at least 12 characters**, one line.

Paste only the actual password as the secret value, not quotes, brackets or a
RUNPOD_SECRET expression. Store it in your password manager; RunPod does not show
its value again. Keep the existing huggingface_token and civitai_token secrets.

## 4. Set your H3 Portrait - Automatic template

| Field | Exact value |
| --- | --- |
| Container image | ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean |
| Container disk | 300 GB |
| Volume disk | 0 GB |
| Network volume | None |
| Expose HTTP ports | **8188,8888** |
| Expose TCP ports | Leave blank |
| Container start command | Leave blank |
| Registry credential | Your existing GitHub GHCR credential |
| GPU | Keep the GPU type you are already using for H3 |

Use this complete custom environment-variable set:

```text
OLLAMA_MODEL=huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M
ASSET_DOWNLOAD_WORKERS=2
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
FILEBROWSER_PORT=8888
FILEBROWSER_PASSWORD={{ RUNPOD_SECRET_filebrowser_password }}
```

Each line is one key/value pair. Keep your exact HF/Civitai secret references if
those secrets have different names. The file manager refuses to start without a
valid password; there is no default password or open-access fallback.

Remove obsolete CUSTOM overrides: CONFIG_REPO, CONFIG_REF, COMFY_ARGS,
MODEL_PROFILES, OLLAMA_MODEL_DIGEST, JUPYTER_TOKEN, JUPYTER_PASSWORD and RESTORE_*.
Do not remove RunPod-generated environment variables. Do not expose Ollama 11434.
**Ports go in Expose HTTP ports as well as the FILEBROWSER_PORT environment value.**
Merely setting an environment variable does not create the RunPod HTTP link.

## 5. Deploy and open the file manager

Save needed results from any old Pod. After the build succeeds, deploy a FRESH Pod.
The existing running Pod does not receive source changes from GitHub.

The file manager starts BEFORE large model downloads. Look for:

```text
FILE MANAGER READY: port 8888 | /workspace | password required | JupyterLab
H3 PORTRAIT 1.3 | Instruct analysis + Thinking director
H3 PORTRAIT RELEASE VERIFIED: 1.3.0 | source=<your new commit>
```

Confirm the source commit matches the successful build. Then:

- **Connect > HTTP Service 8888:** file manager. At JupyterLab's Password or token
  prompt, enter the actual password saved in filebrowser_password. No username or
  separate token is required. Do NOT enter the bracketed secret reference.
- **Connect > HTTP Service 8188:** ComfyUI, once assets/schema/Ollama are ready.

In JupyterLab the left file pane opens /workspace. Use the up-arrow Upload Files
button or drag files into that pane. Double-click a text file to edit and save;
right-click a file and select Download. The + launcher also offers a Terminal
running on the POD, not on your work Mac.

Useful locations relative to the file manager's root:

```text
ComfyUI/input/                    Uploaded image references
ComfyUI/output/                   Generated videos and prompt records
ComfyUI/models/loras/Shared/      Downloaded shared LoRAs
ComfyUI/user/default/workflows/  Saved workflow JSONs
```

A workflow source modification in the Pod is not a GitHub commit. Download what
you want to keep before terminating. /workspace is TEMPORARY in this setup because
no persistent volume is attached. The file manager provides access, not storage.
Keep the password private: Jupyter's editor/terminal grants control of this
container. It is not a sandbox. Port 8888 authentication does not add a separate
login to ComfyUI on 8188; keep your Pod access appropriately restricted.

## 6. First H3 run

Wait for H3 PORTRAIT ASSETS READY, H3 PORTRAIT SCHEMA PASS and H3 PORTRAIT OLLAMA
READY. Open **H3_Portrait_Auto** from ComfyUI's workflow sidebar.

1. Upload 1-9 references and check the displayed order.
2. In **2. Tell Ollama what you want**, describe your references and request in
   ordinary language. No reference grammar, OmniNode or RefMod is required.
3. Set aspect 9:16 or 2:3, quality Standard, seconds 5, seed 42 and
   prompt_variation 0. Select your full-H3-compatible LoRA in lora_1 and use its
   author's strength. Leave lora_2=(none) unless deliberately combining adapters.
4. Start with mode **Draft only**. Review What Ollama wrote. Then change only mode
   to **Generate video** and Run again. Direct Generate video is still available.

Pass 1 uses 32B Instruct for the compact image map. Pass 2 uses 32B Thinking for
MiniMax prompt writing. They unload sequentially before H3. Original images go
into native H3; downloaded LoRAs are only applied when selected. Standard stays at
20 steps, Preview at 12 and High fidelity at 25. No speed adapter is auto-enabled.

## Later LoRA additions

Edit config/lora_links.txt on GitHub, commit, wait for the portrait build, then
deploy a fresh Pod with the SAME h3-portrait-clean tag. No other fields need edits.
The list is shared by both images; only rebuild the one you intend to use.

## Verification boundary

See VALIDATION.md. Source/regression checks passed and a real LOCAL Jupyter file
manager passed HTTP tests. The whole Docker images and GPU inference are NOT
executed here. The new exact Jupyter dependency pins are checked in the GitHub
Docker build; local HTTP tests used the installed Jupyter versions listed in the
validation report. LoRA access/compatibility depends on the links you supply.

# Comfy Workbench HQ

**Same repository, new task-based setup.** This distribution replaces the contents of your existing `runpod-comfy-stateless` repository. It does not create another repository. See **[Mac installation without Terminal](docs/mac-install.md)**.

A task-first, local-model ComfyUI workbench for RunPod. Choose a model, choose the task, prepare the required assets, and open the matching workflow.

**Release status: source candidate, not a deployed or GPU-certified release.** This package contains working configuration and runtime code, migrated graphs, offline tests and build workflows. Docker images have not been built/published here. GPU inference, custom-node imports against actual installed ComfyUI, and generated-image quality still need validation on the built images.

## Start here

Open **[site/index.html](site/index.html)** in your browser. The page is self-contained and does not need a web server, API key, external script or internet connection to generate settings. Browser policies can restrict local HTML. The optional **Publish configurator to Pages** GitHub Action hosts the public catalog without running a local server. Do not enter secrets into the page.

Select **Qwen**, **FLUX**, **H3**, or **Restoration**. Pick tasks and optional assets, then copy the RunPod settings. The configurator never accepts secret values. Selecting different model families produces separate compatible workspaces, not an untested all-in-one image.

| Workspace | Standard model stack | Main use |
|---|---|---|
| Qwen Image 2.1 | BF16 diffusion model, encoder and VAE | Reference editing, identity, clothing, local repair, multi-person editing |
| FLUX.2 klein 9B | Source BF16 distilled checkpoint and matching components | Native generation and reference-guided masked photo editing |
| MiniMax H3 | Full BF16 diffusion/encoder; native FP16 video and FP32 audio VAEs | Video/audio, reference packages, media guidance and still-frame generation |
| SeedVR2 7B | FP16 model and matching VAE | Independent image/video restoration |

Every model has a `00_Standard_HQ` baseline. These are source-native baselines or integration examples, not falsely labeled unmodified official downloads. Specialized tasks stay separate.

**58 catalog entries, 70 editable workflow files, and a disposition for all 82 original source workflows.** The catalog contains 51 retained task choices, four Standard HQ entries, two standalone SAM tools and a Qwen prompt-preview tool. See [the task catalog](docs/tasks.md) and [migration details](docs/migration.md).

## Qwen prompt-enhancement update (0.1.2)

All Qwen generation templates now include an optional **official Qwen I2I BF16 prompt enhancer**, OFF by default, with a shared switch across every edit pass. Prompts stay in their existing fields. Masks, references, negative prompts and generation settings stay unchanged. **Toolbox U10** previews a rewrite without image generation. [Usage, storage and validation details](docs/qwen-prompt-enhancer.md).

For this incremental update, **do not empty the repository**. Upload the patch folder's contents to the current repository root, validate, then rebuild only `qwen`. The patch contains no hidden files and preserves your GitHub Actions and secrets. Use the newly built image digest. It adds about 19 GB to the Qwen model cache; the configurator includes that estimate. Existing saved personal workflows are not overwritten.

## Replace the existing repository, without Terminal

Use the **same GitHub repository**. Keep its URL and history. Do not delete the repository in GitHub Settings and do not upload the delivery ZIP as a single source file.

**GitHub Desktop:** Clone your existing repository, show it in Finder, and replace the working files with the contents of this folder. Keep the local `.git` folder. Show hidden files with **Command + Shift + Period** before selecting the replacement files. Review changes, commit, and click **Push origin**. Do not use Create repository or Publish repository.

**Browser only:** The companion browser-upload kit contains three upload batches plus a local HTML guide with visible, copyable versions of the six hidden configuration files. Upload each batch's contents to the root of the same repository, then use **Add file > Create new file** to create the hidden files at their exact paths. The kit needs no Terminal, CLI, GitHub Desktop, API token, or third-party upload service. Its wrappers and guide are not repository files.

The repository root must contain `Dockerfile`, `README.md`, `src`, `catalog`, `workflows`, `site`, and `.github`, not another enclosing folder. Details and safety checks are in [docs/mac-install.md](docs/mac-install.md).

In GitHub **Actions**, run **Validate repository**, then **Build HQ workspace**. Select one workspace or all four. The build automatically uses the actual owner/repository name and reports the real image digest. Do not launch an assumed image tag before its build succeeds.

The configurator defaults to `jsanso1497/runpod-comfy-stateless`. Its owner/repository field remains editable. New HQ image tags are separate from legacy tags; use the successful build's digest for reproducibility. For private GHCR images, configure RunPod container-registry credentials. Repository visibility and container-package visibility are separate settings.

## Configure RunPod

Your existing secret names are wired correctly:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
```

Create **one additional secret, `comfy_password`**, containing a unique password of at least 16 characters. The runtime requires it instead of exposing ComfyUI unauthenticated. Login username: `workbench`.

Use the configurator's image, disk sizes and environment rows. Expose **HTTP 8188 only**. Leave the container start command blank. ComfyUI and Ollama listen internally on loopback; the single public port is an authenticated gateway. Always use RunPod's HTTPS proxy, not unencrypted public direct access.

Private library secrets are optional: `comfy_loras` and `comfy_checkpoints`. They register compatible models without applying them automatically. See [private libraries](docs/private-libraries.md) and [organized environment variables](docs/environment.md).

## In ComfyUI

Open the **Workbench** sidebar. It separates Tasks, Toolbox and Files. Open a workflow, inspect missing assets or node types, and prepare only the additional catalog assets you need. The sidebar has a fallback button for older frontends without custom sidebar support.

The main graph remains editable. Source compact reference/director nodes are retained. App Mode is not force-enabled or assumed compatible with these pinned frontends; use it only after validating the particular workflow. This release's delivered simplified interface is the task browser and the existing compact task nodes.

Optional SAM, SeedVR2 and Ollama are not inserted into other graphs. Generic masking/compositing needs no generative model. U08 selects the named object from original **A**, expands/feathers that selection, and reveals aligned pixels from candidate **B**. It never selects the object from B.

Factory workflows are copied once into a versioned `Workbench Factory` folder. Existing files there are not overwritten, and your saved graphs remain separate. Immutable factory copies can always be opened again from the sidebar. API exports retained from the source are under [automation](automation/index.json); other graphs can be exported with ComfyUI's native API export.

## Testing and deployment gates

For normal installation, use **Actions > Validate repository** and **Actions > Build HQ workspace** in your browser. Local commands below are optional developer tools, not installation requirements.

```bash
python scripts/build_site.py
python scripts/validate.py
PYTHONPATH=src pytest -q
node tests/test_configurator.cjs
```

After a Docker image builds, start a Pod and run `scripts/validate_live.py` against its actual node schema. Then execute representative image/video workflows before treating it as production-ready. No workflow should be labeled GPU-tested based only on a JSON or schema check.

Reports are in [reports](reports). Read [security](SECURITY.md), [architecture](docs/architecture.md), [troubleshooting](docs/troubleshooting.md), and [third-party provenance](THIRD_PARTY.md) before publishing or extending the setup.

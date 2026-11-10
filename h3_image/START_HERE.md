# Native H3 Image: James's manual-prompt workspace

Release: 2026-11-10.2. This is the first-party MiniMax-H3-Image pipeline, not Fizgig, video REF2VA, Eros, or Pruna P-H3.

## 1. Apply on your Mac

Save `update_h3_image_workspace_v2.py` to Downloads. Open **GitHub Desktop > runpod-comfy-stateless > Repository > Open in Terminal** and run:

```bash
python3 "$HOME/Downloads/update_h3_image_workspace_v2.py"
```

Python 3.8+ is supported by the installer. It adds the independent `h3_image/` workspace and `.github/workflows/build-h3-image.yml`. It also recognizes the exact earlier 2026-11-10.1 package and can upgrade those files with backups inside your Git metadata directory. Unrecognized edits stop the installation before any file is changed. Re-running the installer is safe. Use `--check` to preview.

Review and commit in GitHub Desktop, then manually **Push origin**. The installer does not stage, commit, push, run an Action, install anything on your Mac, or change the old H3 Lab/Eros/Qwen workflows.

## 2. Build before renting a GPU

In GitHub: **Actions > Build Native H3 Image Workspace > Run workflow**, branch `main`.

Use a NEW run from the updated commit. This is not the Lab `ph3` profile. A separate image tag `h3-image` is published only after the build checks pass. The job first resolves the exact three model filenames, freezes repository revisions and SHA-256 checksums, and stops if any required file is unavailable. No model weights are baked into the Docker image or downloaded on the GitHub runner.

Copy the successful run's **RunPod image reference ending in `@sha256:...`**. Do not reuse a previous Lab digest. There is no fabricated digest in this package.

The build exports model metadata, software pins, workflow JSONs, the dependency inventory, native CPU checks, and browser/Jupyter test reports. A successful build is not a GPU-quality or capacity benchmark.

## 3. RunPod template

| Field | Value |
| --- | --- |
| Name | Native H3 Image - Manual INT8 |
| Image | Actual digest from the successful new Action |
| GPU | A40 48 GB initial test, or A100 80 GB for more headroom; neither GPU was tested here |
| Host driver | Compatible with the image's CUDA 13.0 PyTorch |
| System RAM | 128 GB or more as a planning allowance, not a measured minimum |
| Container disk | 30 GB |
| Persistent storage | 100 GB for a separate empty workspace; 200 GB recommended when sharing existing Lab files |
| Mount | `/workspace` |
| HTTP ports | `8188,8888` |
| TCP ports | None required |
| Start command and entrypoint override | Both blank |
| `COMFY_PASSWORD` | Your existing secret value, for example `{{ RUNPOD_SECRET_comfy_password }}` |
| `HF_TOKEN` | Optional read token, for example `{{ RUNPOD_SECRET_hf_token }}` |

Save your old image reference before switching. Preserve storage: do not terminate an ordinary Pod volume containing needed images or models. A replacement Pod does not automatically inherit another Pod's local volume. Only an actually shared/reused `/workspace` can reuse its encoder and files.

First boot downloads the locked model set sequentially, verifies each checksum, then starts ComfyUI. Files and Jupyter start independently before those downloads.

### Model set

- `minimax_h3_image_pruned_int8_convrot.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_image_vae_fp16.safetensors`

Comfy-Org lists this combination. The new INT8 diffusion file is approximately 21 GB. Exact individual sizes and total download size are printed by the build. The encoder is reused from `/workspace/h3-still/models/text_encoders` only when its size and checksum match. A reused encoder is linked, not copied or modified. Deleting its old source would break that link.

There is no BF16 diffusion checkpoint, prompt enhancer, prompt builder, old Turbo adapter, Akatz adapter, or experimental accelerator. The necessary Qwen3-VL text/vision encoder is not a prompt enhancer.

## 4. Open the services

**8188: ComfyUI.** **8888: independent file browser and JupyterLab.** Sign in using your existing Comfy password, no username.

Use your ACTIVE Pod's RunPod Connect HTTPS links, not a saved link from an old Pod. If needed, use the fresh sign-in path:

```text
/__h3/login-v3
```

For files, replace the URL path with:

```text
/__h3/files
```

Port 8888's home page also offers JupyterLab. In Jupyter choose **Launcher > Terminal**. RunPod's own Web Terminal toggle is not needed.

Persistent paths:

```text
/workspace/h3-image/input
/workspace/h3-image/output
/workspace/h3-image/output/H3_Image/reports
/workspace/h3-image/logs/comfy.log
/workspace/h3-image/exports
/workspace/h3-image/user/default/workflows
/workspace/h3-image/custom_nodes
```

Manager is installed using the pinned core's manager requirements and enabled automatically. Security is not lowered. User settings and the configured added-node directory persist. Arbitrary third-party Python dependencies installed later do not become part of the image automatically; reinstall those through Manager or deliberately bake them into a future image. Do not use Update All for the first test.

## 5. Your six workflows

Open **Workflows > `00_H3_Image_Manual_v2`**. Earlier v1 graphs and user-saved graphs are not overwritten.

| File | Inputs |
| --- | --- |
| `00_Generate_No_References` | Scene prompt only |
| `01_Edit_One_Image` | Picture 1 is the photograph to edit |
| `02_Generate_From_References` | 1-16 optional still photos; assign roles yourself |
| `03_Replace_Person_Manual` | 1 = target scene, 2 = face/head, 3 = body |
| `04_Two_People_Heads_Bodies_Manual` | 1 = A head, 2 = A body, 3 = B head, 4 = B body |
| `05_Replace_Body_Face_Scene_Last` | 1 = body, 2 = face/head, 3 = target scene |

Every prompt box is blank. Nothing is prefixed, rewritten, or appended. Dynamic prompt expansion is off. Role labels are reminders, not identity-routing algorithms. Use the exact `<Picture 1>` spelling in your own prompt. Empty/disabled slots are omitted and active slots are numbered in order; the report shows their mapping.

Seed defaults to fixed. Change it deliberately when exploring, then keep it fixed when comparing prompts/settings. This is not an identity lock.

All workflows use the uploaded model's six-step PDD sigma grid, Euler, one image per run, and no classifier-free guidance. The schedule helper rejects a model without that exact native image grid. Do not change to 50 steps or add a video Turbo LoRA.

### Scene size and reference sizes

The output control is `output_megapixels`, default **4 MP**. Sizes align to 32 pixels. Supported controls cover 4-20 MP using ComfyUI's 1024-squared convention.

`follow_image_1_aspect = true` activates source-aspect sizing. `aspect_source_slot = 0` means the first active picture. The scene-last workflow sets it to **3**, so it follows the scene's proportions without moving the body/face references. This affects size only, not the prompt or reference order. Turn source-aspect sizing off to use the aspect-ratio dropdown.

The reports show source dimensions and the actual native vision/VAE reference dimensions. Native H3 Image can enlarge small references. Do not assume shrinking an upstream reference reduces its final conditioning cost. Start with the few useful references the task requires; 16 slots are not a promise that 16 large images fit your GPU.

Edit/replacement workflows also save a **source-left/result-right comparison**. The preview contains the source without stretching. It is intentionally smaller for convenient review; the separate FINAL PNG is full output resolution. This is still whole-image generation, not pixel-protected inpainting. No SAM model is downloaded automatically.

## 6. First test

1. Run 00 at **4 MP**, fixed seed. Enter a simple scene description.
2. Run 01 with one photo at **4 MP**. Describe a clear local change.
3. Test 03 with your successful scene/face/body photographs and manual prompt. Do not add extra adapters.
4. Try 04 or 05 only after the simpler runs work. Increase to **8 MP** after confirming memory use and visual quality.

The model card's approximate 16 MP preference is not an A40 capacity guarantee. Do not treat a 1 MP output as a representative quality test of this model.

## 7. Memory, restart, status, and ZIP commands

Run on the POD in Jupyter Terminal, not your Mac.

Status, container memory limits, and queue counts:

```bash
python3 /opt/h3-image/control.py status
```

Release model memory while idle:

```bash
python3 /opt/h3-image/control.py free
```

Restart ComfyUI through its existing supervisor:

```bash
python3 /opt/h3-image/control.py restart
```

Running or queued jobs block these actions. To intentionally cancel them, add `--interrupt`. If the backend is unresponsive and queue state cannot be established, `restart --interrupt` explicitly acknowledges that a run could be interrupted. The file browser/Jupyter stay separate. The backend has a singleton lock to prevent the earlier duplicate-launch problem. This does not cure insufficient hardware memory or stop GPU billing.

The image defaults to no execution cache and disabled pinned host buffers to limit retained memory after your OOM history. This trades repeat-run speed for reduced retention; it is not an OOM guarantee. Do not run an old h3-still restart command in this workspace.

Create a ZIP of all outputs while idle:

```bash
python3 /opt/h3-image/control.py zip
```

It streams files without recompressing PNGs, uses ZIP64, skips symlinks, checks disk space, and never deletes originals. It prints a unique archive path in `h3-image/exports`. Download that ZIP through the file browser. Finish generation before exporting.

## Limits and authorization

The model's standard license excludes the United States, United Kingdom, European Union, and South Korea, with a separate authorization route. Confirm your authorization before self-hosted use; neither this package nor GPU rental grants model rights. See the official model LICENSE and MiniMax's license application page.

No GPU generation, identity fidelity, A40 capacity, or live RunPod proxy has been tested in this preparation environment. See `VALIDATION.md` for the exact local tests and the tests deferred to your GitHub build.

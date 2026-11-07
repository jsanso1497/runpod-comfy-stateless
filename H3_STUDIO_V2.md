# H3 Still Studio v2: photos, video references and person replacement

Release 2026-11-07.2. Prepared against `jsanso1497/runpod-comfy-stateless` commit `9832332b2eec18a93ec79e6da10066f9f4bda9fe`.

This is an incremental update to the existing H3 Still Studio, not another model workspace. It adds a bounded video-reference input, three ready-to-use workflows, a mask-protected person-replacement option, and a new browser-access layer. The original Studio nodes and workflows remain available. Other model workspaces are not changed.

## 1. Install locally, then build

Save `update_h3_studio_video_person.py` to Downloads. In GitHub Desktop, select `runpod-comfy-stateless`, then choose **Repository > Open in Terminal**. Run:

```bash
python3 "$HOME/Downloads/update_h3_studio_video_person.py"
```

The installer supports Python 3.8 or newer, checks all changes before writing, backs up replaced files under Git's private metadata directory, and rejects unrecognized edits. It makes no network requests and does not stage, commit, push, build or deploy. `--dry-run` previews changes. Running the same update twice is safe.

Review and commit in GitHub Desktop, then manually **Push origin** there. In GitHub, start a NEW **Actions > Build H3 Still Developer Workspace** run on the updated branch. Do not rerun the old commit's build. The new image is published only after its tests pass. Use the new run summary's exact image digest in RunPod.

The workflow JSONs in this package require the rebuilt image because they use the new media helper nodes. Importing them into the old image alone is insufficient. Do not apply older H3 repair installers after this update.

## 2. RunPod template

| Field | Value |
|---|---|
| Image | Exact new image@sha256 from the successful build summary |
| Existing rolling tag | `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still` |
| First-test GPU | A100 80 GB or comparable supported CUDA GPU |
| System RAM | 128 GB or more as a planning target |
| Container disk | 30 GB |
| Persistent storage | 150 GB recommended, mounted at `/workspace` |
| HTTP ports | `8188,8888` |
| TCP ports | None required |
| Start command and entrypoint overrides | Leave blank |

The pinned image still uses PyTorch CUDA 13.0 and requires a compatible host driver. GPU/RAM values are planning targets, not measured minimums for every workload.

Keep these environment variables:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Only `COMFY_PASSWORD` is required; use the existing unique 16-72-byte password without newlines. There is no new browser token, username, external account or prompt-enhancer setting.

Preserve the existing `/workspace` data when changing images. An ordinary Pod volume survives stopping, but not termination. Do not terminate a Pod containing needed files before backing them up or confirming they are on a retained network volume. A new network volume does not automatically contain the old Pod's files.

## 3. Browser access: what changes

Both ports now display a normal password sign-in page. Use the value of `comfy_password`. **No `workbench` username and no HTTP Basic Auth popup are used by v2.**

For the first visit after replacing the old image, use RunPod's HTTPS Connect URL and open the following fresh path on either service:

```text
/__h3/login
```

This path avoids relying on the previously cached root page. The application sets no-store headers on login, API and proxied responses, and handles ComfyUI's post-login cache-busting automatically. It does not disable authentication when a query string is present.

An already cached denial at the provider's edge or in the browser cannot be purged by a container update. The original RunPod proxy behavior has not been observed directly. If the old root still shows a denial, use the explicit sign-in path above in a new tab. If it still returns the old denial, confirm that the running Pod uses the NEW image digest and the active Pod's HTTPS URL. Clearing site data or using a fresh Pod hostname may be needed, but preserve your volume first.

### Port 8188: ComfyUI

Sign in with the password. The gateway starts before model preparation. If ComfyUI is not ready, its page offers Connection diagnostics rather than an authorization bypass. The file browser on port 8888 remains independent.

### Port 8888: Files, JupyterLab and terminal

After sign-in, choose **Open file browser** or **Open JupyterLab**.

**Open file browser** is a small independent interface, not the JupyterLab application and not the third-party File Browser binary. It lists `/workspace`, supports downloads, and uploads large files in 4 MB chunks. It remains usable even if JupyterLab does not initialize. Existing files are never overwritten by an upload; choose a different filename for a replacement.

**Open JupyterLab** uses `/jupyter/lab/workspaces/h3-studio-v2?reset` with a fresh interface workspace and settings. Its backend listens privately on loopback; the gateway handles browser login and supplies a private backend token without exposing it in URLs or page configuration. All API and WebSocket traffic follows the same authenticated proxy path. XSRF protection is not globally disabled.

For a shell, choose **JupyterLab > Launcher > Terminal**. RunPod's managed Web Terminal toggle is not required. If the Lab interface still fails, use the independent file browser to retrieve diagnostics instead of being locked out of your files.

Logs are available under:

```text
/workspace/h3-still/logs/access.log
/workspace/h3-still/logs/jupyter.log
/workspace/h3-still/logs/comfy.log
```

The new launcher writes detailed backend logs there. RunPod container logs show service startup/exit messages. Jupyter and the file service stay available if ComfyUI stops. This does not stop GPU billing.

## 4. Workflows

Open **Workflows > 00_H3_Studio_v2**. The original `00_H3_Studio_v1` folder and your saved Turbo workflow are not overwritten.

| Workflow | Use |
|---|---|
| `00_H3_Studio_Photos_Video_INT8` | General generation, editing and reference compositions, with optional photos and a video |
| `01_H3_Video_Reference_To_Still_Turbo` | Ready-to-use short video plus optional photos to generate one still with REF2VA Turbo |
| `02_H3_Replace_Person_Image1` | Replace the person in an existing photograph using other photographs and/or a video |

### General Studio behavior

Auto with no active media generates from text. Auto with one photograph edits that photograph. Auto with two or more photographs, or any enabled video, uses reference generation. Choose **References** explicitly when one photo should guide a NEW scene instead of being edited.

Empty photo/video slots remain `(none)`. A disabled slot is truly ignored. Active photographs are numbered consecutively in slot order as `<Picture 1>`, `<Picture 2>`, etc. The optional video is `<Video 1>`. The route report shows the actual ordering, reference sizes and selected model.

Direct prompting is retained. No prompt enhancer, captioning model or hidden rewriting is added. The general workflow starts at 2.5 MP for a conservative first test; higher values up to 8 MP remain available. These are proposed operating defaults, not tested quality optima.

## 5. Video reference to a still

Open `01_H3_Video_Reference_To_Still_Turbo`. Upload a clip into the **Optional video reference** node. You can also upload it through the independent file browser into `h3-still/input`, refresh ComfyUI's node definitions, and select the file.

Start with:

- `start_seconds`: the point in the source clip to begin.
- `duration_seconds`: **3.1** for the initial test.
- `video_cap_megapixels`: **0.5** initially.
- Sampling preset: **REF2VA Turbo - 8 steps**.
- Output: the shipped approximately 1 MP portrait canvas for the initial test.

The helper decodes only the selected interval, normalizes it to 24 fps, scales it to a bounded reference size, and trims the tail to H3's 17*n+5 frame grid. A sufficiently long 3.1-second selection normally yields 73 usable frames, about 3.04 seconds. It does not manufacture a full reference clip when the selected interval is too short. Audio is ignored and no audio VAE is needed.

The selected reference duration is passed to the native H3 conditioning node, avoiding the previous five-frame truncation. The sampler still receives the separate Fizgig single-frame latent. The output is **one newly generated image**, not a source-video screenshot or a generated video.

Optional photographs can supplement the video, for example a sharp face photograph plus a video showing body proportions and alternate angles. Refer to the clip as `<Video 1>` and the photographs by their active Picture tags. A video does not count as an extra Picture number.

Start with a clear, short clip and one or two useful photos. Longer clips and more references increase conditioning cost. The loader bounds decoded CPU memory, but it does not guarantee GPU memory capacity. Eight-step still generation with video conditioning remains a deployment test, even though the photo-only Turbo route worked for you.

## 6. Replace the person in image 1

Open `02_H3_Replace_Person_Image1`.

1. Upload the EXISTING scene into the native **TARGET IMAGE 1** Load Image node.
2. Upload the replacement person's identity/body photographs into the other photo slots. An optional video can supply additional views.
3. The default has `protect_background = true`. Open the target Load Image node's **Mask Editor**, paint the whole person to replace, including the clothing/outline to change, and save the mask. White selects replacement pixels; black preserves the source. Include any intended changed shadow area in the mask.
4. Keep **Replace person** mode. Edit the prompt to clarify which references define the face, body or clothing.
5. Run. A separate source-left/result-right comparison is saved.

The replacement preset begins with Quality at 50 steps. **REF2VA Turbo - 8 steps** is also available directly in its preset dropdown. Personal identity/style LoRAs are optional and independent of Turbo.

**What mask protection guarantees:** the final composite keeps zero-mask pixels from the decoded original image and returns the original canvas dimensions. It does not preserve original file encoding, metadata or compression bytes. Soft mask edges blend the candidate into the original.

**What it does not guarantee:** H3 is still generating a whole-image candidate. This mask is applied at final compositing, not supplied as native inpainting conditioning. The new person must align with the desired position/pose, and a poor candidate can create clipped limbs or seams inside the chosen region. Review the result and adjust the prompt or mask accordingly. No SAM model is downloaded.

A blank mask raises an explanatory error rather than silently returning an unchanged image. To intentionally allow whole-scene regeneration, set `protect_background = false`; other scene details can then change. Image 1 remains the target scene, not the replacement identity source.

## 7. Sampling, models and LoRAs

The existing FL2VA/REF2VA pruned INT8 checkpoints, NVFP4 encoder and INT8 video VAE are unchanged. No BF16 base model, alternate VAE, P-H3 stack, audio model, SAM or enhancer is added.

The REF2VA Turbo preset reuses:

```text
minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
```

This is the 1.96 GB adapter used by the working standalone Turbo workflow. Its filename describes the adapter's weights, not a BF16 base model. Startup reuses a matching cached file; if absent, it downloads and verifies this adapter. The base model cache stays about 61 GB, plus the adapter. Model/download manifests are kept separately so existing integrity checks remain in place.

Quality uses 50 steps with er_sde/simple and no Turbo. REF2VA Turbo uses 8 steps, Euler/simple, strength 1, and video/audio shifts 12/3. The same shifted model feeds the guider and scheduler. Developer FL2VA fast retains 20 steps, er_sde/simple and Larry Turbo strength 0.38, and is rejected for reference/person modes.

Two separate personal H3 LoRA slots remain off by default. Upload trusted compatible safetensors to `/workspace/h3-still/models/loras`. Refresh the model list, choose an adapter, enable its slot, and use its documented strength. Both Turbo adapters are excluded from personal slots to avoid accidental double application. FL2VA/REF2VA adapter compatibility still needs checking; a loaded file is not a likeness guarantee.

## 8. Validation and limits

Local execution covered all offline regression tests, real FFmpeg interval decoding at 24 fps, 73-frame reference routing, mask-required errors and exact original-pixel compositing. It also exercised the new gateway over HTTPS with real Jupyter APIs and static assets, password rejection, query-string authentication, origin checks, chunked uploads/downloads, and a real terminal shell command over secure WebSockets.

The authoring environment's managed Chromium policy blocks navigation, so the full graphical JupyterLab test could not run locally. The GitHub build now runs that test in a clean disposable browser container: sign-in, file browser initialization, file upload and a terminal command through the same HTTPS gateway. The production image does not receive Chromium or Playwright. Publication is blocked until these checks pass.

The build also imports actual pinned ComfyUI, validates node schemas, and exercises native image/video conditioning with synthetic encoder/VAE fixtures. These tests do NOT load model weights. GPU sampling, deployed RunPod proxy behavior, identity quality, 8 MP capacity and large reference sets remain untested here. Run one short video, one two-photo reference job and one masked replacement before scaling up.

H3 model rights and authorization requirements remain unchanged. This source update distributes no model weights and does not grant model rights.

## Primary source references

- ComfyUI native H3 reference implementation: https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy_extras/nodes_minimax_h3.py
- Fizgig implementation: https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/10d5171f6e45b0757a03df977f53a660e060c33d
- Existing quantized models: https://huggingface.co/Comfy-Org/MiniMax-H3
- REF2VA Turbo adapter: https://huggingface.co/lightx2v/Minimax-h3-Turbo
- Jupyter server proxy/auth configuration: https://jupyter-server.readthedocs.io/en/latest/other/full-config.html
- RunPod storage persistence: https://docs.runpod.io/pods/storage/types

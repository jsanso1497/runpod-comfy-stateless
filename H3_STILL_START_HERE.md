# H3 Still Developer Workspace

Release: **2026-11-05.1**

An independent workspace in `jsanso1497/runpod-comfy-stateless`, based on the developer's actual Fizgig H3 Still examples. It has its own Docker image, source pins, downloads and persistent directory. It does not change the existing H3 video, Qwen, MOSS or restoration workspaces. It is launched with the dedicated GitHub Action, not the existing Workbench configurator.

## 1. Apply the downloaded update locally

Save `install_h3_still_workspace.py` in Downloads. Open a terminal inside your existing repository and run:

```bash
python3 "$HOME/Downloads/install_h3_still_workspace.py"
```

The installer adds files only. It verifies its embedded payload, rejects conflicting existing files, preserves unrelated changes, and makes no network requests. It does not stage, commit or push. Running it again with the same files is safe. A preview is available with `--dry-run`.

Review the added files in GitHub Desktop, commit them, then use **Push origin** in GitHub Desktop. Do not replace the repository or delete existing files.

## 2. Build your dedicated image

In the repository's **Actions** tab, run **Build H3 Still Developer Workspace**.

The action performs offline tests, checks out pinned ComfyUI and developer sources, verifies the upstream graph blobs, creates portable presets, and builds the image without publishing it. It then starts actual ComfyUI on CPU, checks its node/model dropdown schemas, and tests the password gateway and WebSocket upgrade. Requests without valid credentials must fail, including requests with a query string.

Only after those checks pass does it publish `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still` and a commit-specific tag. The run summary and `h3-still-build/image-reference.txt` contain the real immutable image digest. Use that digest in RunPod. Do not use an assumed digest or an old H3 image.

The **h3-still-build** artifact contains the generated portable JSON workflows, untouched original JSON workflows, source and model locks, upstream licenses, installed package versions and the CPU test report. The exact upstream graph files are fetched during the Docker build, rather than reconstructed or bundled as approximations in this source update.

The action uses the repository's standard `GITHUB_TOKEN` with package-write permission. If the resulting GHCR package is private, supply appropriate registry credentials in RunPod or intentionally make that package public. A public source repository does not automatically make its image public.

## 3. RunPod setup

| Setting | Value |
|---|---|
| Container image | Exact digest from the successful build summary |
| First-test GPU planning target | A100 80 GB, or comparable supported 80 GB+ GPU |
| System RAM planning target | 128 GB or more |
| Persistent volume | 100 GB, mounted at `/workspace` |
| Container disk | 30 GB |
| HTTP port | `8188` only |
| Container start command | Leave blank |
| Login username | `workbench` |

These are conservative planning targets, not measured minimums or guarantees for 8 MP inference. GPU testing is still required. The image uses official PyTorch 2.9.1 CUDA 13.0 wheels to follow the H3 model card's INT8/ConvRot guidance. Select a host with a compatible NVIDIA driver. The NVFP4 text encoder is not restricted to Blackwell GPUs according to the model card.

Required variable:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

The existing RunPod secret name is reused, but the environment variable for this independent image is **COMFY_PASSWORD**, not WB_PASSWORD. The secret must contain a unique password of 16-72 UTF-8 bytes without newlines.

Optional variable:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

The pinned repositories were public and ungated when checked. The optional token can help with authenticated Hugging Face access. No Civitai token, Ollama model or prompt-enhancer setting is required.

Do not add `WB_WORKSPACE=h3`, expose port 8189, or reuse the legacy H3 startup command. Always open port 8188 through RunPod's HTTPS proxy.

On first startup, approximately **40 GB** of weights download to the persistent volume. The logs show download and verification progress. The public port opens after preparation and schema validation. Later starts reuse verified local files. Stop the Pod through RunPod when finished; this workspace does not stop GPU billing.

## 4. Choose a workflow

Open **Workflows > 00_H3_Still_Developer_2026_11_05_v1**.

| Workflow | Preset |
|---|---|
| `00_H3_Still_8MP_NoTurbo.json` | Developer's 8 MP text-to-image example, 50 steps, `er_sde`, `simple`, Turbo strength 0 |
| `01_H3_Still_Turbo_2_5MP.json` | Developer's 2.5 MP text-to-image example, 20 steps, `er_sde`, `simple`, Turbo strength 0.38 |
| `02_H3_Still_Edit_README_2_5MP.json` | Image editing, 20 steps, Turbo 0.38, README's 2.5 MP reference resize |
| `03_H3_Still_Edit_Original_2MP.json` | Same editing graph with the original downloaded JSON's 2 MP resize |

For a first deployment test, run the smaller Turbo example before the 8 MP preset. This checks the deployment; it is not a claim that Turbo produces the highest quality. The developer describes the no-Turbo example as the highest-quality option and recommends 3 MP or greater for the strongest still results. The smaller official Turbo and editing examples remain available as supplied.

For editing, **upload your image in LoadImage first**. The filename saved in the developer's original graph refers to their own sample, which is not included. Write the edit in the native reference node's prompt, retaining the `<Picture 1>` marker. No prompt-rewriting model is inserted. These are native editable ComfyUI graphs, not a separately tested App Mode interface.

The saved PNG comes from **Fizgig H3 Still Decode**. The standard VAE Decode preview remains in the graph only as the developer's comparison. It is not the saved final. Keep the native video node's LATENT output disconnected: the sampler uses **Fizgig H3 Still Latent** instead. The required Qwen3-VL encoder is part of H3 conditioning, not an optional prompt enhancer.

## 5. Exact differences from the source downloads

The three original upstream JSON files are preserved byte-for-byte in the built image and build artifact. Portable copies make only these integration changes:

1. Replace the Windows LoRA subfolder selection with the actual Linux model filename. Correct stale VAE and LoRA download metadata to match the selected pinned files, including the developer's Larry v4 step600 EMA LoRA rather than the unrelated LightX2V metadata link.
2. Set the 8 MP selector's size multiple from 16 to 32, following the developer's stated H3 dimension rule. Its 8 MP target, sampler, 50 steps and zero Turbo strength stay unchanged. The README describes the 8 MP example as 3872 x 2176.
3. Provide both the edit JSON's saved 2 MP setting and the README's documented 2.5 MP setting, rather than silently choosing one and discarding the other.
4. Assign each portable workflow a unique stable ID and add provenance metadata. Retain the developer's prompts, seeds, links, other settings and decoder-comparison branch.

The editing graph uses the **unmodified official `AILab_ImageResize` class** from pinned ComfyUI-RMBG. A narrow registration shim exposes that class only; the complete pinned RMBG source and GPL license remain in the image. Unrelated background-removal nodes and their extra models are not installed or registered. Fizgig's two custom nodes are installed unmodified.

## 6. Model files

| Component | File |
|---|---|
| Diffusion model | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Required encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` |

This intentionally follows the developer's quantized stack. It is not your separate full-BF16 H3 video stack. No audio VAE, Ref2VA checkpoint, alternative still VAE, restoration model, SAM model or separate local language model is downloaded.

Model revisions and SHA-256 values are in `h3_still/models.lock.json`. Downloads are checked before use. A corrupted model is not silently replaced. The runtime uses a single persistent download copy and model-path links rather than keeping duplicate full caches. Verification metadata avoids rehashing all weights on each ordinary restart. Set `H3_STILL_VERIFY_ALL=1` to force a full checksum pass.

## 7. Persistence and troubleshooting

Inputs, output PNGs, model files and saved workflows are under `/workspace/h3-still`. Keep that volume when replacing the Pod. Factory workflow copies are seeded without overwriting existing files or personal edits.

If the build fails, inspect the first failing Action step. It has not published a new image. If startup has no public port yet, check container logs for the current download, disk-space, password or node-schema error. A CUDA initialization error usually requires checking the host driver and GPU allocation, not changing the model filenames. Never bypass authentication to troubleshoot.

After the Pod says Ready, an additional schema check can be run from its web terminal:

```bash
python /opt/h3-still/validate_live.py
```

That checks schemas only. A real image generation is the GPU validation step. After a successful first image, test both the no-Turbo generation and editing workflows using representative images before relying on them for production.

## Validation status

The delivery includes **36 passing offline unit tests** covering synthetic graph fixtures, preset preservation, paths, cache verification, password handling and the schema validator. They do not execute the original workflows or load model weights. The container build, actual ComfyUI startup, proxy checks, GPU inference, peak VRAM, speed and visual quality have **not** been executed in the authoring environment.

Your dedicated build action performs the real CPU startup, node/schema and authentication checks before image publication. GPU inference and image quality remain a separate test on your Pod. Python dependencies without exact upstream version pins are resolved at build time and recorded in `build-pip-freeze.txt`; this is not a fully hash-locked Python environment.

## Sources and licenses

See `h3_still/SOURCES.md` for exact primary-source links and pins. The Fizgig node pack is MIT-licensed; ComfyUI and RMBG have their respective GPL licenses. Model and LoRA licenses are separate from the node-code licenses. Do not infer that all weights are MIT-licensed or automatically cleared for every use.

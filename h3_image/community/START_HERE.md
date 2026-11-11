# Native H3 Image: community sampler tests

Version: 2026-11-10-community.1

These are **new experimental workflows for the Native H3 Image workspace**. They are not for the old H3/Fizgig Lab, Eros, or Pruna P-H3 workflow 11. Your existing 00-05 files, core/model pins, CPU build repair, Manager, authentication, model downloads and service launchers are unchanged.

## What the supplied conversation supports

The Reddit screenshot recommends 16 MP and SamplerLCM with s_noise between 0.95 and 1.20. It mentions some users testing a Pruna VAE. It does not specify the LCM step count, scheduler, exact VAE file/precision, full graph, prompt, seed, or which reference-processing path was used. Replies report improvements, but one explicitly says the VAE changed the image without establishing that it was better.

The Discord screenshot separately recommends 16 MP, er_sde and simple at 4-6 steps. This is an alternative sampler recipe, not an instruction to combine LCM with ER-SDE. Another screenshot says 50 steps is too high and displays three different results described as 6/12/50 steps. The screenshot does not supply a fully controlled comparison or unambiguous per-image settings.

The final uploaded image arrived as `minimax_h3_image_t2i_00066_.jpeg`, 1536 x 657 pixels. Its JPEG segments, EXIF including the nested IFD, metadata keys, and bytes were checked. No workflow or prompt JSON was present, and no appended file data was found. It cannot establish the original generation resolution. This package recreates the visible recommendations; it is NOT the original Discord workflow recovered from that image.

## Technical interpretation

The supplied official model card also recommends approximately 16 MP as the quality target. The model is distilled to six steps. Euler at six steps remains the control. ER-SDE and LCM are community experiments; four steps is an experimental departure from the six-step recipe. There is no 50-step preset here.

For LCM, six steps with the simple scheduler is our selected controlled starting test, not a detail recovered from the screenshot. The native SamplerLCM node has `s_noise`, `s_noise_end`, and `noise_clip_std`. Keep the first two equal for a constant noise setting, and leave clipping at zero. No LCM LoRA or model-sampling patch is added.

The Pruna trial uses its published INT8 VAE because the screenshot does not identify a specific file. It uses that VAE only to decode an already sampled latent. The same latent also goes to the native image VAE. This deliberately avoids changing reference encoding at the same time. Shared tensor shapes are not evidence that the decoders are interchangeable in quality.

## Fastest way to test now

Use the supplied JSONs in your **running Native H3 Image workspace**. Drag one onto the ComfyUI canvas. No restart, new nodes, new base model, or rebuild is required for workflows 10-17, provided your workspace has the checked native SamplerLCM inputs.

All prompt fields are blank. Write your prompt directly. Seeds default to **fixed**, and every workflow starts at **16 MP**. The target resolution is explicit, not an upscale pass. If memory capacity is unknown, lower `output_megapixels` to 4 for one smoke test first. Then test 16 MP only if it fits. Do not judge the 16 MP claim from a 4 MP result.

### Recommended first comparison

1. Run **10_Community_T2I_Euler_16MP_Control**.
2. Run **11_Community_T2I_ER_SDE_6_16MP** using the exact same prompt, seed, aspect ratio and resolution.
3. Run **13_Community_T2I_LCM_6_16MP** with both noise fields at 1.00.
4. On the same LCM workflow, test both noise fields at 0.95, then 1.10, then 1.20, one job at a time. Keep every other value unchanged.
5. Only after the six-step comparisons, try **12_Community_T2I_ER_SDE_4_16MP**.

Do not alter resolution and sampler in the same comparison. At fixed dimensions, a matching seed controls the starting random draw but does not force the same image through different samplers. Some samplers also inject stepwise noise.

### Workflow list

| File prefix | Task | Sampler | Steps | Optional model |
| --- | --- | --- | --- | --- |
| 10 | Text-to-image control | Euler | 6 | None |
| 11 | Text-to-image | ER-SDE | 6 | None |
| 12 | Text-to-image | ER-SDE | 4 | None |
| 13 | Text-to-image | LCM, noise 1.00 to 1.00 | 6 | None |
| 14 | Replace one person | ER-SDE | 6 | None |
| 15 | Replace one person | LCM, noise 1.00 to 1.00 | 6 | None |
| 16 | New scene with two people | ER-SDE | 6 | None |
| 17 | New scene with two people | LCM, noise 1.00 to 1.00 | 6 | None |
| 18 | Same latent, two decoders | Euler | 6 | Pruna INT8 VAE only |

For **14/15**, Picture 1 is the scene, Picture 2 the head/face, and Picture 3 the body. Output follows Picture 1's aspect ratio.

For **16/17**, Picture 1 is A's head, Picture 2 A's body, Picture 3 B's head, and Picture 4 B's body. Select the aspect ratio for your new scene. Specify all reference roles and the desired scene in your own prompt; the labels are only reminders.

For a matched replacement control, open your existing **03_Replace_Person_Manual**, set the same 16 MP output, and use the same references, prompt and seed. There is no need to replace that original file.

There is no SAM, masked inpainting, prompt builder, prompt enhancer, Akatz, Eros, Turbo, Spectrum or Spark in these added graphs. Background protection is not implied by the word replacement.

## Permanent repository installation

Download **update_h3_image_community.py** to your computer's Downloads folder. In GitHub Desktop, choose `runpod-comfy-stateless`, then **Repository > Open in Terminal**. Run:

```bash
python3 "$HOME/Downloads/update_h3_image_community.py"
```

This installer supports Python 3.8+. It adds new files only. It refuses to overwrite a differing existing file and validates all destinations before writing. It does not stage, commit, push, fetch, or modify anything remotely. The original workspace must already be present.

Review and commit in GitHub Desktop, then manually Push origin.

Start a **new** run of **Build Native H3 Image Workspace** on main. Do not rerun an old commit. Its existing CPU schema checks include the new API templates; new tests also check native sampler constructors on CPU inside the built image.

After the build succeeds, use its new image digest on RunPod. Preserve `/workspace`, the existing password, ports `8188,8888`, and blank start-command/entrypoint overrides. Your old native workflows stay intact.

At first launch these new files are added to:

```text
Workflows > 00_H3_Image_INT8_Encoder_v3
```

Look for prefixes **10 through 18**. The existing startup copies missing files without overwriting user-saved workflows. If you have already saved a different copy under the same name, use the downloadable JSON directly rather than expecting a restart to overwrite it.

## Optional Pruna decoder test

Only workflow 18 needs this. The matching file is:

```text
p-h3-image_vae_int8_convrot.safetensors
```

Its published size is 3,146,996,440 bytes, approximately 3.15 GB. No Pruna diffusion model, encoder, prompt enhancer or BF16 base is downloaded.

After rebuilding, run inside the **Pod's Jupyter terminal**:

```bash
python3 /opt/h3-image/community/manage.py download-vae
```

For immediate testing without rebuilding, upload the supplied standalone **setup_h3_image_community.py** to `/workspace` and run:

```bash
python3 /workspace/setup_h3_image_community.py download-vae
```

The tool reuses a checksum-matching file at `/workspace/h3-still/models/vae/` when present; otherwise it downloads the exact published revision and verifies the SHA-256 before registration. Existing mismatched files are not overwritten. It uses a persistent symlink to avoid duplicating the model on disk. Allow 3.15 GB plus 4 GiB free headroom. Existing licenses still apply.

Refresh ComfyUI model definitions, then open workflow **18**. Both native and Pruna outputs come from ONE sampling pass. Save A and B separately; the smaller side-by-side comparison is not the full-resolution final image. A is native on the left, B is Pruna on the right.

This is a decoder-only experiment. It intentionally does not claim to reproduce a workflow that replaced the reference encoder and decoder together. Cross-decoder output quality and GPU execution remain untested here.

### Optional node preflight

```bash
python3 /opt/h3-image/community/manage.py status
```

Or, using the standalone script:

```bash
python3 /workspace/setup_h3_image_community.py status
```

This reads the local ComfyUI node schemas and checks the LCM controls. It does not queue a generation, install packages, restart ComfyUI, change settings, or prove that 16 MP fits.

## Results and recovery

Final PNGs save under:

```text
/workspace/h3-image/output/H3_Image/Community_Tests/
```

Timing reports remain under `H3_Image/reports/`. The reports record actual dimensions, step count, seed and timings. Effective sampler, noise and VAE settings are in the workflow embedded in each saved PNG. The `run_label` and filename describe the original preset, so consult actual values after changing sliders.

Run one job at a time. If a 16 MP run fails, keep the error log and lower output size rather than starting duplicate ComfyUI processes. While idle, your existing controls are unchanged:

```bash
python3 /opt/h3-image/control.py free
```

```bash
python3 /opt/h3-image/control.py restart
```

Do not assume an OOM warning is only GPU memory; container RAM can also be exhausted. Preserve your data and existing model cache. No new memory workaround or automatic package update is installed.

## What has and has not been verified

See VALIDATION.md. Offline structural, safety and installer tests are not GPU generation or visual-quality tests. The supplied sample picture is not enough to establish the original prompt, seed, sampler, VAE, checkpoint precision or generation resolution.

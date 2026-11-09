# H3 Studio Lab: install once, test one change at a time

Release 2026-11-08.1. Prepared against repository commit `c8ad356ff6f7d2ca327c0395b21ec608062b900c`.

This is an additive extension of your existing H3 Still Studio. It does not replace the working Studio source, old workflows, model cache, login gateway, Jupyter, file browser, or other workspaces. Your original build action and `h3-still` image tag are unchanged. Experimental results are not presumed better.

## 1. Apply on your Mac

Download `update_h3_studio_lab.py` into Downloads. In GitHub Desktop select `runpod-comfy-stateless`, then Repository > Open in Terminal:

```bash
python3 "$HOME/Downloads/update_h3_studio_lab.py"
```

The installer supports Python 3.8 or newer. It adds files only, checks the complete payload before writing, refuses conflicts, and makes no network requests. It does not fetch, stage, commit, push, build or deploy. Run with `--dry-run` for a preview. Running it twice is safe. An explicit `--repo /path/to/repository` is also supported.

Review and commit in GitHub Desktop, then manually **Push origin** there. Do not rerun any older repair installer afterward.

## 2. Build the stable Lab profile first

In GitHub Actions choose **Build H3 Studio Lab**, not the old build action. Click Run workflow, leave **profile = stable**, and start a new run on the updated branch.

The action builds on your previously successful Studio image. It resolves that image to its actual digest and verifies the original revision label. It downloads pinned optional provider code, not model weights. It then performs real ComfyUI CPU schema/preprocessing checks, native conditioning contract checks with synthetic encoder/VAE fixtures, provider import/signature checks, and the existing HTTPS/Jupyter browser/terminal test before publication.

Use the **new image@sha256 reference printed in that successful run's summary**. No image from this package has been built or published by ChatGPT. The rolling stable Lab tag is `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still-lab`, but prefer the actual digest.

### RunPod settings

| Field | Value |
| --- | --- |
| GPU | Keep your A100 80 GB, or use the A40 48 GB test configuration |
| Host driver | Compatible with the existing CUDA 13.0 image |
| System RAM | At least 128 GB recommended as a planning target |
| Container disk | 30 GB |
| Persistent volume | Preserve `/workspace`; 200 GB recommended when also testing P-H3 |
| HTTP ports | `8188,8888` |
| TCP ports | None required |
| Start command / entrypoint overrides | Leave blank |

Keep the existing variables:

```text
COMFY_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Only COMFY_PASSWORD is required. Access paths are unchanged: `/__h3/login-v3` and `/__h3/files`. Port 8888 is the independent file browser/Jupyter gateway. No new password is introduced. Old models may still need their normal initialization if you attach an empty volume.

Do not terminate a Pod containing needed data on an ordinary Pod volume. This update does not stop GPU billing. No recommendation here is a measured A40 VRAM or speed guarantee.

## 3. First test: retain your known-good baseline

Open **Workflows > 00_H3_Studio_Lab_v1**.

First open **00_Known_Good_Replacement**. This is the same diagnostic JSON that you reported worked well. Upload the same target, face and body photographs and use the same successful prompt.

Then open **02_Replace_One_Person**:

- Picture 1: target scene.
- Picture 2: face/head reference.
- Picture 3: body reference.
- Leave video OFF, personal LoRAs OFF, accelerator None and character-swap adapter OFF.
- Keep 2.5 output MP, Quality at 50 steps, reference caps at 2.5 MP, and the fixed seed.
- For a matched comparison, turn **use_role_builder OFF** and paste the exact successful prompt into **request**.
- Run once. Inspect **A. RAW H3** and **B. FINAL**. They should match when background protection is OFF.

The enhanced baseline delegates model loading, reference conditioning, sampling and decoding to the same upstream implementation. Only GPU comparison on your Pod establishes actual numerical/image equivalence.

### How the new controls work

**One output MP control:** `output_megapixels` controls whichever mode you selected, including Replace person. The old confusing Generation MP versus Edit MP distinction is removed in the Lab controls, not changed in your old workflows.

**Per-reference crops and MP caps:** each supporting picture has its own crop-margin percentages and cap. Leave all crop margins at zero to keep the complete photo. Crop previews are shown for the first three inputs. The report also records the final dimensions after the native match/max resize. An unused photo remains absent; the small preview placeholder is not supplied to H3.

**Visible role builder:** `face_slot = 2` and `body_slot = 3` refer to upload slot numbers. The builder resolves active-slot numbering into correct Picture tags and fails clearly if a selected role points to an empty slot. `target_description` identifies the single target person. `match_lighting` adds one literal lighting instruction. No LLM or automatic subject description is used. The exact prompt appears in its own report.

**Manual step test:** keep Quality and set `manual_quality_steps` to a desired count, such as 25. Zero preserves 50. It does not silently change eight-step Turbo. Compare the same seed, references, resolution and prompt before keeping a lower count.

## 4. Workflow list

| File | Purpose | Extra download |
| --- | --- | --- |
| 00 Known Good Replacement | Unmodified successful diagnostic graph | None |
| 01 Studio Generate/Edit/References | No-image generation, one-image editing or photo/video reference composition | None |
| 02 Replace One Person | Role mapping, crops, resolution controls, raw/final and timing | None |
| 03 Replace One Person SAM | Same replacement with SAM selecting the person | SAM if not already downloaded |
| 04 Spectrum Test | Replacement with Spectrum as the single experiment | Provider code included in image |
| 05 Character Swap Test | Replacement plus Akatz V1.5 adapter | About 155 MB |
| 06 Spark A40 Test | Replacement plus isolated Spark SM86 backend | About 18 MB release ZIP, GPU setup required |
| 07 Original Reference Contact Sheet | Arrange original references without generation | None; does not load H3 |
| 08 Generated Character Sheet | Generate additional views as an interpretation | None |
| 09 Save RefMod | Save one preprocessed reference with the H3 VAE | Provider code included in image |
| 10 Generate From RefMods | Load saved face/body references and generate | No new base model |
| 11 P-H3 INT8 Comparison | Separate native P-H3 image pipeline | About 29.5 GB and the ph3 build profile |
| 12 SAM Mask Only | Check the selected person before H3 sampling | SAM if missing |

For two different replacements in one scene, start with two controlled passes: replace A, then use that final output as Picture 1 to replace B with a different mask and references. This package does not claim a validated four-person simultaneous identity-binding mechanism.

## 5. Optional tests and exact Pod commands

Run these in **Jupyter > Launcher > Terminal on the Pod**, not in your Mac repository terminal. Downloads are revision- and SHA-256-locked and refuse to overwrite conflicting existing model files.

### SAM

```bash
python3 /opt/h3-still/lab/manage.py download sam
```

The model is `sam3.1_multiplex_fp16.safetensors`, about 1.75 GB. Existing matching files are reused. The checkpoints search path is registered at every Lab start, so the old manual symlink repair is no longer needed for the Lab image. H3 base models remain INT8.

Run **12_SAM_Mask_Only** first. White should cover only the person being replaced. Then use **03_Replace_One_Person_SAM** with the same short target description. That description is connected to both SAM and the replacement prompt. SAM selection can be wrong; check overlapping hands/arms. Margin defaults to zero. Background preservation is final compositing, not H3 masked inpainting.

### Spectrum

Open **04_Spectrum_Test** with the same photos, prompt and seed as 02. No terminal installation or model download is needed. This uses the pinned Spectrum provider with the native ER-SDE single-pass/full-mode settings, not an arbitrary stack of accelerators.

The actual/forecast messages in the run report and Comfy log are the evidence that acceleration ran. A selected node or a successful render alone is not proof of speedup. Compare sampling seconds and image quality. First run loading/compilation overhead and cached reference preparation affect total runtime.

### Character Swap V1.5

```bash
python3 /opt/h3-still/lab/manage.py download swap
```

Refresh model definitions, then use **05_Character_Swap_Test**. It starts with the adapter enabled at 1.0, no Turbo and no accelerator. This adapter's documented purpose is character transfer in video; its still-image behavior remains an experiment. Partial LoRA warnings still need inspection in `h3-still/logs/comfy.log`; a positive patch count is not complete compatibility proof.

### Spark, A40/SM86 only

On an A40, after finishing any active generation:

```bash
python3 /opt/h3-still/lab/manage.py setup spark
```

The command verifies the published archive checksum, installs only the isolated `spark-h3-kitchen` 0.2.37 wheel and invokes the official installer's GPU probes. It checks that torch and the official comfy-kitchen versions did not change. Restart the Pod once after successful setup, then open **06_Spark_A40_Test**.

Do not run this test on A100. The package explicitly rejects SM80. On a recreated compatible container, the Lab launcher restores the already-downloaded wheel from persistent storage without a new network download. A failed optional restore does not block the login/file browser.

Spark defaults to fidelity, uses the actual plan step count, and checks for incompatible execution rather than silently substituting an unrelated kernel. A tiny still can be below acceleration thresholds; retained provider logs and measured sampling time determine whether it helped.

### Saved RefMods

Open **09_Save_RefMod**, upload the face photo, check its crop, choose **Full reference**, and save as `person_face_v1`. Repeat with the body photograph using `person_body_v1`. Save each original separately so they do not become a temporally stacked/cropped shared canvas.

Refresh model definitions and open **10_Generate_From_RefMods**. Select those files. The mapping report shows which one becomes Picture 1 or Picture 2. A saved RefMod is a reference encoding, not a trained identity model. Full mode is the baseline. Compressed experimental uses pooling only and can lose facial detail. The vision encoder reconstructs reference pixels from the saved representation, so parity or a speed gain is not assumed.

Files are stored under `/workspace/h3-still/models/refmods/studio_lab`. Existing names are protected from overwrite; increment the version name.

### P-H3 INT8

This needs a different ComfyUI core, so keep it separate from the stable build. Run **Build H3 Studio Lab** again with **profile = ph3**. This publishes a separate `h3-still-lab-ph3` tag and its own actual digest. It does not replace stable Lab or the original H3 image tag.

Use that successful image digest with the same ports/password and persistent volume. Then in its Jupyter terminal:

```bash
python3 /opt/h3-still/lab/manage.py download ph3
```

This adds the INT8 diffusion model, INT8 encoder and INT8 P-H3 VAE, about 29.5 GB. It does not download a BF16 base or prompt upsampler. Open **11_P_H3_INT8_Comparison**. Start with one or two references at 4 MP. This uses native P-H3 conditioning, shift 1/1, the published eight-step sigma grid, Euler and the matching native decoder, not Fizgig's video-VAE decoder. The first image is the edit source when images are supplied. Set appropriate reference roles directly in the prompt.

This is a comparison workflow, not a proven superior replacement. It has eight photo slots and no video input. Switching back to the stable Lab image preserves all files; do not terminate needed ordinary-volume storage.

## 6. Compare and send results

Use the same photographs, crops, prompt, seed, output size, reference-size setting and sampling schedule when testing one accelerator/adapter. Changing run_label forces the timed sample to execute again even when its other inputs are cached. Preparation time in the report belongs to the last conditioning-node execution, not necessarily the current queue's total time.

Outputs are saved under `/workspace/h3-still/output/H3_Lab/`. JSON reports are under `H3_Lab/reports/`. They include reference geometry, prompt hash, sampler steps, GPU allocation peaks, sampling/decode timing and captured provider log lines. Peak allocated/reserved memory is PyTorch process memory, not total GPU utilization. The report does not infer successful acceleration merely from the selected option.

For an experiment that fails or changes likeness, keep the baseline PNG, experiment PNG, saved workflow JSON and corresponding JSON report. Do not add more patches to compensate before comparing the single change.

Useful status command:

```bash
python3 /opt/h3-still/lab/manage.py status
```

The GUI JSONs require the Lab image. Importing them into the old image without these new helper nodes is not supported. Old user workflows still open in Lab.

## Validation and limitations

`LOCAL_VALIDATION.md` records checks actually executed while preparing this package. Docker builds, native provider probes and the browser integration test run in your GitHub Action. GPU generation, exact identity, speed improvements, maximum resolution/reference capacity and the live RunPod instance were not tested in the preparation environment. Experimental dependencies are pinned, but a signature/import test does not prove numerical compatibility.

The package ships no model weights, API keys, credentials or fonts. Downloaded sources retain their upstream licenses and attribution. H3/P-H3 model licensing and territorial requirements discussed previously remain unchanged. This package does not grant model rights. Third-party reference libraries are private user data and are not uploaded by the local installer.

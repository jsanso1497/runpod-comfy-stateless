# ComfyUI Quality 3.2: one headshot plus multiple body references

This is the complete corrected Krea-only package with a new ONE-person H3 RefMod path.
It includes the earlier updates. Do not apply old ZIPs after this one.

## Update GitHub

1. Unzip this package. Upload its CONTENTS into the root of `jsanso1497/runpod-comfy-stateless`, merging folders and replacing matching files. Do not upload the ZIP or an enclosing parent folder.
2. Keep your existing `.github/workflows/build-image.yml`; the included copy is unchanged.
3. Commit to `main`. Wait for the build for that final commit to finish successfully.
4. Keep your RunPod image at `ghcr.io/jsanso1497/runpod-comfy-stateless:latest`. Keep every environment variable, secret reference, disk setting and port from the corrected 3.1 setup. No new model or LoRA is required.
5. Download any old Pod results you need, terminate that Pod, then deploy a fresh Pod after the build succeeds.
6. Confirm this new startup banner:

```text
ComfyUI Quality 3.2 | One Person Multi-Reference + Krea-only Rebalance
```

An older banner is not this update. Wait for `WORKFLOW SCHEMA PASS` and model preparation before running a workflow.

## Test RefMod first

### A. Create the reusable person reference

Open `08_H3_Create_Person_RefMod`.

1. Upload your primary face/head image into **1. PRIMARY HEADSHOT - face and hair**.
2. Upload your main full-body photo into **BODY 1**. This photo is the wardrobe authority and a body-proportions reference.
3. Upload additional photos of that SAME person into **BODY 2** through **BODY 8**, as needed. Alternate full-body and side/back views can be used. Leave unused slots at `(none)`. There is no need to delete or reconnect nodes.
4. Set `person_label` in **2. ONE PERSON - name the reference set** to `test_person`, or another short label.
5. Leave `ref_resolution=2048`, `max_total_tokens=131072` and `save=true` unchanged for the first run.
6. Click Run. Check the displayed role map and the saved details. All selected views must be reported, with the headshot first.
7. Download the ZIP from **5. DOWNLOAD person bundle backup ZIP**. The ZIP backs up all saved references in the registered RefMod folder, not just this set.

The saved selection will look like `identities/test_person_<checksum>`. It is ONE file containing separately encoded images with their individual aspect ratios and head/body role labels. It is not a collage, a trained LoRA, an averaged face, or a temporal stack of photographs.

The reference-resolution cap is applied independently to each photo by the existing H3 VAE/RefMod code. These are VAE representations, not lossless originals. An excessive token budget raises an error rather than dropping photos or changing resolution automatically.

### B. Render a short test using the saved bundle

Open `25_H3_One_Person_Saved_RefMod_Quality`.

1. In **1. Choose ONE person bundle in mod_1**, click **Refresh RefMods** and select `identities/test_person_<checksum>` in `mod_1`.
2. Keep `strength_1=1.0`, `copies_1=1`, and `components_1=Visual`. Leave EVERY other `mod` slot at `(none)`. The one selected file already contains all the photos.
3. In **2. ONE person instruction + role checks**, keep the supplied simple motion instruction for your first run. It requests one full-body shot with a small natural weight/head movement. Replace that text later to direct another shot.
4. Keep the supplied 1344x768 canvas, 124 frames, 24 fps, 25 steps, `res_multistep`/`normal`, seed 42, and optional H3 LoRA OFF.
5. Click Run. Review the video in **6. Result video - ONE person**.

This workflow does NOT call Ollama. It automatically groups every selected picture into `<Subject 1>` and gives `<Picture 1>` facial priority. The first supplied body view determines wardrobe when outfits differ. Those priorities are prompt instructions, not hard mathematical guarantees that a face cannot drift.

Do not add an Apply RefMod node to this saved-bundle workflow. Its encoder already attaches the references once.

Start with one headshot and two complementary body photos before filling all eight body slots. More distinct views may supply useful information, but each also adds inference cost. Duplicated, conflicting or low-detail images are not an automatic quality improvement. Exact duplicate images are rejected to avoid accidentally repeating a reference.

## Work directly from the photographs

Open `24_H3_One_Person_Multi_Ref_Quality` to upload the same head/body set and render without a separate creation run. It also saves the bundle.

- The original photos are sent to H3's image-aware encoder, in the same order as the RefMods.
- Each reference is encoded independently and applied once.
- Keep `Use instruction (no Ollama)` for the first test.
- `Generate with Ollama` is available in the prompt node. The local abliterated helper receives EVERY selected photo, not just the first two, and unloads before H3's loaders are released. This path uses 32K helper context without shrinking its existing preview-size setting. It is more memory-intensive and has not been benchmarked here.
- `09_H3_One_Person_Prompt_Draft` writes only a prompt from that same photo set. It does not load H3 generation weights during execution. With unchanged images/order, paste its reviewed result into workflow 24's instruction field and keep instruction-only mode to avoid asking Ollama again.

## What remains unchanged

Krea's Raw BF16 + Identity Edit + Krea-only Rebalance workflows; H3's full BF16 model, 25-step schedule and no Turbo; the abliterated 32B FP16 helper; SeedVR2; the HTTP fix; all model/LoRA catalogs and keys; and the earlier two-person H3 workflows.

Use workflows 08/24/25 for ONE person with many photos. The old two-person workflows still mean two different subjects; they are not the right graphs for this test.

Download your RefMod ZIP, outputs and original photographs before terminating the Pod. Nothing in this update automatically stores them in GitHub or persistent storage.

See `VALIDATION.md` for executed checks and what still requires your Pod.

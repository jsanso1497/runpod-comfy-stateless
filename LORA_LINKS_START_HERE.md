# Your LoRAs: paste links, not JSON

Use this package INSTEAD OF the previous H3 Portrait ZIP. It contains that template
plus the shared link-list update for your existing `:latest` template.

## 1. Add your links

Open `config/lora_links.txt` in this package. Paste one REAL Civitai or Hugging Face
LoRA URL on each new line. The supplied file contains only comments, not fake model
links that you need to fix. No SHA, model ID, file ID, profile, destination or JSON
fields are required.

- Civitai: a model page, version-specific page, or the actual file's Download link.
  A plain model page selects the newest published version and its primary safetensors.
  To get a specific precision such as FP32, copy THAT file's Download link.
- Hugging Face: open the desired `.safetensors` file and copy its page URL. Both
  `blob` page links and `resolve` download links work. A repository page also works
  when it has exactly one safetensors file. For a repo with several files, choose the
  file on the website and copy its link; do not enter numeric IDs.

Blank lines and lines beginning with # are ignored. Never put an API token in a
URL. Your existing HF_TOKEN and CIVITAI_TOKEN RunPod Secrets are reused.

Both templates download EVERY link into ComfyUI/models/loras/Shared/. They do not
filter this list by MODEL_PROFILES. Duplicate file content is downloaded once per
shared-list sync. Checksums, provider filenames and Civitai trigger words are
resolved internally. They are not extra fields for you to fill out.

## 2. Upload this batch to GitHub

Upload ALL CONTENTS of this ZIP to the ROOT of:
`jsanso1497/runpod-comfy-stateless`

Merge folders and replace matching files. Do not upload the ZIP or an enclosing
parent folder. Specifically include:

- `Dockerfile` and `scripts/bootstrap.sh`: replace these existing files so the
  normal `:latest` image reads the shared list too.
- `config/lora_links.txt`: add this file with your links.
- `shared_loras/`: add this entire folder, including its tests.
- `h3_portrait/`: upload/replace this complete folder with this version.
- `.github/workflows/build-h3-portrait.yml`: create/replace this action. It now
  rebuilds when the shared list changes, not only when portrait files change.

Do NOT change `.github/workflows/build-image.yml`, `.dockerignore`,
`scripts/prepare_image.sh`, `scripts/check_rebalance_runtime.py`, existing model
catalogs, or your other workflows. Their previous disk-space/build-check fixes stay.
Do NOT replace or empty `config/loras.json`: it still contains built-in required
adapters such as Krea Identity Edit. Add YOUR future LoRAs only in lora_links.txt.
The old `h3_portrait/loras.json` is now ignored and the included copy is empty.

Commit to main. Wait for BOTH applicable builds for that commit:
1. Build RunPod ComfyUI image (publishes latest)
2. Build H3 Portrait template (publishes h3-portrait)

## 3. RunPod changes

No new environment variables, keys, ports or settings are needed for this patch.
Keep the existing template on `:latest`. Keep the portrait template on
`:h3-portrait`. Fresh Pods are required to use the new built images/list.
Do not install this by restarting an old Pod image or by rerunning an old commit.

If you have not made H3 Portrait - Automatic yet, follow H3_PORTRAIT_START_HERE.md.
Its settings from the previous batch are unchanged.

## 4. Select your LoRA in ComfyUI

In H3_Portrait_Auto, on **2. Tell Ollama what you want**:

- Select your matching H3 LoRA in `lora_1`.
- Set `strength_1` to the LoRA author's recommended value. The initial 1.0 is a
  neutral UI starting value, not a verified best strength for your adapter.
- Leave `lora_2` at `(none)` unless you intentionally want a second compatible LoRA.
- Keep `use_loras=true` to apply the selected files. `(none)` means no adapter;
  merely downloading a file does not select it.
- Civitai trigger words are inserted for selected LoRAs when supplied by Civitai.
  For an HF file needing trigger words, enter them in `extra_trigger_words` or your
  instruction. HF triggers are not guessed from filenames or tag-frequency metadata.

Then upload your references, describe the video normally, choose 9:16 or 2:3, and
Run. The prompt, quality presets, video weights and reference handling are otherwise
unchanged.

In your existing Krea/H3 workflows, use their existing Optional LoRA dropdowns.
The same Shared/ filenames are available there. Keep the Krea Identity Edit adapter
on; select user adapters separately. Use the author's strength and trigger words.

Availability is NOT cross-model compatibility: a Krea LoRA does not become an H3
LoRA by being in the shared folder. Only chosen files are patched into the model.
FP32 LoRA file storage does not make the whole base model run in FP32.

## 5. Later additions

Edit ONLY `config/lora_links.txt` in GitHub, paste additional links, and commit.
Wait for the corresponding image build(s), then deploy a fresh Pod with the same
tag. No extra IDs or hashes are needed. Running Pods do not receive new downloads
from a GitHub commit. Removing a line prevents its shared-list download on future
fresh Pods; it does not delete other catalog-managed adapters or running-Pod files.

Watch for `SHARED LORA LIBRARY: ... ready, ... failed` at startup. An inaccessible
link is reported by line number. Other files and the UI can still load; never
assume a failed item is available. Larger lists use more startup time and disk.

The actual LoRA link has not been provided. Its access, target model and visual
results have not been verified. This package was prepared from the supplied files;
no changes were committed to your live GitHub repository here.

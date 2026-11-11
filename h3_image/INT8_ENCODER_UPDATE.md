# Native H3 Image: larger INT8 encoder default

Release: 2026-11-11.1. Prepared against repository commit `41df6ce84e4d1b68bb9bc476f9e301954cdf26de`.

## Apply locally

Save `update_h3_image_int8_encoder.py` in Downloads. In GitHub Desktop select
`runpod-comfy-stateless`, then Repository > Open in Terminal:

```bash
python3 "$HOME/Downloads/update_h3_image_int8_encoder.py"
```

Python 3.8 or newer; no Python packages are installed on your Mac. `--check` previews
changes without writing. `--repo /path/to/repository` selects a checkout explicitly.
The updater accepts only the reviewed original files or files already updated by this
package. Unrecognized edits stop the entire installation before any changes.
Backups are kept under the checkout's Git metadata directory, outside tracked files.
It never stages, commits, pushes, downloads models, or edits GitHub remotely.

Review and commit locally using GitHub Desktop, then manually Push origin.

## Build and deploy

Run a NEW GitHub Actions run: **Build Native H3 Image Workspace**, branch **main**.
Do not rerun an older failed job. Use the successful NEW run's actual `@sha256` image
reference in RunPod. No digest is invented or supplied by this package.

Preserve the current `/workspace` storage, `COMFY_PASSWORD`, HTTP ports `8188,8888`,
and blank start-command/entrypoint overrides. The old running container does not
change merely because GitHub has been updated. Save its image reference for rollback.

On first startup with the new image the existing downloader prepares:

```text
qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

The encoder is **27,141,342,152 bytes**, approximately **27.1 GB (25.28 GiB)**.
Allow **35 GB free on /workspace** before an upgrade when that file is absent.
The complete new encoder must be downloaded; this is not an 11 GB patch. Existing
NVFP4 files remain in place. The same diffusion model and VAE are reused when they
match the newly built manifest. Models are downloaded on the Pod, not baked into
Docker and not downloaded on the GitHub Actions runner.

The build verifies the encoder's published size and SHA-256, records an immutable
model repository revision, and startup verifies the downloaded bytes. A matching
INT8 encoder already under the native workspace is reused. A matching INT8 copy
under `/workspace/h3-still/models/text_encoders` may be linked without modifying it.
NVFP4 is never renamed, substituted or silently used as a fallback.

## Open the updated workflows

```text
Workflows > 00_H3_Image_INT8_Encoder_v3
```

All **six standard and nine community** workflows installed from the repository now
select the INT8 encoder. The v3 folder is separate deliberately: the startup loader
never overwrites a saved workflow, and your existing v2 folder may contain personal
settings. Existing v2 workflows, current browser tabs, imported workflow JSONs and
workflows loaded from older PNGs keep their own saved model selection.

To use a previous saved or imported workflow, select this file in its **CLIPLoader**:

```text
qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

Leave `type = minimax` and `device = default`. There is no need to replace the whole
graph to change the encoder. The optional `H3_Image_INT8_Encoder_Workflows.zip` also
contains updated copies of the previously supplied restoration and video-reference
tests, for direct import after the new encoder is available. Those supplemental JSONs
are not added to the repository automatically.

No prompt enhancer, prompt builder, BF16 diffusion checkpoint, new sampler, extra
LoRA, custom-node pack, or package upgrade is added. Reference order, seed, prompt,
resolution, sampler and VAE selections are preserved. Community graphs remain at
16 MP; standard graphs start at 4 MP, exactly as before this update.

## First comparison

Start with one source image at **4 MP** and the same fixed seed, prompt, sampler and
VAE as your earlier test. Compare identity, expression and reference adherence before
sharpness. If the larger encoder causes memory pressure, reduce the workload or use
more memory rather than assuming the stored file size is its exact VRAM requirement.
The update does not establish that INT8 improves quality or speed.

Log and optional presence check, on the Pod:

```bash
tail -n 40 /workspace/h3-image/logs/comfy.log
ls -lhL /workspace/h3-image/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

The definitive selected encoder is the filename in the graph's CLIPLoader. File
presence alone is not proof that an older workflow has switched to it.

Existing controls, authentication, Manager and file browser are unchanged. Do not use
old `/opt/h3-still` restart commands for this native-image workspace.

## Revert

Your original container image and saved workflows remain available. Switching back
to that image on the same persistent volume restores the previous defaults. Do not
terminate storage containing needed files. To undo only the local repository update,
review/discard its uncommitted changes in GitHub Desktop or restore the files from
the backup location printed by the installer. Do not discard unrelated work.

## Provenance and limitations

- Official file: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
- Raw LFS pointer: https://huggingface.co/Comfy-Org/MiniMax-H3/raw/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
- Native image model-card: https://huggingface.co/Comfy-Org/MiniMax-H3-Image
- SHA-256: `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923`.

Local checks are recorded separately in the delivered validation file. No Docker
build, full encoder-weight download, GPU generation, runtime-memory measurement,
quality comparison, or live-Pod modification was performed here. The existing native
CPU/browser publication checks stay enabled. Existing model license requirements
continue to apply; this update grants no additional model rights.

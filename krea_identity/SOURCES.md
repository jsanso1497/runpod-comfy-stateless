# Source and implementation notes

Verified against the attached source snapshot on 2026-10-06. The live user
repository was not changed. A GitHub connector lookup for its BUILD_INFO.json
returned 404, so the uploaded archive is the baseline for this update.

## Primary sources

- Krea official inference repository: https://github.com/krea-ai/krea-2
- Identity Edit model card and documented input order/settings:
  https://huggingface.co/conradlocke/krea2-identity-edit
- ComfyUI-Krea2Edit source at the SAME pin already used by the attached repository:
  https://github.com/lbouaraba/comfyui-krea2edit/tree/86f886dac23013d88996e3a2e99093ba44d322fb
- Screenshot node implementation, same existing pin:
  https://github.com/nova452/Rebalance-Pack/blob/53c147c72c2fbd9444765af87118caee81a26d01/krea2.py
- Turbo BF16 weight upload and SHA-256:
  https://huggingface.co/Comfy-Org/Krea-2/commit/a267023ad1b5d57600de22dfad8f77eed3beaa16
- Full Identity Edit v1.2 upload and SHA-256:
  https://huggingface.co/conradlocke/krea2-identity-edit/commit/55bdbc7985fe5a9bc8e0f179a5101bbe32c98086
- SeedVR2 documentation: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler

## Deliberate boundaries

The stock editor uses scene-first/subject-second image grounding and VAE
conditioning. The new three-upload helper makes a sheet from two ORIGINAL
photographs so those same two conditioned inputs can be retained. It does not
modify the upstream neural network, add an untested three-reference patch,
train identity weights, or synthesize a replacement reference face.

Reference-sheet support in the model card is not a controlled demonstration of
this exact face/body fusion task. The helper layout, 1.5 MP working size,
rebalancer-off default and separate approval/upscale steps are implementation
choices, not claimed benchmark winners.

The node in the screenshot is a per-layer conditioning scaler. It is not the
separate TuZZiL adaptive/normalized node. This package does not infer which
particular layer controls identity, pose, aesthetics or safety.

SeedVR2 SOURCE comes from /opt/seedvr2 in the repository's existing digest-pinned
base images. Its node schemas/settings are based on the attached repository's
already-versioned SeedVR2 workflows. This update does not replace it with a
rolling upstream checkout. Docker's new schema smoke check must validate the
actual inherited copy before publishing the image.

No third-party model weights are included in this source package. The small
local helper code is new; editing and rebalancing stay in the upstream packs.
Private user LoRA links are outside the patch and excluded from source hashing.

# H3 BF16 high-precision profile

Release: 2026-11-01.2. Base reviewed: `0a0f8b0fc29e88a37ef027e4474dc16cf1a6b743`.

## What this is

A separate image, `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap-bf16`,
built by **Build H3 BF16 high precision**. The existing character-swap image tag is
not replaced by that action. No image is published until you push and run the build.
The package includes the pending workflow recovery, native previews and automatic
FileBrowser changes. It can install over the reviewed source or that previous update.

This is unquantized, original-source 16-bit inference, NOT FP32 throughout and NOT
an established visual-quality winner. The Akatz author's published qualification
uses the quantized baseline. We preserve that option. Compare the same short shot,
reference, mask and seed. More precision does not prove better character adherence.

## Model choices

| Component | BF16 image |
| --- | --- |
| H3 Ref2VA | `minimax_h3_ref2va_bf16.safetensors`, original full/unpruned BF16, 66,280,487,368 bytes |
| H3 reference/text encoder | `qwen3vl_32b_minimax_h3_bf16.safetensors`, BF16, 51,506,295,256 bytes |
| Attention | `pytorch attention`, replacing the existing `comfy_kitchen_int8` path |
| Video VAE | Existing original `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | Existing `minimax_h3_audio_vae_fp32.safetensors` |
| Person masks | Existing `sam3.1_multiplex_fp16.safetensors` |
| Character adapter | Original Akatz V1.5 release, strength 1.0 |
| Turbo adapter | Original FL2V Turbo 8-step BF16 release, strength 1.0 |
| Optional analyzer | `huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16`, the user's preferred former helper |
| Preview | TAEH3 tiny decoder, only for approximate previews, never final generation |

The two large new files total 117.79 GB (109.70 GiB). The analyzer is a separate
approximately 67 GB download. Reused matching files are not downloaded again or
silently overwritten. The build resolves exact revisions, file sizes and hashes;
startup verifies bytes before use. SHA256 is checked against the published original
BF16 files, not a file obtained by converting low-bit weights back to BF16.

The image sets `--bf16-unet`, `--bf16-text-enc` and PyTorch attention. It does NOT
force FP32 everywhere, enable GPU-only residency, use a quantized fallback, add an
upscaler, or change the matched 8/er_sde/normal/CFG 1 recipe. SAM, VAE and LoRA
checkpoint dtypes are reported from actual safetensors headers on the Pod instead
of being guessed from their names.

## RunPod

Recommended starting host: **one NVIDIA B200, 180 GB VRAM**, with at least **256 GB
system RAM** and **24 vCPUs**. RunPod currently lists B200 with 283 GB RAM and 28
vCPUs. H200 141 GB with normal offloading is a fallback candidate, not a measured
minimum or equal-speed claim. No H3 peak memory or render-speed benchmark was run
in the authoring environment. One short generation still has to fit.

The retained image uses pinned PyTorch 2.11.0 / CUDA 12.8. B200 is supported by that
CUDA architecture generation. Do not select B300 solely for more VRAM on this image:
its newer architecture needs a separate compatibility qualification. Two GPUs are
not a pooled-memory substitute here; this recipe has one GPU execution path.

Use **500 GB** network storage for a fresh installation, or **1,000 GB** when keeping
the older library, the 32B helper and several multi-pass masters. These are planning
allowances, not measured disk minima. Keep at least 100 GB free after weight setup
for initial long/multi-pass trials. Downloads, generated masters and rejected
attempts consume disk. Use an 80 GB container disk; models live on `/workspace`.

Create/choose a network volume in a data center with the desired GPU, mounted at
`/workspace`. A network volume survives Pod deletion; a normal Pod volume does not.
Do not terminate an old Pod until its unique files are secured.

Expose **HTTP 8188 only**, no extra TCP ports. Leave start command and entrypoint
override blank. Keep the existing RunPod secrets. Use the new image digest printed
by the BF16 build, not an unrelated upstream PyTorch image in the RunPod template.

Environment values are in `RUNPOD_BF16_ENVIRONMENT.txt`. This is a build-time image
profile, not an environment switch on the old image. The native/simple workflows
do not call the analyzer even when its service is enabled. Set
`H3_SWAP_PROMPT_ASSIST=0` to skip the optional helper download/service entirely.

## Effective first test

1. Apply this package locally, review/commit and manually push in GitHub Desktop.
2. Run **Build H3 BF16 high precision**, not Build HQ workspace and not the baseline
   Build H3 character replacement. Deploy its successful digest with the volume.
3. Wait for `ready: true` at `/_swap/status`; check `model_profile: bf16_full`.
   The Pod performs a tiny CUDA BF16 matmul/SDPA check and a weight-header audit.
   Neither check is a full H3 memory or quality qualification.
4. Open `H3_Character_Replacement_2026-11-01.2_BF16` in ComfyUI.
5. Start with `02B_Akatz_Native_Simple_Preview`. Use the same single reference,
   exact source section, person-mask settings and seed as the best previous native
   result. It still defaults to roughly 5.17 seconds and the 768 pixel-budget preset.
6. Leave this prompt unchanged for the comparison:

   ```text
   Replace the selected person with the reference character. Preserve source performance, scene and audio timing.
   ```

7. Inspect the full decoded result, not the approximate sampling thumbnail. The
   preview node offers a cancellation hold after the first decoded window.
8. Then try `10_Akatz_Simple_Single_Reference_Chain`, 10 seconds and Review every
   chunk. Start a NEW 48-second run only after both the first chunk and join work.

`02C_Akatz_BF16_HighRef_Preview` is an optional second comparison using
`ref_image_size=max`. It preserves more of a large reference but uses more memory
and time. It does NOT increase output resolution or invent missing reference detail.
Both reference-size variants use the same full BF16 generation weights and sampler.

Existing single-reference/native/mask/analysis/manual-chunk/face-body workflows
remain available in the BF16 factory folder. Their worker recipes are compiled to
the BF16 loaders as well. The two-image mode remains experimental. For the quality
comparison, do not change models, prompt length, two-reference roles, seed and
reference sizing simultaneously.

No 1080p finishing was added. The 768 preset retains its existing area budget, not
literal 768 pixels on every aspect ratio. No model change is claimed to repair the
previous skin-color discrepancy. Chain preprocessing, masks and continuation still
need a comparison against native.

## Recovery, previews and file access

- ComfyUI: `/`
- FileBrowser: `/files/`
- Run status/model preparation: `/_swap/status`
- Chain review/Approve/Retry/Resume/Stop: `/_swap/chain`
- Recover submitted workflows: `/_swap/recovery`

Use the same HTTPS HTTP-8188 service and `workbench`/ComfyUI password for all paths.
FileBrowser is loopback-only behind the authenticated gateway, not public 8080.
The classic FileBrowser release is archived and remains a security limitation;
its command runner is disabled. See FILE_BROWSER.md.

Native MP4s have workflow metadata plus `.workflow.json` and `.api.json` sidecars
when saved by the new saver. Chains save repeat/resume JSONs and final metadata.
Prompts and filenames can be included in metadata: inspect before public sharing.

Run names and saved precision profiles are kept separate. A baseline run cannot
resume into BF16 midway. Finish it in its original image or begin a NEW BF16 run.
Use the new BF16 factory canvases rather than old saved graphs that explicitly
select the quantized filenames. Existing original weights and outputs are retained.

## Verify the installed profile

Run in the Pod terminal:

```bash
python -m swapkit.precision
python -m swapkit.check --live
cat /workspace/h3-swap/precision-audit.json
nvidia-smi
```

The audit records actual weight dtype counts, selected filenames, verified hashes
and byte sizes. It is not a benchmark. While the optional analysis is running:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 ollama ps
```

Expected helper: the 32B FP16 tag, context 32768, ideally 100% GPU. After analysis,
an empty list is normal because the helper is deliberately unloaded before H3.

## Limits

Offline tests cover file/graph contracts, profile isolation, no quantized fallback,
checkpoint header parsing, metadata, source media handling and installer behavior.
They use synthetic media and fake inference for full chain simulations.
No actual H3 BF16 render, complete Docker build, live preview or measured GPU peak
was performed here. GitHub's CPU smoke check is retained; the first GPU render on
RunPod remains necessary. Higher precision and more hardware do not guarantee a
more faithful face, body transfer, natural skin or seamless motion.

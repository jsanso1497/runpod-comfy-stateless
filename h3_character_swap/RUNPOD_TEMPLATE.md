# RunPod template: H3 Character Replacement

Use **Pods**, not Serverless. Create a new template instead of editing the old
H3 HQ template. The image is produced by the new action only after you push the
added files and run **Build H3 character replacement** successfully.

## Image

Expected tag:

```
ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap
```

Best choice: replace that tag with the exact `ghcr.io/...@sha256:...` reference
from the successful action summary. A tag definition does not establish that an
image has already been published. Do not use the old `:h3-hq` image for these graphs.

## All template and Pod fields

| Field | Value |
|---|---|
| Template name | H3 Character Replacement - Akatz V1.5 |
| Deployment type | GPU Pod, Linux x86_64 / amd64 |
| Container image | Successful build's digest, or the tag above after publication |
| GPU | Start with your existing NVIDIA A100 SXM 80 GB |
| GPU count | 1 |
| System RAM | 128 GB or more recommended; 256 GB gives additional headroom |
| vCPUs | 16 recommended for source processing; not a measured minimum |
| Cloud | Secure Cloud preferred for private character/source media |
| Container disk | 40 GB |
| Persistent volume | 300 GB for a fresh setup |
| Reusing the full old H3/model library | Prefer 400 GB total volume capacity |
| Volume mount path | /workspace |
| Network volume | Recommended for surviving Pod replacement/termination; attach at /workspace |
| Expose HTTP ports | 8188 only |
| Expose TCP ports | Leave blank |
| Container start command | Leave blank |
| Entrypoint override | Leave blank |
| Registry credentials | None if the GHCR package is public; otherwise select saved GHCR registry credentials in RunPod |
| Jupyter | Not installed; do not expose 8888 |
| SSH server | Not configured; do not expose 22 |
| ComfyUI backend | Internal 127.0.0.1:8189; do not expose it |
| Ollama | Internal 127.0.0.1:11434; do not expose it |
| Browser connection | RunPod HTTPS proxy link for HTTP 8188 |
| Browser username | workbench |
| Browser password | Value of your comfy_password secret |

RAM and CPU are selected with the GPU host configuration; they are not necessarily
editable fields in the template form. Do not allocate a second separate 300 GB
volume when your selected network volume already supplies /workspace.

These capacity recommendations are planning targets, NOT measured minimums for
this package. The developer qualified the native Akatz recipe on an RTX 4090 24 GB
with normal offloading. Your A100 is a reasonable first test target; an H200 is
not being required by this kit. NVFP4 uses a dequantizing fallback on non-Blackwell
hardware, so memory/runtime do not follow the checkpoint's disk size alone.

The quantized H3 base/encoder are new filenames and do not overwrite your BF16
weights. The same original VAEs and SAM checkpoint are reused when their hashes
match. The optional Qwen3-VL 8B prompt helper adds about 6.2 GB. The EXACT required
model bytes for each variant are recorded in the build artifact's
`build-provenance.json` and `assets.lock.json`. Allow space for downloads and video
outputs rather than sizing the disk to the raw model total.

## Recommended environment block

Add each line as an environment-variable name/value pair in RunPod:

```text
H3_SWAP_VARIANTS=akatz
H3_SWAP_PROMPT_ASSIST=1
H3_SWAP_PROMPT_MODEL=qwen3-vl:8b
H3_SWAP_DATA=/workspace
H3_SWAP_MAX_UPLOAD_MB=1024
WB_AUTH_MODE=basic
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

Secrets:

- `comfy_password`: an existing or newly created password of at least 16 characters.
- `hf_token`: your existing Hugging Face download token. Accept any applicable model
  access terms in that HF account first. Do not put a registry PAT into this field.

No Civitai key, OpenRouter key, OpenAI key or hosted model account is needed.
No secret values belong in GitHub or in the downloaded source package.

Runtime fixes the helper server to a 16,384-token default, one loaded model and
one parallel request. Every actual analysis request also explicitly sets the same
context and output budget and unloads afterward. Do not carry over the old
262K-context helper configuration.

## Optional BFS comparison

Read the custom BFS license first. Then change/add only these variables:

```text
H3_SWAP_VARIANTS=akatz,bfs
H3_SWAP_BFS_LICENSE_REVIEWED=1
```

Keep the other recommended variables unchanged. Restart the Pod with its existing
volume so the two optional BFS adapter files can be prepared. No second image
build is needed. The explicit license-review flag is an installation safeguard,
not legal advice or a statement that any particular commercial use is authorized.

## Without prompt assistance

```text
H3_SWAP_PROMPT_ASSIST=0
```

This prevents starting/pulling the helper. Write the prompt directly in the Akatz
node or the character-description field in BFS. The main character-swap recipes
continue to work without Ollama. Prompt Studio itself requires the helper enabled.

## Startup and first run

1. Use the new successful image, selected GPU, persistent volume and environment.
2. Open the HTTP 8188 HTTPS service and authenticate.
3. Open `/_swap/status` on that same service. Wait for `ready: true`, then refresh
   ComfyUI so newly prepared models appear in the menus.
4. Open Workflows sidebar > H3_Character_Replacement_2026-10-22.1.
5. Run optional 00 prompt analysis, 01 mask check, then 02 Akatz replacement.

For a public repository, the GHCR package may still be private. Make that package
public or supply RunPod registry credentials. A pull error is not an H3 inference
error. The action summary only supplies an image reference after publication.

Application-side upload limit is 1 GB; an external proxy can impose a smaller
limit. Files already in /workspace/input can be selected without another browser
upload. Persistent volume contents survive according to the chosen storage type;
terminate a Pod only after confirming the desired files reside on a network volume
or have been copied elsewhere.


## October 26 low-memory chain update

Rebuild **Build H3 character replacement**, then use the newly published image
reference. All template fields and environment variables above remain the same.
No extra model downloads are introduced. Keep `/workspace` persistent. Allow
additional free disk space for lossless intermediate masters and checkpoints;
50 GB free is a planning allowance for a two-minute trial, not a guarantee.

The new factory folder is `H3_Character_Replacement_2026-10-26.2`.
Open `06_H3_LowMemory_Chain_Control`; monitor/review at `/_swap/chain` on the
existing authenticated 8188 service. Start with 19 seconds and Review every
chunk. Only after inspecting those outputs attempt 120 seconds.

This update does not add FileBrowser startup. Existing 8080 exposure may remain,
but a manually installed FileBrowser still needs starting after a Pod restart.
See LOW_MEMORY_CHAIN.md for control fields, disk requirements and validation limits.

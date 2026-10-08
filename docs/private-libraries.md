# Private LoRAs and custom checkpoints

The excluded private LoRA text file was not read, extracted, copied or committed. The new Docker build accepts no provider secrets or private library data.

## LoRAs

Create the RunPod Secret `comfy_loras`. Its value can be the same kind of private one-link-per-line list you already use. For deterministic compatibility, prefix each link with its supported family:

```text
qwen | https://huggingface.co/OWNER/REPOSITORY/resolve/REVISION/adapter.safetensors
flux | https://civitai.com/models/MODEL_ID?modelVersionId=VERSION_ID
h3 | https://huggingface.co/OWNER/REPOSITORY/resolve/REVISION/adapter.safetensors
```

These are format examples, not working download links. Plain provider links are accepted when metadata can establish the exact supported family. Ambiguous entries are rejected with a request for explicit family metadata rather than guessed from a filename.

The same secret can contain several families. Only entries matching the active workspace are downloaded; the other families are left for their own workspace. Map the secret to `WB_LORA_URLS`. Compatible entries appear in family-specific LoRA dropdowns. Downloading an adapter does not enable it. Strength and selection are saved in your personal workflow, not in an environment variable.

Provider URLs must not embed a token. Authentication comes from your existing `hf_token` / `civit_token` secrets. The runtime supports HTTPS Hugging Face model/file links and Civitai model/version/download links. Other providers and executable pickle checkpoints are not accepted by this downloader.

## Custom checkpoints

Create `comfy_checkpoints` with a JSON array such as:

```json
[
  {
    "name": "my-qwen-checkpoint",
    "url": "https://huggingface.co/OWNER/REPOSITORY/resolve/REVISION/model.safetensors",
    "family": "qwen",
    "architecture": "qwen-image-2.1",
    "kind": "diffusion_models",
    "precision": "bf16"
  }
]
```

Supported declared architectures are `qwen-image-2.1`, `flux2-klein-9b-distilled`, `minimax-h3-ref2va`, and `seedvr2-7b`. The declaration is a compatibility contract supplied by you, not a weight-level proof. The file's checksum, safetensors structure and precision are checked separately. A graph must still pass real inference with that model.

Split models use `diffusion_models`, `text_encoders` or `vae`. `checkpoints` means a genuine all-in-one checkpoint supported by ComfyUI's checkpoint loader, not any safetensors file. The Qwen Photo, FLUX Photo and H3 model nodes expose compatible diffusion selections. Separate family loaders are available for advanced graph assembly. SeedVR2's integration has its own model loader/storage convention; custom SeedVR models need an explicit integration test before replacing that baseline.

The source FLUX checkpoint is distilled. A non-distilled, Turbo or otherwise differently wired architecture is a new model integration, not a valid dropdown replacement. No automatic model-family or precision substitution occurs. Standard HQ workflows stay fixed to their source baseline; customize a personal copy or a task workflow instead.

## Keep the text file instead

Place your own file at a private runtime path, for example `/workspace/private/lora_links.txt`, and set `WB_LORA_FILE` to that path. It is never included in the image or repository. A private runtime JSON file works the same way through `WB_CHECKPOINT_FILE`.

After changing an existing runtime file, use **Refresh private library** in the sidebar. Updating an environment variable itself is a Pod configuration change; it is not required for selecting already registered files in ComfyUI.

## Privacy boundaries

Private URLs are not written to model indexes, the public task catalog, generated RunPod settings or normal download logs. Private filenames and local metadata are kept in a restricted runtime index and appear locally where needed for model selection. Standard exports omit image workflow metadata by default. Hand-exported workflow/API JSON can reveal local private model names, prompts and references: sanitize it before sharing.

Runtime environment variables and files are not hidden from privileged container processes. Use only trusted node packages. Do not place actual secrets in this page, GitHub Actions inputs, Docker build arguments, issues or screenshots.

Deleting a library entry does not delete its cached weight file automatically. Manage private caches deliberately; the public Files tab exposes only input/output media, not credentials or private indexes.

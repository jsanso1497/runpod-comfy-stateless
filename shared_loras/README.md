# Shared LoRA library

For current installation and build instructions, use [START_HERE.md](../START_HERE.md).
This library is included in both images. Edit only `config/lora_links.txt` to add
user LoRA URLs; choose compatible adapters in each workflow. Existing required
Krea adapters in `config/loras.json` are separate and retained.

## Downloader details

# Shared LoRA link resolver

Only config/lora_links.txt is user-edited. sync.py reads provider metadata and
checks file hashes internally. It runs at Pod startup, not during Docker builds.
There are no provider keys or model weights embedded in either Docker image.

Supported: Civitai model/version/download links, Hugging Face model-repository
links with one safetensors file, and explicit blob/resolve safetensors file URLs.
HF repository links with multiple files require the chosen file's page URL.
Civitai version file-selection parameters, including precision, are honored.

Content must be safetensors with recognizable adapter keys. This structural check
is NOT proof of model compatibility. Only your workflow's chosen LoRAs are applied.

Metadata resolves SHA-256 for Civitai and HF LFS files. Git-stored small HF files
are checked against their Git blob SHA-1 and a local SHA-256 is recorded afterward.
A missing published checksum fails the file instead of disabling integrity checks.
The code records the resolved version/commit and checksum in link-library.json.
No manual pin is needed, but moving source links can resolve new files on later
Pod launches. Use a version-specific Civitai/file link when a precise variant matters.

Credentials go only to the original provider host, never its redirect CDN. Initial
links accept only official HTTPS provider domains. Redirects must be HTTPS and
resolve to public addresses. Network proxies from environment variables are not
used. Download exceptions do not print signed CDN URLs or API tokens.

Each link is independent; failures are recorded by source line and the other links
continue. Downloads resume with valid ranges and are published atomically only
after checksum/structure checks. Eight GiB of disk headroom is reserved. LoRAs
are not auto-converted in precision, activated, or trained by the downloader.

Provider API documentation used:

```text
https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md
https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.model_info
https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
```

# Security and private-data handling

No private LoRA TXT was read or included in this release. No API key values were supplied to the build. Known private input names, weight formats, `.env` files and runtime directories are ignored by Git and Docker. The Dockerfile also uses an explicit source allowlist instead of copying the entire build context.

The source's historical H3/FLUX Dockerfiles copied the private library TXT. Existing repository history or previously built images may therefore retain it. This release does not inspect or erase that history. Review old artifacts separately and revoke any actual credential that may have been published. A model URL is not necessarily a credential, but its contents can still be private.

Provider tokens come only from runtime secrets. The downloader rejects credentials in URLs, restricts provider types, validates HTTPS/public redirect destinations, does not send provider authorization to arbitrary redirect hosts, verifies provider hashes and source-pinned hashes, checks safetensors integrity, and downloads through an atomic partial file. DNS and HTTPS protections reduce risk but are not a complete sandbox for untrusted code.

The public port requires Basic authentication and must be used through HTTPS. ComfyUI and Ollama bind to loopback. Cross-origin writes and WebSocket handshakes are rejected. The unauthenticated health endpoint reports only gateway/workspace liveness, not assets, files or credentials. It is not a GPU-ready signal.

Model API tokens and private library values are removed from the ComfyUI child's environment. Processes sharing the container user/root still have broad access. Do not treat Unix file modes or environment scrubbing as isolation from malicious custom nodes. Install only trusted, reviewed source pins.

The configurator contains no field for secret values and no telemetry. A hosted copy exposes the task catalog and public model details. Publish it only after reviewing that information. Exported images omit normal workflow metadata by default; hand-exported graph/API files may contain private model names and prompt/reference information. Review them before sharing.

Report a vulnerability privately to the repository owner. Do not include real tokens, private links, patient/client data or personal reference images in a public issue.

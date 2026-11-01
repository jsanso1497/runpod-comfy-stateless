# FileBrowser automatic startup

Combined release 2026-11-01.1. Apply this package instead of the previous continuity ZIP,
or over that already-installed update. No H3 model, sampler, prompt, frame history,
queue runner, or seam-checking algorithms have been changed in this add-on.

## RunPod settings

Use the NEW digest from a successful **Build H3 character replacement** run.
Image tag: `ghcr.io/jsanso1497/runpod-comfy-stateless:h3-character-swap`.
Leave the container start command and entrypoint override blank. Expose HTTP
**8188 only**, with no extra TCP ports for FileBrowser. Keep persistent `/workspace`.

The recommended environment is in RUNPOD_ENVIRONMENT.txt. The only new variable is:

```
H3_SWAP_FILE_BROWSER=1
```

It defaults to 1 in this new image. Setting it to 0 disables the file service and
its gateway route. This flag has no effect in the older images.

For the optional BFS comparison, after reviewing its supplied model license:

```
H3_SWAP_VARIANTS=akatz,bfs
H3_SWAP_BFS_LICENSE_REVIEWED=1
```

Otherwise leave H3_SWAP_VARIANTS=akatz and omit the license-review flag.
No WB_WORKSPACE, WB_TASKS, WB_TOOLBOX, ENABLE_H3_MEDIA, Civitai key, or OpenAI key
is needed in this isolated image. Do not reintroduce the old oversized Ollama
context: the runtime still caps it at 16384, with one loaded helper at a time.

## Where to open it

Open RunPod's HTTPS HTTP-8188 service, then change its URL path to `/files/`.
The `/_swap` landing page also has a FileBrowser link. Use the EXISTING gateway
login, username **workbench**, password from **comfy_password**. There is no
separate admin login or FileBrowser password to set. The authenticated gateway
asserts the single user to the private service.

The same root service provides ComfyUI at `/`, chain review at `/_swap/chain`,
and startup diagnostics at `/_swap/status`.

FileBrowser serves `/workspace`. Its `input`, `output`, and `models` folders
remain the same folders used by H3. Existing manual-FileBrowser databases are
NOT imported, modified or deleted. The new private database is stored at
`/workspace/.h3-filebrowser/filebrowser.db`; settings persist with the volume.
Its internal state is denied by a global FileBrowser rule. Dotfiles are hidden.
Do not rename/delete active H3 inputs, model files, or chain checkpoints.

## Startup and failures

The binary is downloaded and checksum-verified during image build, not on every
Pod boot. The existing managed runtime launches it automatically at Pod startup.
It starts alongside ComfyUI and can serve files before model preparation finishes.
The service is monitored with at most three start/restart attempts per Pod run.
A FileBrowser error is reported without intentionally stopping working H3 jobs.

Status: `/_swap/status`, under `filebrowser`.
Log: `/workspace/h3-swap/logs/filebrowser.log`.
A failed real binary/config/auth check stops the image build before publication.

Only the authenticated gateway is public. FileBrowser listens at **127.0.0.1:8081**.
Do not expose 8081, 8189 or 11434. Remove the legacy **8080** HTTP port from the
new template. A stale RunPod 'File Browser' 8080 service card is not the new route.
This installation intentionally replaces the earlier direct-8080 instructions.
RunPod template edits alone do not replace an existing Pod's already-running image.
Deploy the newly built digest with the persistent volume and these settings.

## Security and limitations

This preserves the familiar classic FileBrowser UI, using its final upstream
**v2.63.23** release. The original project is archived and receives no new security
fixes. Its maintainers advise using a TLS-terminating authenticated reverse proxy
and keeping the command runner disabled. This integration uses RunPod HTTPS,
the existing Basic-auth gateway, a loopback-only backend, fixed injected identity,
no public signup, no sharing/admin/execute permissions for the file user, disabled
command execution, and no external symlink traversal. The child process gets a
minimal environment without gateway passwords, HF tokens or RunPod API keys.

It still runs under the container's existing user so it can manage root-created
workspace files. It is NOT an OS sandbox, a separate unprivileged container, or a
security boundary against custom ComfyUI code. A compromised Pod process remains
trusted on loopback. Authentication and these restrictions do not make archived
software maintained. For long-term multi-user or sensitive deployments, migrate
to a maintained file manager and separate privilege boundaries.

The 1024 MiB setting is an application-side per-request upload limit shared with
the gateway. External proxies and browser limits can be lower. No GPU/model is
loaded by FileBrowser, but ordinary uploads/previews can still use CPU/RAM/disk.

## Validation boundaries

New offline tests use real local HTTP for gateway routing, authentication, blocked
cross-origin writes, fixed identity, streaming uploads and Range downloads. CLI
processes are mocked in offline tests. The real pinned FileBrowser binary could
not be downloaded in the authoring container. The Dockerfile includes a separate
real-binary configuration/restart/proxy-login/listing/private-state smoke test.
That smoke, Docker build, live browser behavior and RunPod restart still require
verification in your GitHub build and Pod. No new GPU inference is needed for the
file service, and no H3 visual-quality claim follows from these tests.

## Sources

- Archive/security notice: https://github.com/filebrowser/filebrowser
- Pinned release and asset checksum: https://github.com/filebrowser/filebrowser/releases/tag/v2.63.23
- Proxy authentication contract: https://github.com/filebrowser/filebrowser/blob/v2.63.23/docs/authentication.md
- CLI config flags: https://github.com/filebrowser/filebrowser/blob/v2.63.23/docs/cli/filebrowser-config-init.md
- User flags: https://github.com/filebrowser/filebrowser/blob/v2.63.23/docs/cli/filebrowser-users-add.md
- Rule flags: https://github.com/filebrowser/filebrowser/blob/v2.63.23/docs/cli/filebrowser-rules-add.md
- RunPod secret syntax: https://docs.runpod.io/pods/templates/secrets
- RunPod ports/HTTPS: https://docs.runpod.io/pods/configuration/expose-ports

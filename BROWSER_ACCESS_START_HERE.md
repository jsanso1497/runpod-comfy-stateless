# RunPod browser access fix 1.2.0

## What this update is

An incremental update for `jsanso1497/runpod-comfy-stateless`, based on commit
`5b480a648e8efb7dbb5f7482fdec6be1e26f6b9b`.

It adds **explicit password-free access on the normal RunPod HTTP 8188 link**:

```text
WB_AUTH_MODE=none
```

No SSH keys, Windows PowerShell, Cloudflare tunnel, or password-entry tests are
needed. ComfyUI and the Workbench sidebar remain behind the same gateway. ComfyUI
itself remains on the internal loopback port 8189.

**Security:** `none` removes Workbench authentication. Anyone who discovers the
public Pod URL can access the images/workflows exposed by the application and run
jobs on your GPU. Do not use this mode for confidential material. Cross-origin
browser-write protection remains, but it is not access control. Keep the link
private and stop the Pod when you are done. Normal password protection remains the
default unless you explicitly select `none`.

This does not remove or replace your separate Comfy account sign-in for the GPT
Partner node. No OpenAI API key is added or needed by this update.

## 1. Update the existing GitHub repository

1. Extract `RunPod_Browser_Access_Fix_v1.2.0.zip` on your computer.
2. Open the existing repository and select `main`.
3. Choose **Add file > Upload files**. Drag the EXTRACTED CONTENTS into the upload
   area. Preserve their paths. Do not upload the ZIP itself or an enclosing folder.
4. Commit the files to `main`.

**Do not delete anything first. Do not replace entire repository folders.** This
package replaces only `Dockerfile` and `src/workbench/runtime.py`; everything else
it contains is a new transport module, tests, instructions, or a validation report.
It contains no replacement model catalog, model weights, workflow JSON, secrets,
or hidden `.github` files. If your Git client shows file deletions, cancel and
correct the merge before committing.

Do not apply the old recovery workflow or older Green Suit ZIPs again.

## 2. Validate and build Qwen

In GitHub **Actions**:

1. Run **Validate repository** against the new `main` commit. The existing checks
   remain enabled and automatically discover the new gateway tests.
2. After validation succeeds, run **Build HQ workspace**. Select `qwen`, not `all`.
3. After the build succeeds, copy the complete `image_reference` from the build
   summary or `image-qwen` artifact. It starts with:

   ```text
   ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:
   ```

Use the actual newly published digest, not an old digest. The build also publishes
`ghcr.io/jsanso1497/runpod-comfy-stateless:qwen-hq`, but the digest is unambiguous.
Repository validation by itself does not publish an image.

**Skip Publish configurator to Pages.** The page is not part of this repair, and
its current generator does not include the Green Suit add-on. Set the access
variable directly in RunPod as described below. No FLUX, H3, or restoration-only
image rebuild is required for this Qwen Pod.

## 3. Update the RunPod template and launch the new image

Use the new image reference from step 2. Keep HTTP port **8188** exposed, TCP ports
blank unless you need them separately, and the container start command blank.
Do not expose 8189.

Add this environment row:

| Key | Value |
| --- | --- |
| `WB_AUTH_MODE` | `none` |

Remove the `WB_PASSWORD` environment row for this password-free setup. Keep your
existing `HF_TOKEN`, `CIVITAI_TOKEN`, `WB_WORKSPACE`, `WB_TASKS`, `WB_TOOLBOX`, and
optional private-library rows. Do not remove the Hugging Face/Civitai secrets.
There is no need to create another login secret or an OpenAI API key.

For a fresh Qwen + Green Suit template, these rows are sufficient:

```text
WB_WORKSPACE=qwen
WB_AUTH_MODE=none
WB_TASKS=SQ,Q01,QGS1
WB_TOOLBOX=sam,seedvr2
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
```

An existing longer `WB_TASKS` list can be retained. Selecting only these three
startup tasks does not hide other Qwen workflows: the unchanged runtime still
copies all workflows compatible with the Qwen workspace. The required Qwen,
SAM, and SeedVR2 assets are prepared. Prompt-enhancer preparation is unchanged.

Launch a Pod using the updated template/image. Updating a template does not patch
an already running container. Changing/resetting an existing Pod can erase its
container disk, so export unique files before doing that.

The new startup logs include:

```text
[WORKBENCH ACCESS 1.2.0] NO LOGIN on port 8188. Anyone with the URL can access this Pod.
```

## 4. Open ComfyUI

In the NEW running Pod, use **Connect > HTTP 8188**. Use the current link shown by
RunPod, not an old Pod URL and not a `trycloudflare.com` link.

The Workbench username/password popup is not required in `WB_AUTH_MODE=none`.
Any old Basic Authorization header cached by the browser is ignored for Workbench
access in this mode and is stripped before forwarding to ComfyUI.

Sign in to your normal Comfy account inside ComfyUI for the GPT Partner node.
Open Green Suit from Workbench, or from the regular workflow browser:

```text
Workbench Factory / 0.1.0 / QGS1 /
Green_Suit_Qwen_to_GPT25_Overlay_v1_0.json
```

The model downloads still need to complete on a fresh empty disk. The update does
not falsely treat a running gateway as a completed model download or GPU test.
The public `/healthz` response now includes `access_version: 1.2.0` and
`auth_mode: none`. `/readyz` distinguishes ComfyUI readiness from gateway liveness.
These are status endpoints, not password troubleshooting steps.

## Storage and stopping for the night

Your posted `df -h /workspace` result showed `overlay` mounted on `/`, not a
persistent volume. Naming a directory `/workspace` does not make it persistent.
Stopping or replacing that container can lose the model cache and any files you
uploaded or generated there. Save irreplaceable inputs, outputs, or personal
workflows before stopping the Pod to stop GPU billing. Do not terminate it before
saving anything you need. A running Pod continues to incur compute charges.

For the NEXT Pod, use a **200 GB network volume mounted at `/workspace`**, selected
when you create the Pod, in a datacenter with your chosen GPU available. A starting
container-disk allowance of 80 GB matches the repository's persistent-storage
configuration. Keep your working A100 80 GB selection rather than changing GPUs
to solve this access issue. Increase storage if your own model library needs it.

A new empty network volume still needs the first model download. Reuse that SAME
volume for later Pods so models and results persist independently of Pod deletion.
A network volume cannot simply be added over the current container directory to
preserve its existing contents. Existing files must be copied or downloaded first.
Network-volume storage remains billable separately from GPU compute.

## What changed technically

- `WB_AUTH_MODE=none`: no Workbench password or browser Basic Auth challenge.
  `basic` remains the default for other deployments. Typos fail closed.
- A dedicated proxy module handles normal HTTP and WebSockets. Bodyless GET/HEAD
  requests are no longer sent upstream with an empty chunked request body.
- Upstream compression negotiation is normalized to `identity`; if an upstream
  nevertheless sends encoded bytes, its encoding and length are preserved.
- Response lengths, content types, byte ranges, and repeated cookie headers are
  retained. Connection-nominated hop-by-hop headers are removed.
- Security headers are installed before responses are sent, not after a stream
  has already been prepared. A partial stream is not followed by a second HTTP
  status block on failure.
- WebSocket handshakes are rebuilt correctly and text/binary traffic is relayed.
- Browser same-origin write protection stays enabled in both access modes. The
  expected public origin can be derived from the real RunPod Pod ID.
- Shared upstream client cookies and implicit environment-proxy authentication
  are disabled. Native non-Basic authorization is not discarded.
- Docker starts Tini with `-s` and checks the PyAV color imports and gateway
  factory during the build, without starting inference or downloading models.

The previous gateway already removed `Transfer-Encoding`. The earlier suggestion
that it did not was inaccurate. The supplied logs did not establish one conclusive
cause of the Chrome error. This patch removes the login requirement you do not
want and fixes several specific transport behaviors; it is not a claim that the
live Cloudflare/Chrome fault was reproduced here.

## Verification and limitations

42 new regression tests passed using real local HTTP and WebSocket servers, with
model preparation and ComfyUI responses simulated. They cover no-login and legacy
password modes, HTML/JavaScript, compression, JSON, multipart upload, byte ranges,
redirects, cookies, origin guards, WebSocket text/binary/close, and failures.

The browser smoke-test script is included, but this environment's Chromium policy
blocked navigation to the test loopback server with `ERR_BLOCKED_BY_ADMINISTRATOR`.
No successful browser test is claimed. This is a limitation of this test environment,
not a diagnosis of your browser.

No Docker image was built or published here, and no live RunPod, real ComfyUI
frontend, GPU generation, or paid GPT request was executed. The existing full
repository test suite must still run in GitHub Actions after upload.

The patch leaves all model/download/asset-catalog code, Green Suit workflow JSON,
SAM controls, Qwen/GPT prompts, model precision, and Comfy account integration
unchanged. All unrelated functions in the runtime were compared structurally to
the fetched source and retained.

## Restore password protection later

Set `WB_AUTH_MODE=basic` and restore:

```text
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
```

Use a NEW secret value of at least 16 characters, because a previous password was
exposed in your terminal transcript. The old value is not included in this patch.
Deploy/apply settings only after preserving any temporary-disk data.

## Reference material

- Repository baseline: `5b480a648e8efb7dbb5f7482fdec6be1e26f6b9b`.
- RunPod storage: https://docs.runpod.io/pods/storage/types
- RunPod storage lifecycle: https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage
- aiohttp server response lifecycle: https://docs.aiohttp.org/en/stable/web_reference.html
- aiohttp client response handling: https://docs.aiohttp.org/en/stable/client_reference.html

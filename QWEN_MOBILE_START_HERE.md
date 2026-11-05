# Qwen Mobile App + automatic JupyterLab 1.0.0

Incremental update for `jsanso1497/runpod-comfy-stateless`, inspected at
`05e957f3381e3d0e8ffa8fc1b7d7a9cd96a429fb` (November 4, 2026).

## What this adds

- **QAPP: Qwen Mobile: Native BF16 Photo Edit.** An editable workflow that opens
  in native ComfyUI App Mode. Upload one photo, enter the edit instruction,
  optionally change resolution/seed, and Run. No external mobile app or cloud API.
- **Automatic, token-protected JupyterLab on port 8888.** File browser, editor,
  uploads/downloads, notebooks, and browser terminals rooted at `/workspace`.
  Installed at image-build time in `/opt/jupyter-venv`, separate from ComfyUI.
  Starts again with each Qwen HQ Pod boot; no terminal installation/start command.
- Existing Qwen/SAM workflows, model catalog assets, authentication gateway, and
  other model-family runtime behavior are retained. No existing workflow JSON is
  overwritten. SAM remains available in its existing workflows; the simple app
  deliberately has no mask, reference-stack, LoRA or restoration controls.

This is a source update, not an already published Docker image. Docker build,
RunPod-proxy, real mobile-browser, and GPU inference tests remain to be performed.

## Apply locally

Download `apply_qwen_mobile_jupyter_v1.py` into your Downloads folder. From a
terminal opened inside your existing repository, run:

```bash
python3 "$HOME/Downloads/apply_qwen_mobile_jupyter_v1.py"
```

The standard-library-only updater checks every affected file before writing,
refuses conflicting changes, backs up replaced files under your Git metadata
folder, and is safe to rerun. It does not contact GitHub, install dependencies,
commit, reset, delete repository files, or push. `--check` performs a dry run.
The optional ZIP contains complete replacement/additional source files for review.

Review the changes, commit and **Push origin manually in GitHub Desktop**.
In GitHub Actions, run **Build HQ workspace** with workspace **qwen**. The existing
validation job remains in place. Use the new successful build's `image_reference`
from its summary or `image-qwen` artifact when deploying RunPod. The rolling tag is
`ghcr.io/jsanso1497/runpod-comfy-stateless:qwen-hq`.

## RunPod settings

Keep your existing A100 80 GB / persistent-storage settings. The previous
planning setup was 80 GB container disk and a 200 GB network volume at `/workspace`.
This update does not add any image-model weights. Expose HTTP **8188,8888**;
leave TCP ports and the container start-command override blank. Do not expose
internal ComfyUI port 8189 or Ollama port 11434.

```text
WB_WORKSPACE=qwen
WB_QWEN_ONLY=1
WB_APPROVED_ONLY=0
WB_TASKS=SQ,Q01,QNATIVE,QOVERLAY,QAPP
WB_TOOLBOX=sam,seedvr2
WB_JUPYTER=1
WB_AUTH_MODE=basic
WB_PASSWORD={{ RUNPOD_SECRET_comfy_password }}
JUPYTER_TOKEN={{ RUNPOD_SECRET_comfy_password }}
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civit_token }}
```

Enter these as RunPod environment rows, not terminal commands. Keep your optional
private-library variables unchanged. `comfy_password` must resolve to at least
16 characters. A separate Jupyter secret is also supported: put its RunPod secret
reference in `JUPYTER_TOKEN`. Never paste real tokens into GitHub source files.

The Qwen-only switch prevents enhancer-model preparation. The app itself contains
no enhancer branch even when that switch is off. The required Qwen image encoder
is not a prompt enhancer and remains required. SAM and SeedVR2 are separate tools.

Launch the newly built image. A template update does not patch a running Pod.
Preserve inputs, outputs and saved workflows before replacing a Pod or any
operation that resets its container disk. Reuse your mounted network volume.

## Use on your phone

Open RunPod **Connect > HTTP 8188**. The Workbench login is username `workbench`
and the actual password stored in your secret. In the Workbench task browser,
open **QAPP / Qwen Mobile: Native BF16 Photo Edit**. It is also copied into the
versioned Workbench Factory workflow folder at startup.

The workflow is preset to App Mode. On a narrow screen, use the native input and
output tabs. Upload your image before running; the factory workflow intentionally
has no personal/sample image selected. Enter a short edit instruction and Run.
The Save Image output is displayed and saved under `/workspace/output/Qwen_Mobile`.
Use the output's native image/download controls to save it to your device.

App controls: source image, instruction, resolution, seed, and seed-change mode.
Sampler, model loaders, encoder and VAE are hidden from the app but remain editable
in the node graph. Native defaults: 25 steps, CFG 1, Euler, simple scheduler,
denoise 1, auto-device cache with default/lossless precision.

Resolution defaults to **2048**, the native approximately 2048 x 2048 pixel-budget
control, preserving the source aspect ratio and rounding dimensions to multiples
of 32. It is not an exact longest-edge setting and is not unlimited source size.
Use 1536 (approximately 2.4 MP) or 1024 (approximately 1 MP) for smaller tests.
Avoid 0 on large phone photos because it requests almost the original dimensions.
Peak GPU memory is not certified. Native generation does not guarantee pixel-exact
alignment; use your existing masked crop/stitch workflows for protected pixels.

`randomize` produces variations. Use `fixed` and an unchanged seed for prompt
comparisons. No automatic prompt rewriting is performed. No negative prompt is
exposed because the native CFG-1 path does not use it.

## Use JupyterLab

Open RunPod **Connect > HTTP 8888**, or its normal Jupyter HTTP service link.
In the password/token box enter the **actual value** of `comfy_password`, not
`workbench`, the secret name, or the `{{ ... }}` reference. Jupyter has no separate
username. Authentication is independent from the Workbench login on 8188.

Jupyter starts automatically for Qwen HQ. `WB_JUPYTER=0` disables it. If no explicit
Jupyter token is set, it can use a valid `WB_PASSWORD`; without either valid value
it does not start. `WB_AUTH_MODE=none` does not remove Jupyter authentication.
The approved-only image and other model-family images do not start this sidecar.

Jupyter keeps CSRF and same-origin checks, uses secure cookies for the RunPod
HTTPS proxy, does not allow unauthenticated API requests, and does not silently
choose another port if 8888 is busy. Model credentials are removed from the
Jupyter child environment. The login token is removed before kernels/terminals
inherit it. This is not a sandbox against a user with terminal access to the Pod.

A Jupyter failure is reported without stopping ComfyUI. It starts on Pod boot,
but a crashed Jupyter process is not repeatedly auto-restarted during that boot.
Inspect `/workspace/workbench/jupyter/server.log` and `status.json` when needed.
The status file distinguishes process liveness from authenticated HTTP readiness.
Do not start your old manual 8888 command on this new image.

Notebooks use Jupyter's separate Python environment. A browser terminal retains
the normal container PATH for ComfyUI maintenance. Models and personal data are
not installed into the Jupyter environment.

## Validation and limitations

37 focused CPU tests cover workflow topology, app widget references, model names,
native latent sizing and sampling, additive catalog registration, fail-closed
Jupyter authentication, environment separation, private logs, process cleanup,
and startup behavior. An actual local Jupyter service passed API authentication,
file write/read/delete, terminal creation and WebSocket-command smoke checks.

The local smoke used the environment's JupyterLab 4.5.3 / Jupyter Server 2.17.0.
The image pins JupyterLab 4.5.11 / Jupyter Server 2.21.1, verified published on PyPI.
Those exact pins still need the Docker build and deployment test. No GPU model
execution, live RunPod access, full-repository test run, or iPhone-browser render
is claimed. The existing GitHub validation/build gates remain necessary.

The native workflow and App Mode data were checked against your existing core
commit `bab80ef793ac42ad3f25589a6b0507ca537b1dd1` and frontend 1.55.17.
Neither is upgraded. A changed pin fails the new build check for review.

## Primary references

- https://docs.comfy.org/interface/app-mode
- https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1
- https://github.com/Comfy-Org/ComfyUI/blob/bab80ef793ac42ad3f25589a6b0507ca537b1dd1/comfy_extras/nodes_qwen.py
- https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.55.17/src/stores/appModeStore.ts
- https://jupyter-server.readthedocs.io/en/stable/operators/security.html
- https://jupyter-server.readthedocs.io/en/stable/other/full-config.html
- https://pypi.org/project/jupyterlab/4.5.11/
- https://pypi.org/project/jupyter-server/2.21.1/

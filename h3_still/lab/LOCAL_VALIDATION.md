# H3 Studio Lab: validation performed for this package

Release 2026-11-08.1. Prepared against the user's repository commit
`c8ad356ff6f7d2ca327c0395b21ec608062b900c`.

## Executed locally

- 77 unit tests passed, with no skipped tests in the local run. They cover all 13 graph structures and reciprocal connections, stable graph generation, default experiments OFF, independent reference crop/cap geometry using small Torch tensors, literal prompt role mapping, empty inputs, sampler/decode wrappers with fixtures, experiment guards, reviewed Spectrum/Spark argument mapping, RefMod tag numbering, pinned download metadata, checksum reuse, archive traversal/symlink rejection, and refusal to overwrite conflicting weights.
- 18 installer tests passed. They cover payload integrity and tampering, Python 3.8 syntax parsing, repository root/subdirectory detection, an actionable home-directory error, explicit --repo, dry run, additive writes, idempotence, full conflict preflight, destination symlink rejection, preservation of existing files, no Git staging, and rollback including a partially written new file.
- All package Python files compile in the local interpreter. The installer uses Python 3.8-compatible syntax and standard-library APIs; a Python 3.8 runtime was not available for an actual execution test.
- The copied known-good replacement workflow is byte-for-byte unchanged from the earlier successful diagnostic JSON provided in this conversation.
- The new GitHub Action parses as YAML. Its intended order is build, test, then publish.
- The deliverable ZIP is checked for file integrity and excludes __pycache__ and .pyc files.

Unit-test mocks are interface tests, not evidence of successful provider model execution. Small synthetic tensors are not H3 model weights.

## Checks built into the GitHub Action, not yet executed here

- Build against the previously successful Studio image, resolved to its actual immutable digest.
- Real ComfyUI CPU startup and object_info validation for the 13 workflows.
- Actual non-model preprocessing/contact-sheet graph execution through ComfyUI's HTTP API.
- Native reference-conditioning calls with synthetic encoder/VAE fixtures, baseline reference-tensor equality, and preservation of unmasked original pixels.
- Actual pinned Spectrum and RefMod imports and Python call-signature checks.
- Existing HTTPS gateway, browser file upload/download, full Chromium JupyterLab initialization, and terminal WebSocket test.
- In the optional ph3 build profile, availability of native P-H3 conditioning and single-frame latent geometry.

The Action does not publish if its required checks fail. No new Docker image was built or published in the preparation environment.

## Still requires the user's Pod

- Actual H3/P-H3 generation and visual equivalence to the successful baseline.
- Facial identity, body proportions, SAM selection, and compositing quality.
- Maximum resolution/reference counts or peak memory on A100/A40.
- Actual Spectrum speed/quality and fallback behavior.
- Spark release-archive extraction, GPU kernel probes and speed on SM86. The published checksum and interface were reviewed, but the binary archive could not be downloaded into the preparation container.
- Character Swap V1.5 still-image benefit and complete adapter key compatibility.
- RefMod reconstruction fidelity and whether reuse improves end-to-end runtime.
- Live RunPod proxy behavior after deployment.

No new model weights are included. No BF16 base, prompt-enhancer model, credentials, or remote repository changes are part of this package.

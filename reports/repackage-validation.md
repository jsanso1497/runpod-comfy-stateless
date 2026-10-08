# 0.1.1 same-repository distribution: validation

This distribution updates the default repository and installation process, not the model stack or generation workflows.

## Rerun checks

| Check | Result |
|---|---|
| Python tests | 69 passed; two non-failing aiohttp warnings |
| Configurator logic | 111 assertions passed |
| Static graph/catalog validation | 57 tasks, 69 editable graph files, 82 source dispositions; no errors |
| Desktop/mobile configurator | Passed; no JavaScript errors or external network requests |
| Existing repository default | jsanso1497/runpod-comfy-stateless |
| New-repository CLI publishing scripts | Removed; CI dependency removed |
| Six required hidden files | Included; presence covered by validation/tests |
| Editable workflows, API exports, node/runtime source, config, deploy code | Byte-identical to 0.1.0 |

The package adds same-repository Mac/Desktop and browser-only instructions. The separate browser-upload kit contains visible file batches and visible copies of hidden files. Its packaging manifest verifies that these reconstruct the same source tree.

No GitHub repository was modified, no Docker image was built or published, and no GPU inference was run. Existing model/download/runtime validation limits from 0.1.0 still apply. Browser tests exercised local HTML, not a signed-in GitHub upload or a native Mac Finder session.

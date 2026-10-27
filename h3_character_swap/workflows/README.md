# Workflows

00_Analyze_Clip_Build_Prompts and 03_BFS_V11_Guide_Comparison are included here
as editable canvases and paired API exports.

The image build installs the pinned developer-native Akatz canvas and API graph
from akatz-ai/h3-relay at e2a7665b6eb8ea37a88c0de2c9c6b69a8f7c9220.
It generates 01_Akatz_V15_Mask_Check and
02_Akatz_V15_Native_Replacement automatically after validating the upstream
recipe. It changes only the adapter's local filename, initial clip duration,
neutral source filenames, editable reference-role prompt and output prefix.
The canonical model, masks, loops, sampling and continuation path are retained.

This build-time retrieval deliberately avoids hand-reconstructing the complex
native loop graph from partial snippets. Both final JSONs are included in the
GitHub Actions artifact and copied to the Pod's ComfyUI Workflows sidebar.
The build stops rather than guessing if the pinned graph's widget schema differs.

Do not drag these files into an old h3-hq image. Use the new h3-character-swap
image, which contains the required pinned runtime and helpers.


## 2026-10-26.2 additions

- 05_Face_And_Body_Style_Sheet: two-photo layout utility, no synthesis.
- 06_H3_LowMemory_Chain_Control: launch/review/resume independent short jobs.
- 07_Akatz_LowMemory_Manual_Chunk.api.json: manual live-Pod fallback.
- 08_H3_Prepare_5s_Clips: CPU-only source splitter inside this image.

06 requires the new checkpoint/control nodes and a rebuilt image. Its controller
returns immediately, then the Pod-local runner queues the actual short jobs.
The original model/adapter recipe and its baseline workflows remain unchanged.
See ../LOW_MEMORY_CHAIN.md. A successful gate is not a seamless-motion guarantee.

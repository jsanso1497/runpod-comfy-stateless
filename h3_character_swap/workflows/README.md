# Current H3 workflow choices (2026-11-02.1)

The same workflow set is built for baseline INT8/NVFP4 and original BF16 images.
The profile is compiled into the image; output pixel budgets are separately editable.

| Workflow | Purpose |
| --- | --- |
| 02B Akatz Native Simple Preview | Native single-reference preview, 768 preset |
| 02D Akatz Native Simple Preview 480 | Explicit lower-resolution native preview |
| 10 Akatz Simple Single Reference Chain | Existing automatic continuation chain |
| 11 H3 Prepare Independent Chunks | Split original footage with optional editing overlap |
| 12 Akatz Independent Chunk Preview | One short native generation without prior generated history |
| 12B Akatz Independent Chunk Preview 480 | Same independent path at the lower pixel budget |
| 13 H3 Join Rendered Chunks | Trim duplicate overlap, concatenate chosen videos, restore original audio |
| 14 H3 Recover Past Generation Workflow | Recover sidecars, embedded metadata, chain config or matching live history |

The BF16-only 02C higher-reference-size comparison remains available. Mask checks,
face/body and analysis workflows are retained. Use 02B or 12 with the simple prompt
first. Read ../INDEPENDENT_CHUNKS.md for instructions and limits.

These are build-generated canvases. Both images need rebuilding for new node code.

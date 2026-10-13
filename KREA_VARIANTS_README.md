# Krea 2 variants

K01 now offers three separate workflows. Optional tools do not clutter the basic graphs.

| Variant | What Krea receives | Use |
|---|---|---|
| Identity Edit HQ | Source, second reference image/sheet, required Identity Edit adapter | Full-frame source editing and reference transfer |
| Standard, no Identity Edit LoRA | Text only, with up to two optional compatible LoRAs | Prompt-guided generation, not source-preserving editing |
| Masked Photo HQ | Working source crop, reference image/sheet, Identity Edit adapter | Manual/SAM selection, preview first, protected crop/stitch |

All preserve Raw and Turbo BF16 options, the Krea-specific 4B text encoder, two optional LoRA slots and four optional reference slots. Enabled references are labeled R1 onward in order. Two or more references use a visible sheet in the second native image input, not independent N-image conditioning.

The default helper is now the same selected full-precision vision model as Qwen. Existing `qwen3.8:27b-bf16` workflow settings migrate to the workspace selection. New graphs use `auto`. Source and reference image loading normalizes embedded color profiles to sRGB, and output PNGs carry an sRGB profile.

See `docs/krea-qwen-parity.md` for controls, migration, optional tools, and validation boundaries. Rebuild the `krea` HQ image; changing repository files does not update a running Pod or replace personal saved workflows.

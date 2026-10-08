# Migration from the supplied ZIP

No source files were overwritten and the private TXT was never opened. The complete machine-readable map is [catalog/migration.json](../catalog/migration.json).

Removed: ten KREA graphs; Q29-Q33; D02-D03; Qwen INT8 and lean head variants. U06/U07 are generic replacement utilities rather than copied torso code. H09 is a Full/BF16 port of the still-image method, not removal of that capability. Four Standard HQ graphs and U08/U09 are newly assembled.

API exports migrated from available source exports: 42. The authoritative editable catalog has 69 graph files and 57 task entries. API exports are not additional tasks.

This is a full replacement of the working files in the **same existing repository**, not a cumulative overlay. Keep the repository URL and Git history. New HQ tags distinguish these images from legacy builds; prefer the successful build's immutable digest. Replacing source files does not update running Pods. Never point a new workspace at an old Lite configuration without explicitly reconciling its persistent config.

Legacy API graphs are adjusted to the new node mappings; they still require live schema validation and inference checks. To generate an API payload for a graph without a source API export, use ComfyUI native API export after loading it in the running image.

See [Mac installation without Terminal](mac-install.md). Private data removed from the latest source can still exist in older Git history or older images; this package does not rewrite history or delete existing registry images.

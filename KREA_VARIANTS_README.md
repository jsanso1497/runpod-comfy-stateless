# Two Krea variants

- **Identity Edit**: full Krea2Edit source/reference conditioning, required Identity Edit v1.2 adapter, two additional optional LoRAs. Use for likeness-preserving image edits.
- **Standard, no Identity Edit LoRA**: native Krea 2 text-to-image conditioning without the edit adapter. Qwen sees source and references for prompt enhancement, but Krea itself does not receive them. This is **not** a source-preserving reference edit.

Both offer Raw/Turbo checkpoint selection and Qwen 3.8 27B prompt enhancement. The standard graph deliberately excludes the adapter rather than bypassing it inside the specialized editor. GPU testing pending.

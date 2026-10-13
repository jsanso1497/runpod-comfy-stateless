# Troubleshooting

**Image not found / denied.** Complete the matching build first. Copy the published digest from the action summary. Private GHCR packages need a RunPod registry credential; the HF and Civitai tokens are not registry credentials.

**Gateway refuses to start.** Check that `WB_PASSWORD` resolves to at least 16 characters and `WB_WORKSPACE` matches the actual image. Placeholders left literally as `{{ ... }}` are errors, not passwords.

**Provider HTTP 401/403.** Confirm the exact RunPod secret mapping. Some source model repositories require accepting access conditions using the same account as the token. Do not paste tokens into URLs or logs.

**ComfyUI opens but a workflow says missing nodes.** This is distinct from missing model weights. Inspect the startup import error, then run `scripts/validate_live.py --workspace MODEL` in the source folder with the image's Python environment. Rebuild with compatible pinned dependencies; do not install arbitrary node updates blindly.

**A workflow says missing model.** Use Prepare assets for that task. SAM for Q01 is optional: its automatic mode also needs the separate SAM feature. Manual mode does not. Private files must declare a compatible family and architecture where required.

**Model does not appear after preparation.** Use Refresh model choices, or save your graph and reload ComfyUI. Existing graphs retain their previous widget values until deliberately changed.

**A model exceeds memory.** There is no fallback to INT8/Lite. The listed GPU/RAM values are planning estimates. Increase available memory or explicitly reduce a task's working crop/reference workload only after reviewing the quality tradeoff. H3 core files exceed 120 GB before the 134 GB local helper pair and caches. GPU VRAM alone is not the complete capacity calculation.

**Build fails under Python dependency constraints.** The build deliberately prevents a node installer from silently replacing the base Torch stack. Resolve that dependency conflict explicitly. This source-only release has not built the four images here, so a clean build is a required gate.

**Mask selects the wrong shirt.** U08 wires SAM only to original A. B is connected only to the final compositor. To include more surrounding pixels, increase expansion; to extend the fade farther out, increase feather. The two controls are separate. Both images must already have identical geometry.

**Where are my files?** The Workbench Files tab lists input/output media through the authenticated gateway. Saved RefMods are under the persistent model `refmods` folder and exported from their dedicated task. Stateless storage is not an archive.

**Can I add a model from another architecture?** Not by renaming a file or changing an encoder dropdown. Add a new explicit integration, workflow baseline, manifest and tests.

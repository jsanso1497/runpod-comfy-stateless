# RunPod ComfyUI: H3 Portrait 1.4 Full and Lite

Read **START_HERE.md** for the browser-only update and exact RunPod settings.
Do not wipe the repository again. Preserve config/lora_links.txt and your GHCR images.

- Full: `:h3-portrait-clean`, 32B Instruct analysis plus 32B Thinking direction, BF16 H3.
- Lite: `:h3-portrait-lite`, 8B Instruct helper, full-architecture INT8 H3, smaller canvas.
- Both: explicit reference roles, pose/camera text-only routing, native manual Ref2VA,
  exact last-decoded-frame PNG, private loopback Ollama and passworded Jupyter on 8888.
- General `:latest`: retained existing Krea, SeedVR2 and RefMod source/workflows, separate build.

The UPDATE ZIP excludes the private LoRA list. The complete source ZIP has a comments-only
placeholder, not your URLs. Do not overwrite your list from that complete copy.

See INVESTIGATION.md for the reference competition analysis and VALIDATION.md for executed
checks and limitations. Local tests are not a successful GPU inference or a built image.

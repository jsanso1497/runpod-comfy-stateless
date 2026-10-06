# H3 Portrait 1.5.3 update

Read `H3_1.5.3_START_HERE.md` for the new protected workflows, startup gate and GPU handoff.
The general-template documentation below is retained for its separate build.

# Additions in 1.2.0

See [the Krea two-reference / H3 MP4 setup guide](RELEASE_1_2_GUIDE.md) for the two new workflows and the opt-in `ENABLE_H3_MEDIA` flag. Existing seven Krea graphs and H3 graphs remain unchanged.

# Krea Identity 1.1 additions

Use `krea_identity/MULTI_REFERENCE_GUIDE.md` for labeled additional references,
text-directed target selection, man/woman single-pass replacement, and protected
separate-identity editing. Workflows 01-04 and all H3 inference graphs are retained.
Rebuild the General/Image image from the new commit; the RunPod settings and model
weights remain the same. The update is a cumulative overlay, not a folder replacement.

---

# RunPod ComfyUI: H3 Portrait 1.5 Full + Lite

See `START_HERE.md` for browser-only update and RunPod instructions.

- **Full**: `:h3-portrait-clean`, BF16 MiniMax H3, 32B Instruct reference analysis + 32B Thinking director.
- **Lite**: `:h3-portrait-lite`, full-architecture INT8 MiniMax H3 and one 8B Instruct Ollama helper.
- **Lite single-image workflow**: Ollama-directed **MiniMax H3 Ref2VA** still generation from multiple subject refs plus optional pose/scene guides. No separate image-generation model is downloaded.
- **Both portrait images**: explicit reference roles, corrected widget serialization, manual native Ref2VA, exact exported-video final-frame PNG, passworded JupyterLab on 8888.
- **General image**: `:latest` remains independent for the preserved general workflows.

The update ZIP intentionally excludes `config/lora_links.txt`.

## Optional Krea Identity still-image module

See [krea_identity/START_HERE.md](krea_identity/START_HERE.md) for the three-upload
Turbo BF16 editor, exact screenshot rebalancer, optional head refinement and
single-image upscaling. No H3 pipeline or private LoRA-list replacement.

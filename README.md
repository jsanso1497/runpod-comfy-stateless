# RunPod ComfyUI: H3 Portrait 1.5 Full + Lite

See `START_HERE.md` for browser-only update and RunPod instructions.

- **Full**: `:h3-portrait-clean`, BF16 MiniMax H3, 32B Instruct reference analysis + 32B Thinking director.
- **Lite**: `:h3-portrait-lite`, full-architecture INT8 MiniMax H3 and one 8B Instruct Ollama helper.
- **Lite single-image workflow**: Ollama-directed **MiniMax H3 Ref2VA** still generation from multiple subject refs plus optional pose/scene guides. No separate image-generation model is downloaded.
- **Both portrait images**: explicit reference roles, corrected widget serialization, manual native Ref2VA, exact exported-video final-frame PNG, passworded JupyterLab on 8888.
- **General image**: `:latest` remains independent for the preserved general workflows.

The update ZIP intentionally excludes `config/lora_links.txt`.

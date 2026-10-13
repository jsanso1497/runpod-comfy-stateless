# Environment variables

## Set once in a RunPod template

| Variable | Default/meaning | Change frequency |
|---|---|---|
| `WB_WORKSPACE` | `qwen`, `flux`, `h3`, or `restoration`; must match the image | Selected automatically by the configurator |
| `WB_TASKS` | Comma-separated task IDs to prepare at startup | Optional; normal switching happens in the sidebar |
| `WB_TOOLBOX` | Optional compatible `sam`, `seedvr2`, `ollama` asset groups | Optional; also preparable from the sidebar |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` | Reuse your existing secret |
| `CIVITAI_TOKEN` | `{{ RUNPOD_SECRET_civit_token }}` | Reuse your existing secret |
| `WB_PASSWORD` | `{{ RUNPOD_SECRET_comfy_password }}` | Create once; 16+ characters |

Lowercase environment aliases `hf_token` and `civit_token` are also recognized. Secret names are not key values; neither belongs in the repository as a value.

## Private libraries, optional

| Variable | Meaning |
|---|---|
| `WB_LORA_URLS` | A multiline URL list, `family | URL` list, or JSON array |
| `WB_CHECKPOINTS` | A JSON array declaring exact family, architecture and component kind |
| `WB_LORA_FILE` | Runtime file fallback for a large or manually managed private LoRA list |
| `WB_CHECKPOINT_FILE` | Runtime file fallback for a private checkpoint manifest |

An explicit environment value takes precedence over its file fallback. They are not silently merged. Empty/unset variables do not cause the repository's old private TXT to be read. Very large libraries should use a private runtime file because environment-size limits vary.

## Advanced settings

`WB_DATA_ROOT` defaults to `/workspace`. `WB_SAVE_METADATA=0` disables normal ComfyUI generation metadata and suppresses workflow metadata in the migrated custom image savers. Setting it to `1` opts into metadata; review filenames, prompts and library identifiers before sharing any output.

Internal ports are fixed: public authenticated gateway 8188, private ComfyUI 8189, private Ollama 11434. They are deliberately not a normal set of knobs. Do not expose 8189 or 11434 in RunPod.

`OLLAMA_MODEL` / `OLLAMA_ANALYSIS_MODEL` are inherited advanced helper overrides. The shipped helpers are 32B F16. Changes require compatibility validation, and no smaller or cloud fallback is supplied. Prefer the private `/workspace/config` settings only after understanding the source helper's role and memory handoff.

The runtime strips provider tokens, private library values, and the gateway password from the ComfyUI child environment. This is not a security boundary against arbitrary code running as the same container user. Third-party custom nodes remain trusted executable code.

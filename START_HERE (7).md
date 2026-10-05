# H3 Portrait 1.3

Use the repository-root [START_HERE.md](../START_HERE.md) for complete current
upload, build, template, and usage instructions. Do not layer older update ZIPs
or live-hotfix scripts over this release.

Expected startup prefix:

```text
H3 PORTRAIT 1.3 | Instruct analysis + Thinking director
```

`VERSION`, `settings.json`, both instruction files and `verify_release.py` belong
together. Source, image bundle and active node copies are verified before model
downloads. The new GitHub commit appears in the build summary and startup log.

Pass 1: abliterated 32B Instruct Q4, all photo previews, compact reference map.
Pass 2: abliterated 32B Thinking Q4, text-only map + brief, MiniMax guide prompt.
The first model unloads before the second; Ollama unloads before H3 starts.
One bounded Instruct regeneration is allowed on malformed/truncated output.
No OmniNode, RefMod, manual reference grammar, or new LoRA catalog is required.

The analysis model is configured and downloaded automatically. Keep your existing
`OLLAMA_MODEL` set to the 32B Thinking tag for the director. This uses two separate
model downloads (roughly 21 GB each), not both models resident together.

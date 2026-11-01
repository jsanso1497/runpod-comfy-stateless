# Audit and research record: 2026-11-01

## Reviewed source

GitHub main: `0a0f8b0fc29e88a37ef027e4474dc16cf1a6b743`.
The source subtree reconstructed from the previously supplied packages matched
GitHub's full `h3_character_swap` tree hash exactly:
`81009d3a1d927f61155f046f06ea4f01c4e33c9b`.
This is not a claim that the currently running Pod has that exact image digest.

## Confirmed gaps and changes

1. Runtime unconditionally passed `--disable-metadata`. Metadata is now on by
   default with an explicit opt-out; model and prompt data are not reconstructed
   from a rendered picture.
2. The disk runner queued worker API prompts without an editable workflow in
   extra_data and the final FFmpeg encode stripped all metadata. It now stores
   reconstructed public controller canvases, attaches the resume controller to
   worker history and embeds the repeat controller in the final MP4. Exact
   worker API JSON remains separate, accurately labeled.
3. Canceled native generations could lose their unsaved canvas. A non-mutating
   on-prompt observer now records submitted H3 workflow/API data before sampling.
4. The simple single-reference chain had only been a downloadable JSON. It is
   now a named factory preset. The native preview variant uses that same 110-
   character short instruction. No inference that longer prompts are better.
5. The earlier FileBrowser add-on was not in the reviewed tree. Its private
   authenticated integration was merged without overwriting the later long-
   analysis and face/body work. The 32768 default helper context is retained.
6. The long-analysis logger ignored `message.thinking`. It now counts that field
   separately without retaining its text; no helper-model switch is imposed.
7. Preview support already exists in the exact pinned ComfyUI. The integration
   enables that supported path instead of monkeypatching a new sampler or
   changing the H3 generation VAE. A small actual-decoded preview tap is inserted
   between native decode and append; it returns the original tensor unchanged.

The reported skin-color/realism gap between native and the disk chain is NOT
proven fixed by this patch. No matched native-versus-chain GPU experiment was
possible. Source normalization, mask tracking across jobs and sampling history
remain as previously implemented. The new simple preset helps compare inputs,
and the new records make an actual matched comparison auditable.

## Primary sources inspected

- Pinned core preview implementation (TAEH3, first latent frame, nested AV):
  https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/latent_preview.py
- Pinned H3 latent format declares the `taeh3` tiny decoder:
  https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy/latent_formats.py
- Native SaveImage/PreviewImage metadata and IMAGE pass-through:
  https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/nodes.py
- MP4 custom metadata is preserved through use_metadata_tags:
  https://github.com/Comfy-Org/ComfyUI/blob/79be670e2d9be63e238785af307369d2b9039ed1/comfy_api/latest/_input_impl/video_types.py
- Official metadata meanings and limitations:
  https://docs.comfy.org/development/api-development/workflow-metadata
- TAEH3 tiny decoder, pinned author revision:
  https://github.com/madebyollin/taehv/tree/011dfc2112197741c540e0bdd5b7b67bcc930771
  Asset: safetensors/taeh3.safetensors, 22,709,752 bytes;
  Git blob: 9eff5f6da1f0baa60ede7f097507119b1488219f.
  The actual binary was unavailable in this container; Git blob verification
  and a real CPU decode are build gates, not reported authoring results.
- FileBrowser archive/security guidance and pinned release:
  https://github.com/filebrowser/filebrowser
  https://github.com/filebrowser/filebrowser/releases/tag/v2.63.23
- Ollama separate answer/thinking streams:
  https://docs.ollama.com/capabilities/thinking

Upstream community preview overrides were reviewed as context, but no new
third-party node pack, approximate full-video decoder, sampler wrapper or
unqualified ComfyUI upgrade was adopted.

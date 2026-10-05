# H3 Portrait 1.3: clean repository deployment

Use the repository-root **[START_HERE.md](../START_HERE.md)** for this reset.
It is the authoritative upload, clean-replacement, build, and RunPod instruction
file. Do not use an earlier patch guide after this complete snapshot.

The portrait code and workflow remain version 1.3.0. Use the newly published
`ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean` image tag after the
**Build H3 Portrait template** job succeeds. That job also publishes the original
`h3-portrait` tag for compatibility.

Your workflow is **H3_Portrait_Auto**. Start in **Draft only**, Standard, 5 seconds,
prompt_variation 0 and seed 42. Upload references, write a normal instruction, and
select a compatible full H3 Ref2VA LoRA. The source list is the repository-root
`config/lora_links.txt`; it remains one link per line with no hashes or profiles.

The build includes CPU startup/schema checks. Neither the complete Docker build
nor live GPU inference for this snapshot has been executed in the assistant's
environment. See the repository-root [VALIDATION.md](../VALIDATION.md).

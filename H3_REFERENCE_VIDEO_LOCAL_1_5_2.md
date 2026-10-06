# H3 Portrait 1.5.2: local Hearmeman Reference Video workflow

This release starts from the attached `New Compressed (zipped) Folder(2).zip` source and adds a local-only adaptation of Hearmeman24's MiniMax Reference Pack workflow. Existing Krea Identity 1.2.0, H3 Media 1.2.0, SeedVR2, automatic H3 portrait workflows, manual Ref2VA workflows, models, and shared LoRA configuration are retained.

## What is new

New workflow names after the image is rebuilt:

Full:

```text
H3_Reference_Video_Swap_Local_Full_v1_5_2
```

Lite:

```text
H3_Reference_Video_Swap_Local_Lite_v1_5_2
```

The workflow accepts one or more subject images plus a reference MP4. The reference video can provide motion, pose, camera movement, framing, scene, lighting, and an enabled soundtrack. MiniMax H3 Ref2VA generates a new video using those references.

The workflow is generative. It is not a pixel-exact face swap and it does not guarantee that every frame of the source clip is preserved.

## Local-only prompt writing

The image build installs the pinned Hearmeman24 `ComfyUI-MiniMaxRefPack` 0.3.5 source at commit:

```text
7012734eabf6f98063d6eaf8ce1f9264ee803664
```

The build then patches that copy before it reaches ComfyUI:

- `prompt_provider` options are limited to `local` and `none`.
- `local` is the default.
- OpenRouter model discovery is disabled.
- Hosted OpenRouter execution is blocked at runtime.
- `OPENROUTER_API_KEY` and `LLM_KEY` are unset by the H3 startup script.
- No Gemini or other hosted prompt model is required.

The Reference Pack talks to the existing local Ollama OpenAI-compatible endpoint:

```text
http://127.0.0.1:11434/v1
```

Full uses the existing local analysis model:

```text
huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M
```

Lite uses:

```text
huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M
```

The local VLM receives sampled frames from the reference video. It does not listen to the video's audio when writing the prompt. The native H3 graph still receives the prepared reference video and enabled soundtrack/audio sockets for generation. The existing Ollama service also runs with `OLLAMA_KEEP_ALIVE=0`, so the local prompt model is not intended to remain resident when H3 takes over GPU memory.

## Full quality defaults

Full keeps the source ZIP's BF16 H3 stack and defaults the new workflow to:

```text
quality: High fidelity
sampler: res_multistep
scheduler: simple
steps: 25
reference image size: max
fps: 24
```

Available aspects:

```text
9:16  -> H3 768x1344, exact delivery crop 756x1344
2:3   -> H3 768x1152
16:9  -> H3 1344x768, exact delivery crop 1344x756
```

The small center crop on 9:16 and 16:9 creates the exact requested ratio without stretching the H3 frame.

Lite uses its existing full-architecture INT8 H3 stack with lower-memory canvases and a 16-step Standard default.

## How to use it

1. Rebuild the Full or Lite image after installing this source update. Importing only the JSON will not install the Reference Pack.
2. Open the new `H3_Reference_Video_Swap_Local` workflow for your profile.
3. In **MiniMax References Manager**, add one or more subject images.
4. Add the MP4 you want to use for performance, pose, camera, scene, or soundtrack reference.
5. Keep **Draft only** selected and queue the graph once.
6. Review the locally written prompt and Reference Pack debug text.
7. Adjust the direction field if necessary.
8. Switch only the review node to **Generate video** and queue again.

A useful direction is already included. For a single-person replacement, describe which person in the video should be replaced if there is any ambiguity. Subject images should be sharp, consistent, and all show the same intended identity.

## Voice and audio

If the reference MP4 contains the voice you want H3 to condition from, leave its soundtrack enabled in MiniMax References Manager. The soundtrack is provided to H3 as a native reference. If you want new dialogue, put the requested dialogue in the direction/prompt.

This is reference-conditioned audio generation, not deterministic voice cloning. Similarity must be evaluated from live H3 renders.

## Output

The workflow saves the generated MP4 and the exact final decoded frame of the completed MP4 as a matching PNG for continuation/chaining work.

Default output folder:

```text
ComfyUI/output/H3_Reference_Video_Swap/
```

## RunPod images

Full image:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean
```

Lite image:

```text
ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-lite
```

Run the corresponding GitHub Action after merging/uploading the source update. A currently running Pod will not gain the new node pack from the workflow JSON alone.

## Validation boundary

This package can validate source integrity, graph structure, local routing, profile configuration, unit behavior, build instructions, and the Reference Pack source patch locally. Actual H3 image quality, voice similarity, VRAM use, generation speed, and a complete Docker/RunPod GPU render require the rebuilt image on a GPU Pod.

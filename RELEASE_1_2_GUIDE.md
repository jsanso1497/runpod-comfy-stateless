# Krea Identity + H3 Media HQ 1.2.0

This cumulative update adds two workflows to the supplied repository plus the earlier Krea 1.1 update. It does not push a live GitHub commit. It preserves the seven prior Krea workflow JSON files, existing H3 workflow JSON files, existing model catalogs and the SeedVR2 CPU registration fix.

## Install

Merge this update ZIP into the repository root, including the hidden `.github` directory and `SOURCE_SNAPSHOT.json`. Merge directories; do not delete existing directories because the update contains only changed and additional files. Keep your existing `config/lora_links.txt`, which is deliberately not distributed. Commit, run the selected image build from the new commit, and deploy that newly built image. Do not simply rerun an old failed build.

The General/Image, Full and Lite Docker recipes all bundle the new node source. Existing installed workflows are not overwritten on Pod restart. Import a newly versioned workflow to use the additions.

Krea-only users keep the current template:

```text
MODEL_PROFILES=seedvr2
ENABLE_KREA_IDENTITY=1
KREA_IDENTITY_UPSCALE=1
ENABLE_OLLAMA=0
ENABLE_H3_MEDIA=0
```

To enable the new H3 MP4 workflow in the General/Image deployment, change only:

```text
ENABLE_H3_MEDIA=1
```

The H3 workflow does not need Ollama or an external inference API. Existing Full/Lite startup behavior is unchanged, including their existing Ollama services.

**Resource warning:** the new HQ H3 workflow always uses the full BF16 H3 diffusion model and BF16 32B encoder. Its four model files total approximately 115 GiB according to the inherited catalog. Already-present matching files are verified and reused. On a Krea-only Pod these are additional downloads; on Full they normally reuse its full-model assets. On Lite, enabling this module downloads the full BF16 files in addition to any Lite-specific weights. This is not a lightweight H3 option. Do not assume a GPU/RAM allocation suitable for Krea alone is suitable for this H3 stack. Peak RAM/VRAM and generation speed have not been measured here.

No HTTP ports, registry secrets or container start commands change. This is source plus workflows, not model weights.

## 08: Krea, two references, described scene

Open `Krea_Identity_08_Two_References_Described_Scene_v1_2`.

Upload exactly two original photographs. There is no scene-photo loader, blank placeholder scene, video stage or reference-sheet intermediate. Each original reference is connected independently to both Krea semantic grounding and image-appearance conditioning.

In **START HERE - describe the NEW scene**, select the relationship:

| Mode | What to supply |
|---|---|
| One person - complementary references | Two views of one person, or one face photograph plus one body photograph of that person. |
| Two people - separate identities | One reference per person. Use the text to place person A and person B in the new scene. |

The two **use** fields say what each reference contributes. They are instructions, not mathematical feature masks. For a face/body pair:

```text
Reference A use:
Facial identity, eye shape and spacing, nose, mouth, jaw and natural complexion.

Reference B use:
Physique, body proportions and hair. Use image 1 for facial identity.
Do not copy this photograph's background, pose or lighting.

Scene description:
A photorealistic full-body photograph of this person standing beside a cafe
window in soft morning light. Eye-level camera, relaxed posture, hands visible.
Plain gray T-shirt and dark jeans. Natural skin texture and consistent shadows.
```

For two people, change the use fields to person A identity and person B identity, and state their placement explicitly. The node separates the identities in the prompt, but the model can still mix features.

**Orientation:** choose landscape, portrait or square independently of either reference photograph. The canvas follows the selected aspect ratio on a /16 grid. The working size defaults to 1.5 megapixels and is capped at 2 megapixels. A preset may land slightly below the requested megapixel count to satisfy both the exact aspect and grid.

**Quality baseline:** Krea 2 Turbo BF16, BF16 Qwen3-VL encoder, the existing Identity Edit v1.2 LoRA, 12 steps, CFG 1, Euler/simple, full denoise on an empty output latent, FIT, grounding 1024. These are unchanged model choices, not a newly trained subject LoRA.

Because neither input is a scene, the strength node is labeled **Reference A=4 / Reference B=4 / FIT**. This balanced pair is a starting setting, not an empirically established optimum. With a face/body pair, reduce B to 2 or 3 when its face, clothing, pose or background competes with A. Fix the seed and change one setting at a time.

The exact screenshot rebalancer remains available, initially off:

```text
multiplier = 1.0
per_layer_weights = 1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

Save and judge the native image first. Use existing workflow 04 to upscale an approved result. Neither more steps nor larger final pixel dimensions guarantee better likeness.

## H3 HQ: MP4 voice, identity and action

Open `H3_HQ_MP4_Voice_Identity_Action_v1_2`.

This is an additive HQ media variant built on the attached repository's native full BF16 H3 pipeline. It leaves the existing H3 graphs unchanged. A separately discussed HQ package from another conversation was not available as source here, so this package does not claim to patch that unseen file byte-for-byte.

The initial graph contains two image loaders, one native **LoadVideo** MP4 loader, labeled reference nodes, a scene/dialogue node, native H3 conditioning and sampling, and generated video/audio export. Default settings are full BF16 diffusion and encoder, 25 res_multistep/beta steps, 24 fps, an eight-second target and `ref_image_size=max`. `max` applies to still-reference conditioning, not output resolution, and can cost substantially more memory and time than `match`.

### Start with voice only

Upload the face and body reference photographs and an MP4 containing a clear example of the desired voice. In **MP4 roles**, keep:

```text
visual_role = Ignore video (voice only)
audio_role = Voice timbre - NEW dialogue
start_seconds = 0.0
duration_seconds = 5.0
applies_to = the main subject
```

Choose a clear section with one speaker. Start around three to eight seconds, rather than passing an entire video. The decoder reads only the selected segment. It decodes the source audio to floating-point PCM at its source sample rate, without an intermediate MP3, automatic voice enhancement or normalization. Mono and stereo are supported; additional source channels are downmixed to stereo. `audio_track=0` selects the first audio track; it is not a speaker-separation control.

**Voice-only does not decode or supply the MP4's video frames to H3.** The clip's person and background therefore are not additional visual-conditioning inputs. This is stronger isolation than merely asking the prompt to ignore them.

In **START HERE**, describe the output scene and type the new words:

```text
Scene description:
A natural medium shot of the main subject seated at a kitchen table,
looking toward the camera. Relaxed posture and a small natural smile.
Soft window light, a locked eye-level camera and no background music.

New dialogue:
I wanted to show you how this works. Here is the finished result.

Speaker target:
the main subject
```

`Speaker target` must match the voice reference's `Applies to` text. The builder adds the dialogue markup, stable speaker identifier and audio-reference relationship automatically. Enter plain dialogue in the normal field. Keep the line short enough for the selected output duration.

**The output contains newly generated H3 speech, not the MP4 soundtrack pasted onto a new video.** The clip guides audible characteristics such as voice timbre and delivery. Exact voice cloning, word-perfect delivery and perfect lip sync are not guaranteed. No separate voice-cloning model is installed or trained.

### Enable visual references independently

| Visual role | What actually reaches H3 |
|---|---|
| Ignore video (voice only) | No frames. Audio may still be used. |
| Identity - selected frame | One original frame supplied as a still reference; no motion sequence. |
| Action / motion | The trimmed sequence, converted to 24 fps and supplied to the native video-reference input. The prompt asks it not to transfer the performer's identity. |
| Identity + action | Both a selected original still and the trimmed sequence. |

The **audio role** remains independent. For a silent movement guide choose `Ignore audio`. For the same person's voice and appearance choose `Identity + action` with `Voice timbre - NEW dialogue`.

`identity_frame_position` is relative to the selected segment: 0 is the beginning, 0.5 is the middle and 1 is near the end. Scrub the MP4 preview to choose a clear frame. The clip report gives the actual selected timestamp. Selected stills are downscaled only when necessary, up to a 2048-pixel short edge with a 12-megapixel safety bound; no generative restoration is applied. Motion frames use a bounded approximately 1.1-megapixel decode canvas before native H3 sizing.

**Action is native video conditioning, not merely a written description of the action.** However, the action-only instruction does not remove the performer's visible identity from those frames. Unlike voice-only isolation, this remains a model-dependent separation. It is not skeletal retargeting, frame-locked choreography, exact source-video editing or a pose-control model.

H3 requires a 17k+5 frame grid. The adapter resamples to 24 fps, then trims the reference sequence down to that grid, without looping or speeding it up. The native node also truncates motion references longer than the target clip. The requested generated duration rounds up to its native frame grid, so actual output may differ slightly from the nominal seconds. The reference map reports the output frame count and actual duration.

The landscape default generates at the native 1344x768 canvas. Other selectable native canvases include 768x1344 portrait, 1152x768 / 768x1152, 1024 square and 896x1120. The 1344/768 pair is approximately, not exactly, 16:9; there is no hidden output crop or upscaling in this new graph.

### More clips and existing audio files

Duplicate **LoadVideo + H3 HQ MP4 Voice / Identity / Action**, then chain its `references` input from the prior reference node and connect its output to the next node or the plan. Label each clip and specify which person it applies to.

Up to three source videos are accepted, with native totals of nine still references, three motion references and three standalone audio references. Selected identity frames count against the still total. A source clip using both voice and action occupies one audio and one motion slot.

For existing WAV/MP3-style inputs, add native **LoadAudio** and **H3 HQ Standalone Audio Reference**, then insert that node into the same reference chain. To use no MP4, remove its two nodes and connect the previous reference list directly to the plan. Do not leave an empty required file loader in an active graph.

The normal dialogue builder targets one speaking subject. The **Advanced prompt** field accepts a complete hand-written H3 prompt for more complex or multi-speaker scripts. It replaces, rather than appends to, the generated prompt. The reference-map preview shows the actual `<Picture N>`, `<Video N>` and `<Audio N>` assignments. Nonexistent numbered references raise a clear error. Audio is intentionally supplied through standalone reference inputs, with no duplicated paired soundtrack.

H3 saves an H.264 MP4 at CRF 18 with its generated audio and a matching last-frame PNG extracted from that completed export for chaining.

### Input limits and practical cautions

Use locally uploaded, file-backed MP4s. This adapter deliberately does not fetch remote video URLs or decode an entire generated in-memory VIDEO object. MP4 codec support depends on the bundled FFmpeg decoder. HDR visual footage is rejected with an SDR-conversion message rather than silently feeding incorrect colors. Voice-only extraction can still use an HDR clip's audio because its frames are ignored.

Labels define reference roles, not guaranteed locks. Contradictory identity references can still drift or blend. Use references and voices you have permission to use. Review generated identities and speech before sharing.

## Validation and build behavior

The source includes CPU unit tests, real FFmpeg fixture tests, deterministic graph checks, source-snapshot verification and an actual ComfyUI node-registry gate in the Docker recipes. The new adapter rejects an incompatible native H3 execution signature instead of silently dropping MP4/audio inputs.

The Krea graph generator now bundles its original schema prototypes so tests also run in Docker's isolated `/opt/krea-identity` layout without relying on a nonexistent sibling `/opt/config`. Only a repository-specific Dockerfile assertion is skipped in that isolated test layout; the actual ComfyUI registry gate is not skipped.

See `RELEASE_1_2_VALIDATION.md` for the executed results. Full Docker build/push, live native node execution, model downloads, RunPod GPU inference, memory benchmarks and visual/voice fidelity comparisons were not run in this environment.

## Primary references checked for this integration

- Krea editing node, dual conditioning and reference arrangements: https://github.com/lbouaraba/comfyui-krea2edit/blob/main/README.md
- Krea editor capabilities: https://huggingface.co/conradlocke/krea2-identity-edit
- Native H3 workflow and model behavior: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native
- Official full-reference prompting, dialogue and audio reference versus reuse: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Native H3 reference implementation: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
- Native VideoFromFile source contract: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_api/latest/_input_impl/video_types.py

Upstream public source was reviewed as of 2026-10-06. The repository remains pinned to its existing ComfyUI commit. A real Docker build is required to validate that installed pinned runtime rather than assuming public master and the installed build are identical.

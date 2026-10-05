> Legacy/general workflow documentation. For your H3 Portrait clean reset, use START_HERE.md at the repository root, not this guide.

# H3: your reference definitions -> Ollama prompt -> native reference video

This is a small ADDITIVE patch for your working ComfyUI Quality 3.2 installation.
It does not use RefMod in the new workflow. Your existing Krea, SeedVR2, RefMod and
multi-reference workflows remain unchanged and available.

## 1. Upload this patch to GitHub

Upload ALL extracted contents into the root of:
`jsanso1497/runpod-comfy-stateless`

Replace exactly these three existing files:

- `config/runtime.json`
- `scripts/check_workflows.py`
- `scripts/prepare_image.sh`

Add the new `scripts/local_nodes/ComfyUI-DirectedH3/` folder, the new workflow
`config/workflows/26_H3_User_Directed_Ollama.json`, and
`tests/test_user_directed_h3.py`. Keep all the other existing files in place.

The replacement prepare_image.sh retains the successful fix that explicitly
selects current tests rather than importing the retired upload-app tests. It adds
this patch's regression suite. The Rebalance import fix is not overwritten.

DO NOT change or replace your Dockerfile, GitHub Actions workflow, .dockerignore,
check_rebalance_runtime.py, model/LoRA catalogs, or existing node pins.

Commit to main. Wait for the new build to succeed. This patch has NOT been
committed to your live GitHub repository by the assistant.

## 2. RunPod: no changes

Keep `ghcr.io/jsanso1497/runpod-comfy-stateless:latest` and every current environment
variable, secret, disk setting, registry credential, and port. No extra models,
LoRAs, API keys, ports or Python dependencies are added. Your existing local
abliterated 32B FP16 Ollama helper and full BF16 H3 setup are reused.

After the successful build, save any old Pod files you need and deploy a fresh
Pod. Wait for model preparation and this exact line:

`WORKFLOW SCHEMA PASS: 26_H3_User_Directed_Ollama.json`

The base startup banner remains version 3.2 because this is an additive workflow,
not a base-runtime upgrade. Wait for `[OllamaH3] ready` before drafting a prompt.

Open ComfyUI on HTTP 8188. The new workflow appears in the Workflows sidebar as:

`26_H3_User_Directed_Ollama`

## 3. Define the references yourself

There are nine image slots, each paired with a definition card. Fill Reference 1
onward without gaps. Leave only trailing unused image slots at `(none)`.

For EACH selected image, fill in all three definition fields:

- `belongs_to`: your own stable label, for example Person A, Person B, or Room.
  Use the SAME label on every photo of the same person/content unit. Matching is
  case-insensitive. Different labels remain different subjects; nothing inspects
  a face to decide ownership. Do not type <Subject N> in this field.
- `what_image_contains`: YOUR description of the reference, including which
  person/item is the target if other people/items appear in that photo.
- `use_this_reference_for`: exactly what to carry into the output and what to
  ignore. Explicitly state identity, outfit, pose, environment or style priority.

Example:

| Image | belongs_to | what_image_contains | use_this_reference_for |
| --- | --- | --- | --- |
| 1 | Person A | Headshot of Person A, facing the camera. | Primary facial identity and hair. Ignore the background and clothing. |
| 2 | Person A | Full-body view of Person A wearing a blue jacket and dark trousers. | Body proportions and the target outfit. Do not copy the standing pose. |
| 3 | Person B | Headshot of Person B with their hair tied back. | Primary facial identity and hairstyle. Ignore the background. |
| 4 | Person B | Full-body view of Person B wearing a gray sweater and jeans. | Body proportions and target outfit. |

Replace the example descriptions with what YOUR photographs show. There is no
built-in headshot/body/wardrobe priority. You set it in `use_this_reference_for`.

A scene or object can use its own label, such as Room or Red bag, rather than a
person label. Each card defines one target content unit. For a group photo,
describe which person belongs to that card. It is possible to intentionally use
the same image again for another target, but that sends another reference and
adds processing cost.

`CHECK: the map YOU defined` displays the fixed assignments to <Picture N> and
<Subject N>. You do not need to create those technical labels yourself.

## 4. Define the action

In `2. YOU direct the action; Ollama formats it`, fill in:

`actions` example:

```
Person A stands on the left and Person B stands on the right.
Person A takes a small step toward Person B. They share a brief natural hug,
then remain close together. Show only these two people.
```

`scene_camera_audio` example:

```
A neutral light-gray studio background with soft even lighting.
One continuous medium shot. Stationary eye-level camera.
No dialogue, no music, and no on-screen text.
```

Use `preservation_rules` for any additional constraints. Its initial wording
asks the model to follow YOUR per-reference roles, preserve identities and avoid
beautification or blending different people.

Leave `system_prompt` at the supplied value. It is an implementation-specific
adaptation of MiniMax's full-reference writing guide, not an unmodified copy of
the guide and not a file you must find/install separately.

Leave `let_ollama_look_at_images=false` for your first test. This means the prompt
writer uses YOUR written descriptions rather than guessing visible details. H3
still receives EVERY original photo through its own native vision/VAE reference
path. Switching this to true allows supplemental inspection but does not let
Ollama assign ownership or rewrite the fixed subject-definition section.

## 5. Draft, review, then render in the SAME workflow

First keep:

```
mode: Draft prompt only
length: 124
width: 1344
height: 768
variation_seed: 42
temperature: 0.2
save_prompt: true
```

Click Run. Only prompt writing executes. The H3 model loaders, reference VAE and
sampler are blocked by the review gate, so this does not start the video render.
The uploaded images still load into CPU memory for provenance and the later H3
run. The big Ollama model runs and then unloads.

Read `3. REVIEW DRAFT - copy ALL text to reviewed_prompt`.

The final output has the six H3 sections. The code creates `subject_definitions`
from your cards; Ollama writes the other five sections. It cannot overwrite that
section. Your actions, scene/camera/audio and preservation rules are also retained
verbatim in a HUMAN DIRECTION block. The generated text is checked for unknown
reference/subject labels and attempts to redefine picture assignments.

This is NOT a proof that the generated prose contains no semantic mistakes.
Check that it assigns the requested action to the right person and introduces
nothing you did not request.

When satisfied:

1. Copy the ENTIRE draft preview, including its first `[H3_DIRECT_INPUTS ...]` line.
2. Paste it into `reviewed_prompt` in the same prompt node.
3. Set `mode` to `Render reviewed prompt`.
4. Click Run again.

No second Ollama request is made. The header binds the draft to these exact
pictures, definitions, directions, dimensions and duration. It is removed before
H3 receives the prompt. You may refine the narrative in the five generated
sections, but keep the fixed subject definitions and HUMAN DIRECTION block.
To change who owns a reference or what someone does, edit YOUR fields and draft
again. A stale draft stops before loading H3 rather than using mismatched images.

Result: `7. RESULT VIDEO`, also saved under `ComfyUI/output/video/`.

The one-click mode `Write prompt and render` is available, but it deliberately
runs straight through without a review pause. It is not the initial mode.

## 6. What stays the same

Native MiniMax H3 ref2va full BF16 model and BF16 encoder; original reference
images; video and audio VAEs; `ref_image_size=max`; 25 steps; res_multistep/normal;
24 fps; 124-frame test shot; seed 42; optional H3 LoRA off. No Turbo/Lightning,
RefMod, face replacement, Rebalance or automatic upscaling in the new graph.

Original photos are supplied in order to native MiniMaxH3ReferenceToVideo, with
its video VAE CONNECTED. Do not copy the old RefMod graph's disconnected-VAE setup.
Do not reconnect/rearrange the supplied model-loading review gate.

A free-form action brief is not hard motion control. Written identity/ownership
priorities do not guarantee perfect likeness or correct interactions in the video.
Inspect the result against your photographs.

Prompt records are saved in `ComfyUI/output/prompt_assistant/`, including the
copyable `.review.txt` and a JSON of your inputs. Save these and your outputs
before terminating the Pod. No external hosting or persistent storage was added.

## Source basis

The current 3.2 package and its two successful build-check fixes supplied in this
conversation were the integration base. The application/node/model pins were not
changed. Official sources consulted for the new behavior:

```
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py
https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_execution/graph.py
https://docs.ollama.com/api/chat
```

The graph uses the exact native reference node and model-loading patterns from
your earlier workflow. The director fields, deterministic map, review fingerprint,
and approval gate are additions made for this request.

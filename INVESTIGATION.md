# H3 reference investigation and release decisions

## Evidence and limits

The report describes a three-reference run in which the third pose/camera image appeared to
supply the video's person instead of the first two identity images. The actual three images,
final H3 prompt record, LoRA and resulting clip were not supplied for this investigation. The
specific run therefore cannot be conclusively diagnosed or visually reproduced here.

The live repository connector returned 404. Source review used the attached complete source
package (Portrait 1.3) and ComfyUI code pinned by that package at:
`65787d668397d230bf5839d69a0a7239e2dad378`.

## What the old source does

H3PortraitDirector hands every uploaded image to the job. H3PortraitConditioning constructs
an insertion-ordered dict of ref_image_0, ref_image_1, etc. and calls native
MiniMaxH3ReferenceToVideo.execute with the video VAE.

The pinned native implementation loops over ALL ref_images.values(). Each image is appended
to the text/vision encoder presentation. With the VAE connected, each also becomes an image
latent in minimax_refs. Those references condition the denoising trajectory. There is no
role-specific identity/pose strength input in that node. The source does not overwrite the
first two images with the last one, and the reference path does not automatically designate
the last photo as a first-frame keyframe.

This supplies a concrete route by which an unrelated person's pose photograph competes with
the desired person's appearance. A prompt saying 'pose only' is a semantic instruction, not a
hard exclusion of that photograph's identity pixels. It is a plausible contributor to the
reported result, not proof that image competition is its only cause. Prompt rewriting, LoRA
behavior, reference quality, model conditioning strength and sampling remain other variables.

Primary source inspected:
https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_minimax_h3.py

## The implemented correction

The uploader accepts explicit per-image roles. These override automatic role guesses.
The analyst receives the entire original ordered image set plus roles and brief. It returns
compact role/evidence entries. Pose/camera and expression-only photos are routed to text-only
geometry/expression guidance by default; their raw pixels are excluded from H3, including its
vision encoder and image-latent conditioning. Other appearance references remain visual.

Filtering renumbers native Picture tags. The application records original upload number,
role, delivery route and native Picture number, and rewrites source-number aliases in the
appended user brief. It handles numeric, ordinal and common plural/range image references.
A missing source number fails instead of silently addressing the wrong photo. The complete
unmodified brief remains in the saved prompt record for audit. The geometric guidance is
included in both the director's evidence and the final target prompt. A display panel prints
the actual map, not merely what the language model intended to do.

This is not a pose-control network, and text cannot fully reproduce every camera/limb detail
in a photograph. There is an explicit All images visual comparison mode for users who prefer
native visual pose fidelity and accept its identity competition. It is not the default.
Face, body, wardrobe and scene roles still rely on text to limit which aspects of their raw
visual pixels are transferred. The release does not claim perfect disentanglement or identity.

The analyst instructions now treat differences outside an assigned role as irrelevant rather
than as conflicting instructions. Jewelry in a face photo and exposed skin in a body photo
are not instructions about the video's wardrobe. Clarification is reserved for unresolved
target selection or materially conflicting explicit user requirements. This policy reduces
unnecessary clarification; a smaller or larger language model can still misinterpret a task.
With explicit role selections, a clarification receives at most one bounded reassessment
within the existing analysis deadline. A repeated question still stops the job. No question
or missing map is silently discarded to force video generation.

Official reference guide, paraphrased into the included instructions:
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

## Widget corruption

The old workflow serialized Python inputs but omitted ComfyUI's automatically added seed
control widget. Consequently use_loras=true landed in control_after_generate, and later
combo/numeric values shifted, producing the reported mode=0/NaN/LoRA=1 screen.

The new graph includes 'fixed' after seed. Defaults are checked against the current Python
schema PLUS the frontend-added seed companion, with enum and finite-number validation.
The JavaScript migration corrects untouched old arrays and resets already-coerced broken
controls to safe Draft-only/(none) values. New valid saved values remain untouched.

Frontend source explaining the additional widget:
https://github.com/Comfy-Org/ComfyUI_frontend/blob/7989ad6b0623bc6f708e4b9c06ca9ae3d602d496/src/renderer/extensions/vueNodes/widgets/composables/useIntWidget.ts

## Last frame

The exporter passes the exact completed MP4's path and file identity to the final-frame node.
FFmpeg decodes video frames in presentation order while retaining only the last complete RGB
frame. The PNG is therefore the actual final encoded-video appearance after the aspect crop,
not the last pre-compression tensor. A sidecar records decoded frame count and zero-based
last index. Bounds, cancellation, unsafe paths, missing files and changed exports fail clearly.
A real local synthetic MP4 with audio was tested; its last RGB frame matched an independent
FFmpeg selection of the known last index. Actual native Comfy Video.save_to execution is a
new Docker-build CPU smoke test and still must run in GitHub before publication.

Native video-save interface inspected at the same pinned ComfyUI commit:
https://github.com/Comfy-Org/ComfyUI/blob/65787d668397d230bf5839d69a0a7239e2dad378/comfy_extras/nodes_video.py

A saved PNG does not alone guarantee seamless chaining. Ref2VA identity references are not
hard first-frame anchors. The appropriate keyframe/continuation setup is a separate task.

## Lightweight choice and other changes

Lite uses official full-architecture Ref2VA INT8-convrot weights and the INT8-convrot H3 Qwen
encoder. These reduce downloaded model sizes and model-memory pressure. The separate Ollama
helper is 8B Instruct Q4_K_M, reused across both passes; there is no Thinking stage in Lite.
Standard uses 16 steps and a 576-wide portrait canvas. The native manual workflow bypasses
Ollama entirely. The package preserves the full, not pruned, model family to avoid the known
adaln incompatibility of full-trained LoRAs with pruned models. No user's LoRA was inspected.

No turbo adapter is automatically loaded: the official Comfy guidance warns that a four-step
reference schedule can weaken reference adherence. Both new portrait graphs use the documented
res_multistep/simple combination rather than the prior normal scheduler. This aligns the
configuration with upstream recommendations; it is not a measured quality improvement here.

Sources for sizes, model families and schedule decisions:
https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors
https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
https://ollama.com/huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M
https://docs.comfy.org/tutorials/video/minimax/minimax-h3

Versioned workflow filenames prevent an old user-saved graph from hiding this release's
corrections. Source/profile/workflow/catalog manifests are validated before model downloads.
Full and Lite publish separate tags from the same repository, without requiring the general
ComfyUI image to be rebuilt first. The existing base-image layers are retained, so 'Lite'
describes the running model workload, not an entirely new small Docker foundation.

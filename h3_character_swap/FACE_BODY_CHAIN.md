# Low-memory H3 chain with separate face and body references

Release: **2026-10-27.2**. Incremental update for the current H3 character-swap
source at GitHub commit `073d7404836b5fab27ce754fcfbcb937975910c1`.
The prior long-analysis update is included in that baseline and retained.

## What is added

Two additional workflows, without replacing the original single-reference ones:

| Workflow | Purpose |
| --- | --- |
| `04_Analyze_Face_Body_Build_Prompts` | Long, role-separated analysis of two photographs and source-video samples. |
| `09_H3_LowMemory_Face_Body_Chain` | The disk-backed chain, with two reference photos supplied to EVERY generation job. |

The factory folder is `H3_Character_Replacement_2026-10-27.2` after rebuilding.
`00_Analyze_Clip_Build_Prompts` and `06_H3_LowMemory_Chain_Control` remain available
for the original single-reference approach. Existing saved personal workflows
are not overwritten. No new weights, sampler changes, resolution changes, BFS
workflow, upscaler, or automatic FileBrowser installation are added.

This is an **experimental adaptation** of Akatz V1.5. Native H3 supports multiple
reference images, but its model does not expose separately enforced face and body
regions or independent reference strengths here. More deliberate prompting is
worth comparing; it is not proof of improved identity or body transfer.

## Reference assignment

| Reference | Supplies | Does not supply |
| --- | --- | --- |
| **Picture 1: face close-up** | Facial identity, head/face proportions, ears, hairline and hairstyle. | Outfit, cropped shoulders, static pose or studio background. |
| **Picture 2: body/outfit photograph** | Physique, relative proportions, complete clothing, accessories and shoes where visible. | Facial identity, hair, source performance or scene. |
| **Video 1: source** | Selected performer's movement, expression, gaze, timing, contacts, camera and scene. | Replacement face, physique or wardrobe. |

Use two separate photographs, not the combined two-panel sheet for this variant.
A clear head/shoulders close-up and a full-body/three-quarter image are sensible
starting inputs. Prefer a coherent, compatible pair. A head visible in the body
photograph is explicitly excluded from the identity role by the text; it is NOT
removed or masked from the actual image. A neck-down crop is an optional comparison,
not something the workflow applies automatically. Unseen detail is not recoverable
merely by requesting it in a long prompt.

The same video VAE is wired to native reference conditioning. Both images therefore
enter its text/vision AND image-latent reference paths, rather than supplying only
text descriptions. `ref_image_size=match` remains unchanged to avoid silently
increasing the already significant memory cost. Two references and longer text can
still consume more memory/time per job than one reference.

## Installation

Download the ZIP into Downloads and extract it outside your repository. From a
Terminal window in your existing repository, run its `install_without_python.rb`
with `--repo .`. The script verifies the before/after checksums, refuses conflicting
edits, and backs up changed files outside Git before writing. It performs no network
calls, commits, pushes, image builds, deployments, or deletions of your media.

Review and commit in GitHub Desktop, then manually use **Push origin**. Start a
NEW **Build H3 character replacement** GitHub Actions run. After success, deploy
its new image digest on RunPod. Keep the current `/workspace` volume, secrets,
Akatz-only environment, port configuration and blank start-command override.

The new two-reference nodes require that rebuild. A new JSON alone will not add
them to an old running image. This package does not include a live-Pod installer.
Do not layer an older pending package over the new source without resolving conflicts.

No environment changes are needed. Keep `H3_SWAP_VARIANTS=akatz`. The existing
manual FileBrowser service and its public port, if used, remain unchanged. This
update does not create a `/files/` route or modify authentication.

## First test

1. Finish or stop active generation. Upload the source video, face photo and body
   photo into `/workspace/input`. Keep their filenames distinct and do not edit
   the files during an active run.
2. Check the selected person's FULL edit mask with your existing mask-check
   workflow and the same source section. It must include head, torso, clothing,
   arms and legs as visible, not only a face mask. Do not select neighboring people.
3. Open workflow **04**. Load the video and both photos. Describe the ORIGINAL
   source performer, for example `person wearing the blue shirt`. Keep the larger
   defaults initially: Exhaustive, 16,384 output tokens, 32,768 context, 16 source
   samples, 124-frame analysis interval. The budget now reserves space for TWO
   reference images; the reserve remains an estimate, not an exact token count.
4. Review the evidence for the proper separation: face/hair from the close-up,
   physique and wardrobe from the body photo. Copy **AKATZ CHAIN-SAFE PROMPT**.
   The clip-specific prompt includes timestamps and is not the long-chain prompt.
5. Open workflow **09**. Select the same full source video, face photo and body
   photo in its file dropdowns. Refresh ComfyUI after uploading to refresh choices.
   Set the SAME `target_person` wording used for analysis. Paste the reviewed
   chain-safe draft into `additional_instructions`. That field can also stay blank
   for a comparison using the built-in role prompt without Ollama.
6. Use a new run name, `duration_seconds=10`, `window_frames=124`, `resolution=768`
   and **Review every chunk**. Queue ONCE. Open `/_swap/chain` on the existing
   authenticated HTTP-8188 service to approve/retry each candidate.

The saved workflow defaults to a 10-second test. This exercises at least one
continuation boundary, not only the first generation. Compare the same video,
seed and mask against your single-reference result. Evaluate face AND body/outfit
AND the two non-target people when present. Only then start a NEW 48-second run.
At 24 fps and the default window size, 48 seconds is 1,152 delivered frames across
10 jobs per selected performer. This does not replace all three source people at
once. Each performer still needs its own reference pair and separate pass.

If one job runs out of memory, try a NEW run with `window_frames=56` before reducing
resolution. The existing model still must fit, and the extra reference is not free.
This update does not claim a measured peak VRAM/RAM requirement.

## The deliberate prompt, not only two new inputs

The runner prepends and appends an explicit role contract to every new dual-reference
run. It requests whole-character replacement and equal completion of the face AND
body/outfit edits. It states how to resolve competing faces, clothing, poses and
backgrounds, follows the currently supplied source segment, and preserves non-target
people as they appear in the CURRENT input (including earlier replacement passes).

The contract is automatic even with blank additional instructions. The paired
analyzer returns the same contract plus long, attributed appearance evidence.
Pasting that exact draft is idempotent: it is not wrapped in duplicate role blocks.
A pasted draft for a different target is refused, not silently combined with another
person's mask. Use the same target wording or paste only the relevant appearance
paragraphs. Contradictory freeform instructions can still confuse a model; do not
paste an older single-reference prompt or ask to retain the source wardrobe.

There is no claim that this contract enforces regions or numerical reference weights.
The intended semantic roles are prompt guidance. The existing person mask remains
one full-performer edit mask, not separate face and body masks.

## What is preserved across jobs and resume

For new dual runs, `config.json` records both file paths and SHA256 hashes, their
ordered roles, a role-contract version and the complete effective prompt. An
`effective_prompt.txt` file is saved alongside it for inspection.

The controller snapshots the face and body files into the run's input staging
folder. Each first job, continuation and retry uses Picture 1 and Picture 2 in that
order. Staged image checksums are verified before each queued generation. Resume
also checks the original references. A changed body image cannot silently be mixed
into the middle of a run. Use a NEW run to change images, target, mask or settings.

The 18-frame generated history, matching raw-mask continuation, separate-job memory
cleanup, lossless section masters, review/retry behavior and final assembly are
unchanged. Accepted sections are not regenerated merely to add the second image.
A pre-existing single-reference run remains single-reference; Resume does not
convert it. The saved-run data version remains compatible with existing v2 records,
subject to the existing source/model/build/receipt checks.

The review page additionally lists the reference filenames and their assigned roles.
These labels prove configuration, not whether the model honored the roles visually.

## Output locations

Dual analysis drafts:

`/workspace/output/h3_character_swap/prompt_drafts_face_body/`

They include `_akatz_clip.txt`, `_akatz_chain.txt`, the full evidence/report and
source-sample contact sheet. The contact sheet is for review, not a replacement
for the two actual images supplied to generation. The separate single-image analyzer
continues saving into `prompt_drafts_long`.

Renders retain the current location:

`/workspace/output/h3_character_swap/disk_chain_v2/<run_name>/`

Use that run's `final.mp4` only after completion. Candidate or rejected masters are
not automatically accepted, and the automatic seam checks do not certify likeness,
body transfer, background preservation or seamless motion.

## Validation limits

See the package's VALIDATION.json and test logs for actual checks performed.
The tests use synthetic media and fixture Ollama/H3 replies. They test connections,
image ordering, full text, reference fingerprints, cache changes, job sequencing,
real FFmpeg I/O, resume and installer safety. They do not establish face/body quality.

No full Docker image build, live ComfyUI browser import, actual Ollama generation,
GPU H3 inference or peak-memory measurement is claimed from the authoring environment.
The existing image-build CPU startup check now requires both added nodes. Live
short-shot inspection is still required before a long or three-pass run.

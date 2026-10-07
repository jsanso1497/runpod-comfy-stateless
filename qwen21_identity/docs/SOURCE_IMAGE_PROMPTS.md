# Source-image prompts for ChatGPT

These are copy-ready instructions for separate ChatGPT image-generation chats. They create useful **candidate references**, not guaranteed reconstructions of hidden anatomy or an exact biometric identity. Prefer a suitable genuine photograph over an unnecessary generated replacement. Always retain your original images for comparison.

For each request, upload the relevant photographs together and make their order explicit. Label attachments in your message, especially when filenames are not shown reliably. Use photos of the intended same person; do not include the target scene in the neutral face/body reference requests. The optional angle-matching prompt intentionally uses a scene, but gives it only the pose/expression role.

Generate one image per request, not a contact sheet. Request the highest native output detail rather than claiming that typing "8K" creates true 8K source detail.

## 1. Audit and select references before generating

Upload the original face and body photos. This first prompt asks for an assessment, not an image.

```text
I am preparing reference images of the same person for a realistic image-editing workflow. I need one strong face/head reference and one strong full-body/wardrobe reference.

Review the attached images without generating or modifying anything yet. Refer to each image by its filename or upload number.

Choose the best original face reference based on sharpness, visible facial structure, limited distortion, natural texture, useful head angle and unobstructed hairline. Choose the best original body reference based on a complete visible figure, reliable perspective, visible hands and feet, and clothing that makes actual body proportions reasonably clear.

Tell me which original photos are already suitable without generation. Identify any missing information that would force a generated reference to guess, such as hidden legs, an obscured jaw, missing feet, heavy filtering or uncertain proportions under clothing. Do not claim to recover those details from an image that does not show them.

Recommend only the minimum reference preparation needed. Prioritize preserving the observed person over making an attractive or polished new photograph.
```

## 2. Main face/head reference: maximum identity preservation

Upload the clearest face photograph first. Add a few genuinely useful photos of the same person, preferably from a similar time period and appearance. Do not make a poor generated portrait the primary identity anchor.

```text
Create one photorealistic head-and-shoulders reference photograph of the exact same person shown in the uploaded images.

IMAGE ROLES:
Image 1 is the primary facial-identity anchor. Use its actual facial structure and proportions.
The remaining images are supporting evidence for features that are unclear in Image 1. They are not alternative faces to blend or average together. When references conflict because of expression, perspective, lighting or age, keep Image 1 as the identity anchor.

FRAMING AND POSE:
A single straight-on head-and-shoulders photograph. Head upright with no sideways tilt. Face level and centered, looking toward the lens. Relaxed neutral expression, lips naturally closed, no forced smile. Show the complete top of the head, the natural hairline, both sides of the head, the neck and upper shoulders. Do not crop the chin or top of the hair. Keep hair clear of the eyes and facial outline without changing the hairstyle, hair density or hairline. Use a plain light gray background and soft, even photographic lighting.

IDENTITY AND REALISM:
Preserve the person's observed facial geometry, eye shape and spacing, eyebrows, nose structure, lips, jaw, chin, ears where visible, natural asymmetry, age cues, skin tone and visible distinguishing features. Preserve realistic skin texture without inventing exaggerated pores or removing ordinary lines and marks. Keep recognizable grooming and appearance from the primary reference.

Do not beautify, slim the face, enlarge the eyes, sharpen the jaw, straighten asymmetry, change apparent age, add makeup, apply beauty smoothing, or turn this into an idealized lookalike. Do not make all the references agree by inventing an averaged face. Use supporting photos to clarify, not redesign.

OUTPUT:
One photograph of one person, not a collage, grid, turnaround, labeled sheet or side-by-side. Produce the highest native detail available. Prioritize faithfulness to the originals over stylization or cosmetic polish.
```

A front-facing original is preferable to inventing a frontal view from an extreme side view. A generated straight-on portrait should be treated as a candidate and checked, not automatically promoted to ground truth.

## 3. Main body reference: square, upright, complete figure

Upload the best genuine full-body photo as Image 1, your face reference as Image 2, then supporting real photos as needed. Choose an Image 1 with the wardrobe intended to transfer. This workflow's body adapter transfers clothing as well as physique.

```text
Create one photorealistic, full-body reference photograph of the same person shown in these images, for use as a physique and wardrobe reference in an image-editing workflow.

IMAGE ROLES:
Image 1 is the primary authority for body proportions, build, clothing and footwear.
Image 2 is the primary authority for facial identity, head shape, hairline and hairstyle. Do not use a close-up face image to infer body size.
Any remaining photographs are supporting views of this same person. Use them only to clarify visible anatomical or clothing details. Do not average conflicting perspectives into a new body.

COMPOSITION:
A square image with one complete standing person, front-facing and centered on a plain light gray seamless background. Keep the entire head, both hands and both feet in frame. The person should occupy most of the image height, with a small clear margin above the head and below the feet.

POSE AND CAMERA:
Stand naturally upright with balanced weight and relaxed shoulders. Arms relaxed at the sides with a small gap from the torso, hands naturally visible, legs comfortably aligned rather than crossed. Use a level camera around mid-torso height and neutral, non-wide-angle photographic perspective. Avoid foreshortening, selfie distortion, a low heroic angle or an overhead view. Do not stretch the figure to fit the square canvas.

PRESERVATION:
Retain the observed physique, shoulder width, torso length, waist and hip proportions, arm and leg proportions, natural body shape and realistic head-to-body scale. Keep Image 1's clothing and footwear, including their actual cut and fit, without using clothing folds or perspective distortion as evidence of different anatomy. Keep facial identity and hair consistent with Image 2.

Use soft, even light, realistic skin and fabric texture, and a subtle natural contact shadow. Do not slim, elongate, add muscle, improve posture into a different body, alter apparent age, sexualize the pose or replace the wardrobe with a generic outfit.

If the photographs do not show enough of the full body, hands, feet or clothing to make this reference without substantial guessing, explain which view is missing instead of inventing an idealized body.

OUTPUT:
One square photograph, one person, one view. No collage, turnaround sheet, text or labels. Preserve the actual person and wardrobe over making a more fashionable or conventionally attractive picture.
```

## 4. Conservative cleanup when the real body photograph is already good

This is the preferred alternative to generating new anatomy. Upload the genuine complete-body image first and keep other photos available only for checking ambiguous details.

```text
Use Image 1 as the photographic source. Prepare it as a clean body-reference image without redesigning the person.

Keep the person's original body pose, face, hair, anatomy, proportions, clothing and footwear unchanged. Remove the surrounding people and background objects, then place the existing subject against a plain light gray background with a believable subtle floor contact shadow. Fit the complete person into a square canvas by adding background space, never by stretching, slimming or lengthening the body. Keep the full head, hands and feet visible and retain natural image texture.

The other attached images are supporting references only. Do not replace clear details in Image 1 with an averaged or beautified version. Do not change the person's stance merely to make it more symmetrical. Do not synthesize missing body parts as though they were known. If the original is cropped or occluded, identify that limitation instead.

Return one square photographic reference. No text, collage, pose sheet or cosmetic retouching.
```

Even background replacement is a generative edit in ChatGPT and may alter details. When literal pixel preservation is required, cut out and pad the original photograph with a conventional image editor instead.

## 5. Optional head-angle reference matched to the destination scene

Use this only when there is no suitable genuine photo at the needed head angle and the neutral face reference is not working. It adds an additional opportunity for identity drift.

```text
Create one photorealistic head-and-shoulders reference image of the person in Image 1, shown at the head angle and expression required by the destination scene.

IMAGE ROLES:
Image 1 is the primary identity authority for the replacement person's face, head shape, hairline and hair.
Image 2 is another photograph of that same replacement person and may clarify identity details only.
Image 3 is the destination scene. Use it only for the target person's head rotation, tilt, chin angle, gaze direction and expression. Do not borrow the destination person's facial features, age, skin tone, head shape or hairstyle.

Match the target head orientation and expression from Image 3 while retaining the replacement person's observed facial geometry, natural asymmetry, age cues and distinguishing features from Images 1 and 2. Preserve a believable head-and-neck relationship. Do not copy Image 3's face or blend its identity into the replacement.

Use a plain light gray background, soft neutral light and realistic skin texture. Include the whole head and enough neck and shoulders to read the angle. No beauty smoothing, slimming, eye enlargement or exaggerated sharpening.

Output a single head-and-shoulders photograph, not a collage. Preserve identity first. If the supplied references do not support the requested angle reliably, state the limitation rather than presenting an invented view as an exact reconstruction.
```

## 6. Check a generated reference against the originals

Upload the new candidate as Image 1 and the original source photographs after it. This is an assessment request, not a new generation.

```text
Image 1 is a generated reference candidate. The remaining images are the original photographs it was intended to preserve. These images were supplied as depictions of the same intended person.

Compare the visible features and proportions without generating another image yet. Check face width and length, eye shape and spacing, eyebrows, nose, lips, jaw and chin, hairline, natural asymmetry, apparent age, visible skin marks and head-to-body scale. For a body reference, also check shoulders, torso length, waist/hips, arms, legs, hands, feet and the cut and fit of the clothing.

Separate clear changes from differences that could reasonably be caused by pose, camera perspective, expression or lighting. Do not turn this into a claim of biometric identity verification, and do not invent a numerical accuracy score.

Tell me whether this candidate is a useful reference or whether I should use the best original instead. Name the few most consequential differences to correct. Do not describe cosmetic beautification as an improvement when it changes the observed person.
```

## Prompt fragments for the actual Qwen workflow

### Keep physique but request the scene's clothing

Replace the default body prompt's clothing-transfer clause with:

```text
Use image2 for the replacement person's physique and body proportions, but retain the clothing design, colors and overall outfit from image1. Refit that existing outfit naturally to the replacement body without importing image2's wardrobe.
```

This is a request, not a guaranteed adapter capability. Judge the result before relying on it.

### Prevent perspective-based limb mismatch

```text
Preserve the target pose and contact points while projecting the reference physique into the target camera perspective. Do not copy the reference camera angle, apparent limb lengths or apparent body scale.
```

### Prevent the face pass from changing pose or expression

```text
Use the face reference for identity only. Retain the head rotation, gaze direction and expression of the accepted scene crop, while rendering those features as the replacement person.
```

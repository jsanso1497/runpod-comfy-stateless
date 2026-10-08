# Editable prompts shipped with the workflows

These are prefilled in each Native Encode node. They are proposals to test, not guarantees of identity preservation. Masks are applied by compositing, not read by the image model.

## 00_START_HERE_Scene_Face_Body

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Preserve the scene person's clothing, fitted naturally to the replacement physique. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 01_Person_Multi_Reference

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image4> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Preserve the scene person's clothing, fitted naturally to the replacement physique. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. <image3> supports the same face identity from another angle; <image5> supports the same body anatomy. The primary face remains <image2>.
```

## 02_Person_Keep_Scene_Wardrobe

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Preserve the scene person's clothing, fitted naturally to the replacement physique. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. Do not import any garment from the reference photos; retain the exact design, fabric and color worn in the scene.
```

## 03_Person_Transfer_Body_Wardrobe

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Use the clothing and footwear shown in <image3>, with their actual design and fit adapted to the scene pose. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 04_Person_Separate_Wardrobe_Reference

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Take the outfit only from <image4>, fit it to the referenced physique, and ignore the garment model's identity. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 05_Body_Only_Hard_Protect_Head

```text
Edit only the centered person's body in <image1> using the physique and proportions evidenced by <image2>. Preserve the existing head and face. Project the referenced physique into the current pose, foreshortening and contact points; do not copy apparent limb lengths or camera scale. Preserve the existing outfit and adapt its fit naturally. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 06_Body_And_Wardrobe_Hard_Protect_Head

```text
Edit only the centered person's body in <image1> using the physique and proportions evidenced by <image2>. Preserve the existing head and face. Project the referenced physique into the current pose, foreshortening and contact points; do not copy apparent limb lengths or camera scale. Transfer the outfit from <image2> and adapt it to the current pose. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 07_Head_One_Reference

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 08_Head_Three_Identity_References

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. <image3> and <image4> only clarify the SAME person's appearance from additional views. Do not average identities or copy reference pose.
```

## 09_Face_Only_Keep_Hair

```text
Edit the centered person's face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the existing hair, hairline, ears and outer head silhouette unchanged. Change only the interior face. <image3> only clarifies the same identity from an additional view. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 10_Head_Hair_And_Hairline

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. Use <image3> only as supporting evidence for the same person's hair, hairline and side profile. Include natural hair volume without increasing the head size.
```

## 11_Profile_Angle_Matched_Identity

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. <image3> shows the same person at a useful side angle. Use it to resolve projection and occlusion, while <image2> remains the primary identity anchor. Match <image1>'s exact head rotation.
```

## 12_Rescue_Likeness_From_Originals

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. The face currently in <image1> has drifted and is NOT an identity reference. Correct that identity using <image2>; <image3> only clarifies the same real person. Keep the existing head angle and expression, not the incorrect facial geometry.
```

## 13_Expression_Image2_Only

```text
Image1 is the sole identity and appearance anchor. Edit only the centered person's facial expression in <image1>. Use <image2> solely for expression intensity and facial muscle movement, adapted to <image1>'s own anatomy. Preserve the base person's facial structure, natural asymmetry, skin, age, hair, head position, gaze direction, body and clothing. Do not copy <image2>'s physical features, head angle, lighting or perspective. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 14_Identity_And_Expression_Separate_Refs

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. Use <image3> ONLY for expression and facial muscle movement, adapted to <image2>'s identity. Do not take anatomy, skin, hair, head angle or likeness from the expression reference.
```

## 15_Expression_Text_Only

```text
Edit only the centered person's facial expression in <image1>: a subtle natural closed-mouth smile. Preserve the exact identity, face structure, asymmetry, skin texture, apparent age, head angle, gaze direction, hair, body, pose, lighting, clothing and background. Adapt only the facial muscles required for the smile. Do not beautify.
```

## 16_Local_Repair_No_Reference

```text
Use <image1> as the base photograph. Repair the visible local blending defect on the centered subject. Preserve the exact established identity, expression, head angle, pose, clothing and lighting. Do not redesign already-correct anatomy. Smooth only the inconsistent transition, retain normal skin texture and correct detail.
```

## 17_Local_Repair_With_Anatomy_Reference

```text
Repair the local anatomical or blending error on the centered subject in <image1>. Use <image2> only as evidence for the relevant anatomy of this person. Preserve the established identity, clothing, pose and scene. Project anatomy into the base perspective rather than copying the reference camera angle. Keep already-correct details unchanged.
```

## 18_Hand_And_Wrist_Repair

```text
Correct the centered hand and its connection to the wrist in <image1>. Preserve its gesture, contact points, arm position and foreshortening. Use <image2> only for this person's hand proportions and observed details, not its gesture or scale. Preserve fingers hidden by objects rather than adding visible fingers through occluders. Retain skin texture, scene lighting and all correct surrounding details.
```

## 19_Limb_Perspective_Repair

```text
Correct the centered limb's anatomy and connection to the body in <image1>. Use <image2> only for the person's true limb proportions and thickness. Project those proportions into <image1>'s existing pose and foreshortening. Do not copy the reference's apparent length, perspective or camera scale. Preserve identity, contact points, clothing, lighting and correct surrounding anatomy.
```

## 20_Neck_Head_Seam_Repair

```text
Repair only the transition between neck, jaw and shoulders in <image1> so they form one coherent person. Preserve the existing face identity, expression, head rotation and hairstyle. Use <image2> only for neck and shoulder proportion evidence. Match the scene light and skin transitions without smoothing away normal texture. Do not replace the head again.
```

## 21_Skin_Texture_Light_Touch

```text
Reduce the artificial waxy texture on the centered person's face in <image1>. Use <image2> only for authentic texture and observed skin details of this same person. Preserve exact face shape, expression, age, asymmetry, skin tone, existing marks, light and shadow. Do not enlarge pores, invent freckles, change makeup, sharpen edges aggressively or beautify. Keep already-natural texture unchanged.
```

## 22_Wardrobe_Only_Keep_Person

```text
Keep the exact person, face, hair, physique, pose and skin from <image1>. Replace only the visible outfit using the garment from <image2>. Ignore the clothing model's body and identity. Adapt garment fit, folds, occlusions and shadows to the existing body and scene perspective. Do not change body proportions. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 23_Sequential_Body_Then_Head_Native

```text
Edit only the centered person's body in <image1> using the physique and proportions evidenced by <image2>. Preserve the existing head and face. Project the referenced physique into the current pose, foreshortening and contact points; do not copy apparent limb lengths or camera scale. Preserve the existing outfit and adapt its fit naturally. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. <image3> only clarifies the same person's identity from another view; do not blend faces.
```

## 24_Two_Heads_Sequential

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 25_Two_Full_People_Sequential

```text
Replace the centered person in <image1>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Preserve the scene person's clothing, fitted naturally to the replacement physique. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 26_Two_People_One_Pass_EXPERIMENTAL

```text
Edit the two people in <image1>. The person on the viewer's left gets facial identity ONLY from <image2> and physique ONLY from <image3>. The person on the viewer's right gets facial identity ONLY from <image4> and physique ONLY from <image5>. Do not mix reference groups. Preserve their individual poses, positions, clothing, head angles, expression and contact points. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 27_Described_Scene_Face_Plus_Body

```text
Create one realistic photograph using <image1> only as a blank canvas for size and aspect ratio. <image2> is the sole face and hair identity anchor; <image3> controls only physique and body proportions. Show that exact person standing on a quiet tree-lined sidewalk in soft afternoon light, relaxed natural posture, wearing a plain shirt, fitted trousers and low-profile shoes. Full body visible, eye-level natural photographic perspective. Do not copy the reference backgrounds, poses or camera angles. Preserve natural asymmetry, apparent age and realistic skin and fabric.
```

## 28_Source_Prep_Fitted_Clothing_CANDIDATE

```text
Create one photorealistic full-body reference photograph on the canvas in <image1>. Use <image2> only for the exact person's face and hair, and <image3> only for observed physique. One person standing front-facing, complete head, hands and feet visible, arms slightly away from torso, neutral expression, plain light-gray background and soft even lighting. Plain fitted sleeveless top and fitted mid-thigh athletic shorts without compression, padding or body reshaping. Keep bare feet only if supported by real source evidence; otherwise retain simple footwear. Do not slim, add muscle, exaggerate curves or guess concealed anatomy with false certainty. Output one view, not a collage.
```

## 30_Head_High_Detail_4MP_50Steps

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization. <image3> provides supporting genuine angle evidence only.
```

## 31_Head_Lean_1MP_BF16

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 32_Head_Three_Seed_Comparison

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 90_AB_Head_Native_vs_BFS

```text
Edit the centered person's head and face in <image1> to be the exact person in <image2>. <image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age. Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```

## 91_AB_Body_Native_vs_BFS

```text
Edit only the centered person's body in <image1> using the physique and proportions evidenced by <image2>. Preserve the existing head and face. Project the referenced physique into the current pose, foreshortening and contact points; do not copy apparent limb lengths or camera scale. Preserve the existing outfit and adapt its fit naturally. Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization.
```


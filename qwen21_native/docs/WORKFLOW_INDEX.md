# Qwen 2.1 Native Suite: workflow index

38 workflows. Native profile installs 34; upscale adds 2; BFS opt-in adds 2.

| ID | Workflow | Purpose | Requires |
|---|---|---|---|
| 00 | `00_START_HERE_Scene_Face_Body` | One masked native replacement with separate facial identity and physique. | identity |
| 01 | `01_Person_Multi_Reference` | Two face and two body references with explicit ownership. | identity |
| 02 | `02_Person_Keep_Scene_Wardrobe` | Identity and physique replacement without importing reference clothing. | identity |
| 03 | `03_Person_Transfer_Body_Wardrobe` | Transfer the body reference's clothing with separately anchored face. | identity |
| 04 | `04_Person_Separate_Wardrobe_Reference` | Separate scene, face, physique and wardrobe sources. | identity |
| 05 | `05_Body_Only_Hard_Protect_Head` | Body edit with an explicit pixel-locked head protection mask. | identity |
| 06 | `06_Body_And_Wardrobe_Hard_Protect_Head` | Body plus reference outfit with the original head pixel-locked. | identity |
| 07 | `07_Head_One_Reference` | Minimal native head replacement from one genuine face anchor. | identity |
| 08 | `08_Head_Three_Identity_References` | Head replacement with one primary face and two supporting views. | identity |
| 09 | `09_Face_Only_Keep_Hair` | Face edit while preserving the existing hairstyle and head outline. | identity |
| 10 | `10_Head_Hair_And_Hairline` | Complete head, hair and hairline replacement from identity references. | identity |
| 11 | `11_Profile_Angle_Matched_Identity` | Primary identity plus an actual angle-matched profile reference. | identity |
| 12 | `12_Rescue_Likeness_From_Originals` | Replace a drifted face using original photographs, not the drifted render. | identity |
| 13 | `13_Expression_Image2_Only` | Transfer expression while taking all likeness from the base image. | identity |
| 14 | `14_Identity_And_Expression_Separate_Refs` | Face identity from image2, expression only from image3. | identity |
| 15 | `15_Expression_Text_Only` | Expression-only editing without additional reference images. | identity |
| 16 | `16_Local_Repair_No_Reference` | Small cleanup using only the accepted image. | identity |
| 17 | `17_Local_Repair_With_Anatomy_Reference` | No-LoRA counterpart to the previous local repair workflow. | identity |
| 18 | `18_Hand_And_Wrist_Repair` | Local hand anatomy and wrist connection correction. | identity |
| 19 | `19_Limb_Perspective_Repair` | Correct arm or leg proportions in the existing camera perspective. | identity |
| 20 | `20_Neck_Head_Seam_Repair` | Repair the neck transition without intentionally replacing the face. | identity |
| 21 | `21_Skin_Texture_Light_Touch` | Conservative texture repair from an original close-up. | identity |
| 22 | `22_Wardrobe_Only_Keep_Person` | Change clothing while retaining the existing person and pose. | identity |
| 23 | `23_Sequential_Body_Then_Head_Native` | Two native passes with hard head protection during body editing. | identity |
| 24 | `24_Two_Heads_Sequential` | Two-person replacement with explicit identity ownership. | identity |
| 25 | `25_Two_Full_People_Sequential` | Two-person replacement with explicit identity ownership. | identity |
| 26 | `26_Two_People_One_Pass_EXPERIMENTAL` | Two-person replacement with explicit identity ownership. | identity |
| 27 | `27_Described_Scene_Face_Plus_Body` | Generate a new scene from separately assigned face and body references. | identity |
| 28 | `28_Source_Prep_Fitted_Clothing_CANDIDATE` | Optional synthetic body-reference normalization, never ground truth. | identity |
| 29 | `29_Reference_Crop_Export_NO_GENERATION` | Extract genuine reference detail without a generative model. | identity |
| 30 | `30_Head_High_Detail_4MP_50Steps` | Higher-resolution head-crop comparison, not an automatic quality upgrade. | identity |
| 31 | `31_Head_Lean_1MP_BF16` | Lower crop-memory variant retaining the same BF16 weights. | identity |
| 32 | `32_Head_Three_Seed_Comparison` | Three seeds with every other setting held constant. | identity |
| 33 | `33_Final_Lanczos_No_Model` | Non-generative enlargement without any added model. | identity |
| 34 | `34_SeedVR2_Full_Frame_7B_FP16` | Optional full-frame generative upscaling. | upscale |
| 35 | `35_SeedVR2_Local_Protected_Finish` | Optional generative enhancement of a local patch. | upscale |
| 90 | `90_AB_Head_Native_vs_BFS` | Matched native/adapter comparison; not an assertion of superior likeness. | bfs_optional |
| 91 | `91_AB_Body_Native_vs_BFS` | Matched native/adapter comparison; not an assertion of superior likeness. | bfs_optional |

## Input map for each workflow

The image number in each prompt is the encoder slot, not the order of all Load Image nodes on the canvas. Extra scene loaders provide masks only. They are not extra identity references.

### 00_START_HERE_Scene_Face_Body
One masked native replacement with separate facial identity and physique.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> BODY geometry only**: `body_front.png`

### 01_Person_Multi_Reference
Two face and two body references with explicit ownership.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`
- **<image4> BODY geometry only**: `body_front.png`
- **<image5> BODY supporting view**: `body_side.png`

### 02_Person_Keep_Scene_Wardrobe
Identity and physique replacement without importing reference clothing.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> BODY geometry only**: `body_front.png`

### 03_Person_Transfer_Body_Wardrobe
Transfer the body reference's clothing with separately anchored face.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> BODY and WARDROBE source**: `body_wardrobe.png`

### 04_Person_Separate_Wardrobe_Reference
Separate scene, face, physique and wardrobe sources.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> BODY geometry only**: `body_front.png`
- **<image4> WARDROBE only**: `garment.png`

### 05_Body_Only_Hard_Protect_Head
Body edit with an explicit pixel-locked head protection mask.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> BODY geometry only**: `body_front.png`
- **SAME SCENE | paint HEAD to PROTECT, not edit**: `scene.png`

The second scene loader MUST use the identical original and a painted head/face protection mask. That region is restored exactly after stitching.

### 06_Body_And_Wardrobe_Hard_Protect_Head
Body plus reference outfit with the original head pixel-locked.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> BODY plus desired clothes**: `body_wardrobe.png`
- **SAME SCENE | paint HEAD to PROTECT, not edit**: `scene.png`

### 07_Head_One_Reference
Minimal native head replacement from one genuine face anchor.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`

### 08_Head_Three_Identity_References
Head replacement with one primary face and two supporting views.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`
- **<image4> SAME PERSON second supporting view**: `face_support.png`

### 09_Face_Only_Keep_Hair
Face edit while preserving the existing hairstyle and head outline.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`

Paint the face only, leaving hair outside the feather. Image3 is supporting identity evidence only.

### 10_Head_Hair_And_Hairline
Complete head, hair and hairline replacement from identity references.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`

Mask the complete old and new hair silhouette, ears and required neck transition.

### 11_Profile_Angle_Matched_Identity
Primary identity plus an actual angle-matched profile reference.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> GENUINE angle-matched profile**: `face_profile.png`

### 12_Rescue_Likeness_From_Originals
Replace a drifted face using original photographs, not the drifted render.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`

### 13_Expression_Image2_Only
Transfer expression while taking all likeness from the base image.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> EXPRESSION only, same subject**: `expression.png`

### 14_Identity_And_Expression_Separate_Refs
Face identity from image2, expression only from image3.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> EXPRESSION only, same subject**: `expression.png`

### 15_Expression_Text_Only
Expression-only editing without additional reference images.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`

Replace the smile description with your desired expression.

### 16_Local_Repair_No_Reference
Small cleanup using only the accepted image.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`

Replace the generic defect description with exactly what needs repair.

### 17_Local_Repair_With_Anatomy_Reference
No-LoRA counterpart to the previous local repair workflow.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> RELEVANT genuine anatomical detail**: `anatomy_reference.png`

### 18_Hand_And_Wrist_Repair
Local hand anatomy and wrist connection correction.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> GENUINE hand reference**: `hand_reference.png`

### 19_Limb_Perspective_Repair
Correct arm or leg proportions in the existing camera perspective.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> GENUINE limb/body evidence**: `limb_reference.png`

### 20_Neck_Head_Seam_Repair
Repair the neck transition without intentionally replacing the face.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> GENUINE head and shoulders**: `neck_reference.png`

Mask neck and seam only; keep the accepted face outside the feather.

### 21_Skin_Texture_Light_Touch
Conservative texture repair from an original close-up.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> GENUINE unretouched skin/face reference**: `skin_reference.png`

Generated pores and microdetail are not recovered evidence; compare against the original.

### 22_Wardrobe_Only_Keep_Person
Change clothing while retaining the existing person and pose.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> WARDROBE only**: `garment.png`

### 23_Sequential_Body_Then_Head_Native
Two native passes with hard head protection during body editing.

- **ORIGINAL scene | BODY edit mask**: `scene.png`
- **BODY reference only**: `body_front.png`
- **SAME original scene | HEAD edit and protection mask**: `scene.png`
- **PRIMARY real face identity**: `face_primary.png`
- **Supporting genuine face angle**: `face_angle.png`

### 24_Two_Heads_Sequential
Two-person replacement with explicit identity ownership.

- **ORIGINAL scene | PERSON A mask**: `scene.png`
- **SAME original scene | PERSON B mask**: `scene.png`
- **PERSON A primary face**: `person_a_face.png`
- **PERSON B primary face**: `person_b_face.png`

Change left/right instructions for one-pass mode. Serial edits are the recommended baseline.

### 25_Two_Full_People_Sequential
Two-person replacement with explicit identity ownership.

- **ORIGINAL scene | PERSON A mask**: `scene.png`
- **SAME original scene | PERSON B mask**: `scene.png`
- **PERSON A primary face**: `person_a_face.png`
- **PERSON B primary face**: `person_b_face.png`
- **PERSON A body**: `person_a_body.png`
- **PERSON B body**: `person_b_body.png`

Change left/right instructions for one-pass mode. Serial edits are the recommended baseline.

### 26_Two_People_One_Pass_EXPERIMENTAL
Two-person replacement with explicit identity ownership.

- **ORIGINAL scene | PERSON A mask**: `scene.png`
- **SAME original scene | PERSON B mask**: `scene.png`
- **PERSON A primary face**: `person_a_face.png`
- **PERSON B primary face**: `person_b_face.png`
- **PERSON A body**: `person_a_body.png`
- **PERSON B body**: `person_b_body.png`

Change left/right instructions for one-pass mode. Serial edits are the recommended baseline.

### 27_Described_Scene_Face_Plus_Body
Generate a new scene from separately assigned face and body references.

- **<image2> PRIMARY genuine face**: `face_primary.png`
- **<image3> Genuine body evidence**: `body_front.png`

### 28_Source_Prep_Fitted_Clothing_CANDIDATE
Optional synthetic body-reference normalization, never ground truth.

- **<image2> PRIMARY genuine face**: `face_primary.png`
- **<image3> Genuine body evidence**: `body_front.png`

### 29_Reference_Crop_Export_NO_GENERATION
Extract genuine reference detail without a generative model.

- **GENUINE photo | mask useful reference region**: `reference_original.png`

### 30_Head_High_Detail_4MP_50Steps
Higher-resolution head-crop comparison, not an automatic quality upgrade.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`
- **<image3> SAME PERSON supporting angle**: `face_angle.png`

More memory and time. Compare likeness with 07/08 before preferring this result.

### 31_Head_Lean_1MP_BF16
Lower crop-memory variant retaining the same BF16 weights.

- **<image1> BASE SCENE | paint edit mask**: `scene.png`
- **<image2> PRIMARY FACE identity**: `face_primary.png`

### 32_Head_Three_Seed_Comparison
Three seeds with every other setting held constant.

- **BASE SCENE | paint head mask**: `scene.png`
- **PRIMARY genuine face reference**: `face_primary.png`

### 33_Final_Lanczos_No_Model
Non-generative enlargement without any added model.

- **ACCEPTED native image**: `qwen21_native_handoff/head_latest.png`

### 34_SeedVR2_Full_Frame_7B_FP16
Optional full-frame generative upscaling.

- **ACCEPTED native image**: `qwen21_native_handoff/head_latest.png`

### 35_SeedVR2_Local_Protected_Finish
Optional generative enhancement of a local patch.

- **ACCEPTED native image | mask enhancement region**: `qwen21_native_handoff/head_latest.png`

### 90_AB_Head_Native_vs_BFS
Matched native/adapter comparison; not an assertion of superior likeness.

- **BASE SCENE | paint head mask**: `scene.png`
- **PRIMARY genuine face reference**: `face_primary.png`

### 91_AB_Body_Native_vs_BFS
Matched native/adapter comparison; not an assertion of superior likeness.

- **BASE SCENE | paint body mask**: `scene.png`
- **PRIMARY body reference**: `body_front.png`


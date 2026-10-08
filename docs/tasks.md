# Task catalog

Model precision is not another task profile. Each ID below points to its compatible graphs.

## Qwen

| ID | Task | Description | Workflows |
|---|---|---|---:|
| SQ | Standard HQ: Qwen reference editing | Clean BF16 reference-editing baseline. Native sampling and conditioning, without optional tools. | [Standard HQ](../workflows/qwen/00_Standard_HQ/Qwen_Standard_HQ_Reference_Edit.json) |
| D01 | Compare three seeds for a head edit | Hold other settings constant. | [32 Head Three Seed Comparison](../workflows/qwen/D01_Compare_three_seeds_for_a_head_edit/32_Head_Three_Seed_Comparison.json) |
| Q01 | General masked photo editing | Manual or SAM selection, multiple optional references, crop/edit/stitch. BF16 only; optional SAM requires its separate model. | [Qwen21 Photo 2K SAM3 BF16](../workflows/qwen/Q01_General_masked_photo_editing/Qwen21_Photo_2K_SAM3_BF16.json) |
| Q02 | Person replacement from separate face/body references | Includes the simple scene + face + body graph and the multi-view graph with two face and two body references. | [00 START HERE Scene Face Body](../workflows/qwen/Q02_Person_replacement_from_separate_face_body_references/00_START_HERE_Scene_Face_Body.json)<br>[01 Person Multi Reference](../workflows/qwen/Q02_Person_replacement_from_separate_face_body_references/01_Person_Multi_Reference.json) |
| Q03 | Replace a person, keep the scene outfit | Transfer identity and physique without importing reference clothing. | [02 Person Keep Scene Wardrobe](../workflows/qwen/Q03_Replace_a_person_keep_the_scene_outfit/02_Person_Keep_Scene_Wardrobe.json) |
| Q04 | Replace a person and transfer the body reference outfit | Separate face identity plus body/wardrobe source. | [03 Person Transfer Body Wardrobe](../workflows/qwen/Q04_Replace_a_person_and_transfer_the_body_reference_outfit/03_Person_Transfer_Body_Wardrobe.json) |
| Q05 | Use an independent wardrobe reference | Separate scene, face, body, and clothing references. | [04 Person Separate Wardrobe Reference](../workflows/qwen/Q05_Use_an_independent_wardrobe_reference/04_Person_Separate_Wardrobe_Reference.json) |
| Q06 | Body-only edit with the original head protected | Explicit protection mask restores original head pixels. | [05 Body Only Hard Protect Head](../workflows/qwen/Q06_Body_only_edit_with_the_original_head_protected/05_Body_Only_Hard_Protect_Head.json) |
| Q07 | Body and clothing edit with the original head protected | Body/outfit replacement with hard head protection. | [06 Body And Wardrobe Hard Protect Head](../workflows/qwen/Q07_Body_and_clothing_edit_with_the_original_head_protected/06_Body_And_Wardrobe_Hard_Protect_Head.json) |
| Q08 | Head replacement from identity photographs | One or three identity references. Includes the source high-detail 4 MP/50-step preset. Memory-saving preset removed. | [07 Head One Reference](../workflows/qwen/Q08_Head_replacement_from_identity_photographs/07_Head_One_Reference.json)<br>[08 Head Three Identity References](../workflows/qwen/Q08_Head_replacement_from_identity_photographs/08_Head_Three_Identity_References.json)<br>[30 Head High Detail 4MP 50Steps](../workflows/qwen/Q08_Head_replacement_from_identity_photographs/30_Head_High_Detail_4MP_50Steps.json) |
| Q09 | Face-only replacement, retain hair | Preserve existing hairstyle and head outline. | [09 Face Only Keep Hair](../workflows/qwen/Q09_Face_only_replacement_retain_hair/09_Face_Only_Keep_Hair.json) |
| Q10 | Replace the full head, hair, and hairline | Broader head replacement from identity references. | [10 Head Hair And Hairline](../workflows/qwen/Q10_Replace_the_full_head_hair_and_hairline/10_Head_Hair_And_Hairline.json) |
| Q11 | Angle-matched/profile identity replacement | Use a real supporting reference matching the target angle. | [11 Profile Angle Matched Identity](../workflows/qwen/Q11_Angle_matched_profile_identity_replacement/11_Profile_Angle_Matched_Identity.json) |
| Q12 | Recover likeness from original references | Repair a drifted generated face using original photographs. | [12 Rescue Likeness From Originals](../workflows/qwen/Q12_Recover_likeness_from_original_references/12_Rescue_Likeness_From_Originals.json) |
| Q13 | Transfer an expression from a photograph | Base image supplies identity; reference supplies expression. | [13 Expression Image2 Only](../workflows/qwen/Q13_Transfer_an_expression_from_a_photograph/13_Expression_Image2_Only.json) |
| Q14 | Separate identity and expression references | Identity and expression come from independently assigned sources. | [14 Identity And Expression Separate Refs](../workflows/qwen/Q14_Separate_identity_and_expression_references/14_Identity_And_Expression_Separate_Refs.json) |
| Q15 | Change expression using text only | No extra reference required. | [15 Expression Text Only](../workflows/qwen/Q15_Change_expression_using_text_only/15_Expression_Text_Only.json) |
| Q16 | Local cleanup without a reference | Small corrections using the accepted image. | [16 Local Repair No Reference](../workflows/qwen/Q16_Local_cleanup_without_a_reference/16_Local_Repair_No_Reference.json) |
| Q17 | Local anatomy repair with a reference | Groups native anatomy-reference repair and the earlier no-LoRA local repair graph. | [17 Local Repair With Anatomy Reference](../workflows/qwen/Q17_Local_anatomy_repair_with_a_reference/17_Local_Repair_With_Anatomy_Reference.json)<br>[04 Local Repair No LoRA](../workflows/qwen/Q17_Local_anatomy_repair_with_a_reference/04_Local_Repair_No_LoRA.json) |
| Q18 | Hand and wrist repair | Correct local hand anatomy and wrist connection. | [18 Hand And Wrist Repair](../workflows/qwen/Q18_Hand_and_wrist_repair/18_Hand_And_Wrist_Repair.json) |
| Q19 | Arm/leg perspective and proportion repair | Fit limb geometry to the target camera perspective. | [19 Limb Perspective Repair](../workflows/qwen/Q19_Arm_leg_perspective_and_proportion_repair/19_Limb_Perspective_Repair.json) |
| Q20 | Neck/head seam repair | Repair the transition without intentionally replacing the face. | [20 Neck Head Seam Repair](../workflows/qwen/Q20_Neck_head_seam_repair/20_Neck_Head_Seam_Repair.json) |
| Q21 | Light-touch skin texture repair | Conservative correction guided by an original close-up. | [21 Skin Texture Light Touch](../workflows/qwen/Q21_Light_touch_skin_texture_repair/21_Skin_Texture_Light_Touch.json) |
| Q22 | Clothing-only edit | Keep the existing person and pose. | [22 Wardrobe Only Keep Person](../workflows/qwen/Q22_Clothing_only_edit/22_Wardrobe_Only_Keep_Person.json) |
| Q23 | Sequential body-then-head replacement | Two native passes, with head protection during the body pass. | [23 Sequential Body Then Head Native](../workflows/qwen/Q23_Sequential_body_then_head_replacement/23_Sequential_Body_Then_Head_Native.json) |
| Q24 | Replace two heads sequentially | Independent identity assignment for each person. | [24 Two Heads Sequential](../workflows/qwen/Q24_Replace_two_heads_sequentially/24_Two_Heads_Sequential.json) |
| Q25 | Replace two full people sequentially | Independent person replacement passes. | [25 Two Full People Sequential](../workflows/qwen/Q25_Replace_two_full_people_sequentially/25_Two_Full_People_Sequential.json) |
| Q26 | Replace two people in a single pass | Explicitly experimental in the source. | [26 Two People One Pass EXPERIMENTAL](../workflows/qwen/Q26_Replace_two_people_in_a_single_pass/26_Two_People_One_Pass_EXPERIMENTAL.json) |
| Q27 | Create a described scene from face/body references | Generate a new scene instead of editing a supplied scene. | [27 Described Scene Face Plus Body](../workflows/qwen/Q27_Create_a_described_scene_from_face_body_references/27_Described_Scene_Face_Plus_Body.json) |
| Q28 | Prepare a synthetic body/reference image | Groups the two-reference source-prep and fitted-clothing candidate graphs. Generated approximation, not recovered ground truth. | [28 Source Prep Fitted Clothing CANDIDATE](../workflows/qwen/Q28_Prepare_a_synthetic_body_reference_image/28_Source_Prep_Fitted_Clothing_CANDIDATE.json)<br>[00 Source Prep 2 References](../workflows/qwen/Q28_Prepare_a_synthetic_body_reference_image/00_Source_Prep_2_References.json) |

## Flux

| ID | Task | Description | Workflows |
|---|---|---|---:|
| SF | Standard HQ: FLUX generation | Native FLUX.2 klein BF16 text-to-image baseline with its four-step distilled schedule. | [Standard HQ](../workflows/flux/00_Standard_HQ/FLUX_Standard_HQ.json) |
| F01 | Reference-guided masked photo edit | Original photo + one replacement reference; crop/edit/stitch. Quality Fit 2048 and Native 3072 are presets of this task. | [FLUX Photo 01 Quality Fit 2048 v1 0](../workflows/flux/F01_Reference_guided_masked_photo_edit/FLUX_Photo_01_Quality_Fit_2048_v1_0.json)<br>[FLUX Photo 02 Native 3072 v1 0](../workflows/flux/F01_Reference_guided_masked_photo_edit/FLUX_Photo_02_Native_3072_v1_0.json) |

## H3

| ID | Task | Description | Workflows |
|---|---|---|---:|
| SH | Standard HQ: H3 reference-to-video/audio | Full BF16 manual generation with native video/audio decoding and saving. | [Standard HQ](../workflows/h3/00_Standard_HQ/H3_Standard_HQ.json) |
| H01 | Manual standard reference-to-video/audio | Groups the general BF16 reference graph and the manual portrait Ref2VA graph. Clean HQ baseline remains required while H3 is retained. | [20 H3 Reference BF16](../workflows/h3/H01_Manual_standard_reference_to_video_audio/20_H3_Reference_BF16.json)<br>[H3 Ref2VA Standard](../workflows/h3/H01_Manual_standard_reference_to_video_audio/H3_Ref2VA_Standard.json) |
| H02 | Two-reference video with Ollama prompting | Reviewed prompt or integrated local prompt generation. | [21 H3 Ollama Two Refs BF16](../workflows/h3/H02_Two_reference_video_with_Ollama_prompting/21_H3_Ollama_Two_Refs_BF16.json) |
| H03 | Two-subject video with newly created RefMods | Build reusable encoded visual-reference packages from the source images, then generate. | [22 H3 RefMod Ollama Quality](../workflows/h3/H03_Two_subject_video_with_newly_created_RefMods/22_H3_RefMod_Ollama_Quality.json) |
| H04 | Video from previously saved RefMods | Reuse saved reference packages. | [23 H3 Saved RefMods Quality](../workflows/h3/H04_Video_from_previously_saved_RefMods/23_H3_Saved_RefMods_Quality.json) |
| H05 | One-person video from multiple photographs | Dedicated face/body and alternate-view reference handling. | [24 H3 One Person Multi Ref Quality](../workflows/h3/H05_One_person_video_from_multiple_photographs/24_H3_One_Person_Multi_Ref_Quality.json) |
| H06 | One-person video from a saved person bundle | Reuse a saved multi-view reference package for one subject. | [25 H3 One Person Saved RefMod Quality](../workflows/h3/H06_One_person_video_from_a_saved_person_bundle/25_H3_One_Person_Saved_RefMod_Quality.json) |
| H07 | User-directed video with labeled reference roles | User assigns each reference purpose and directs the action; Ollama formats the prompt. | [26 H3 User Directed Ollama](../workflows/h3/H07_User_directed_video_with_labeled_reference_roles/26_H3_User_Directed_Ollama.json) |
| H08 | Automatic role-aware portrait video | Separate face/body/pose roles; draft review; exact portrait delivery and matching exported last frame. | [H3 Portrait Auto](../workflows/h3/H08_Automatic_role_aware_portrait_video/H3_Portrait_Auto.json) |
| H09 | Single still image using the H3 video model | Generate a short frame packet with the Full BF16 H3 stack and save one frame. HQ port needs GPU validation; no second image model. | [H3 Portrait Image HQ](../workflows/h3/H09_Single_still_image_using_the_H3_video_model/H3_Portrait_Image_HQ.json) |
| H10 | Subject replacement guided by a reference video | Subject images plus motion/scene video with local prompt analysis. Generates a new video, not an exact original-footage patch. | [H3 Reference Video Swap Local](../workflows/h3/H10_Subject_replacement_guided_by_a_reference_video/H3_Reference_Video_Swap_Local.json) |
| H11 | Video using MP4 voice, identity, and/or action references | Independent media roles; new generated dialogue/audio; matching last-frame export. Standalone audio-reference node exists but is not a separate shipped graph. | [H3 HQ MP4 Voice Identity Action v1 2](../workflows/h3/H11_Video_using_MP4_voice_identity_and_or_action_references/H3_HQ_MP4_Voice_Identity_Action_v1_2.json) |

## Restoration

| ID | Task | Description | Workflows |
|---|---|---|---:|
| SR | Standard HQ: SeedVR2 restoration | 7B FP16 restoration baseline using the existing integration reference graph. | [Standard HQ](../workflows/restoration/00_Standard_HQ/SeedVR2_Standard_HQ.json) |
| R01 | Full-image SeedVR2 restoration/upscaling | Groups the identity and native suite copies of the 7B FP16 whole-image finishing workflow. | [06 Optional SeedVR2 FP16 7B](../workflows/restoration/R01_Full_image_SeedVR2_restoration_upscaling/06_Optional_SeedVR2_FP16_7B.json)<br>[34 SeedVR2 Full Frame 7B FP16](../workflows/restoration/R01_Full_image_SeedVR2_restoration_upscaling/34_SeedVR2_Full_Frame_7B_FP16.json) |
| R02 | Protected local SeedVR2 finishing | Enhance a selected patch rather than the entire frame. | [35 SeedVR2 Local Protected Finish](../workflows/restoration/R02_Protected_local_SeedVR2_finishing/35_SeedVR2_Local_Protected_Finish.json) |
| R03 | SeedVR2 video restoration/upscaling | 1080p restoration and 4K upscaling presets, with source FPS/audio connections. | [00 SeedVR2 4K Video Restore](../workflows/restoration/R03_SeedVR2_video_restoration_upscaling/00_SeedVR2_4K_Video_Restore.json)<br>[01 SeedVR2 1080p Video Restore](../workflows/restoration/R03_SeedVR2_video_restoration_upscaling/01_SeedVR2_1080p_Video_Restore.json) |

## Toolbox

| ID | Task | Description | Workflows |
|---|---|---|---:|
| U01 | Draft prompts from reference images | General/two-reference and one-person multi-reference prompt-only graphs; no H3 generation in this task. | [05 Vision Prompt Draft](../workflows/toolbox/U01_Draft_prompts_from_reference_images/05_Vision_Prompt_Draft.json)<br>[09 H3 One Person Prompt Draft](../workflows/toolbox/U01_Draft_prompts_from_reference_images/09_H3_One_Person_Prompt_Draft.json) |
| U02 | Create reusable H3 reference packages | Single-image Full RefMod and multi-view person-bundle creation. These are encoded references, not model or LoRA training. | [06 H3 Create Full RefMod](../workflows/toolbox/U02_Create_reusable_H3_reference_packages/06_H3_Create_Full_RefMod.json)<br>[08 H3 Create Person RefMod](../workflows/toolbox/U02_Create_reusable_H3_reference_packages/08_H3_Create_Person_RefMod.json) |
| U03 | Back up saved H3 reference packages | Export the saved RefMod library. | [07 Backup RefMods](../workflows/toolbox/U03_Back_up_saved_H3_reference_packages/07_Backup_RefMods.json) |
| U04 | Crop and export genuine reference images | No generative model. | [29 Reference Crop Export NO GENERATION](../workflows/toolbox/U04_Crop_and_export_genuine_reference_images/29_Reference_Crop_Export_NO_GENERATION.json) |
| U05 | Non-generative resize/final enlargement | Lanczos resize copies from both Qwen suites. | [05 Final Resize No Extra Model](../workflows/toolbox/U05_Non_generative_resize_final_enlargement/05_Final_Resize_No_Extra_Model.json)<br>[33 Final Lanczos No Model](../workflows/toolbox/U05_Non_generative_resize_final_enlargement/33_Final_Lanczos_No_Model.json) |
| U06 | Preview edit/protection masks without generation | Generic editable/protected mask preview. | [Standalone toolbox](../workflows/toolbox/U06_Mask_Preview.json) |
| U07 | Paste an aligned image patch without AI | Paste image B through a manual mask on original image A. Exact canvas sizes required; no generative model or viewpoint alignment. | [Standalone toolbox](../workflows/toolbox/U07_Composite.json) |
| U08 | Select object in A and reveal B | Standalone SAM selection from Original A, with independent expansion, outward feathering and inversion. | [Standalone toolbox](../workflows/toolbox/U08_SAM_Composite.json) |
| U09 | Select, expand and feather a named object | Standalone SAM selection from Original A, with independent expansion, outward feathering and inversion. | [Standalone toolbox](../workflows/toolbox/U09_SAM_Mask_Preview.json) |


## U10: Qwen prompt enhancement preview

Standalone official I2I BF16 rewrite with source/reference images, one ON/OFF toggle, and original/final text. No image generation. The same toggle is installed in all 35 Qwen generation templates, including Standard HQ. See [the prompt enhancer guide](qwen-prompt-enhancer.md).

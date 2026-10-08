"""Torso garment editing with strict scene-pixel protection. No models/downloads.

Uses the real ComfyUI Qwen 2.1 encoder and AusBoss crop/stitch in the graph.
Masks are compositing constraints, not anatomical detectors or denoising masks.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt, maximum_filter

CATEGORY = "Qwen 2.1 / Torso Lock"
NONE = "__none__"
VERSION = "1.0.0"


def check_image(image, label="image"):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{label}: use one RGB/RGBA still image, not a batch.")
    if min(image.shape[1:3]) < 1 or not torch.isfinite(image).all().item():
        raise ValueError(f"{label}: empty image or non-finite pixels.")


def check_mask(mask, image, label):
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(image.shape[:3]):
        raise ValueError(f"{label}: expected a full-scene mask {tuple(image.shape[:3])}. "
                         "Open the matching Load Image in Mask Editor, paint on MASK, and save.")
    if not torch.isfinite(mask).all().item() or mask.min().item() < 0 or mask.max().item() > 1:
        raise ValueError(f"{label}: mask values must be finite, from zero to one.")
    if not (mask > 0).any().item():
        raise ValueError(f"{label}: mask is empty. White selects the stated region.")


def image_pil(tensor):
    check_image(tensor)
    a=(tensor[0, ..., :3].detach().float().cpu().clamp(0, 1).numpy()*255).round().astype(np.uint8)
    return Image.fromarray(a)


def pil_tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()/255).unsqueeze(0)


def reference_size(image, megapixels):
    """Only resample to fit the budget and the encoder's required 32-pixel grid."""
    check_image(image, "torso/garment reference")
    if not math.isfinite(megapixels) or not 0.05 <= megapixels <= 2.0:
        raise ValueError("Reference budget must be between 0.05 and 2.0 MP.")
    h, w=image.shape[1:3]
    scale=min(1.0, math.sqrt(megapixels*1_000_000/(h*w)))
    # Round down rather than enlarge ordinary references to a megapixel target.
    nw=max(32, int(w*scale)//32*32); nh=max(32, int(h*scale)//32*32)
    if (nw, nh)==(w, h):
        return image
    import comfy.utils
    method="area" if nw*nh < w*h else "lanczos"
    return comfy.utils.common_upscale(image.movedim(-1,1), nw, nh, method, "disabled").movedim(1,-1)


class Q21TOptionalImage:
    """A genuinely optional upload. No dummy image is sent to Qwen."""
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        options=list(nodes.LoadImage.INPUT_TYPES()["required"]["image"][0])
        return {"required": {"image": ([NONE]+[x for x in options if x!=NONE], {"image_upload": True})}}
    RETURN_TYPES=("IMAGE",)
    RETURN_NAMES=("optional_reference",)
    FUNCTION="load"
    CATEGORY=CATEGORY

    def load(self, image):
        if image==NONE:
            return (None,)
        import nodes
        return (nodes.LoadImage().load_image(image)[0],)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if image==NONE:
            return True
        import nodes
        return nodes.LoadImage.VALIDATE_INPUTS(image)

    @classmethod
    def IS_CHANGED(cls, image):
        if image==NONE:
            return NONE
        import nodes
        return nodes.LoadImage.IS_CHANGED(image)


class Q21TPrepareMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_image": ("IMAGE",), "edit_mask": ("MASK",),
            "original_image": ("IMAGE",), "protect_mask": ("MASK",),
            "mode": (["first_pass_same_scene", "refine_accepted_candidate"],),
            "inward_feather_px": ("INT", {"default":8, "min":0, "max":128}),
            "protection_padding_px": ("INT", {"default":0, "min":0, "max":64}),
        }}
    RETURN_TYPES=("IMAGE", "MASK", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES=("base_with_original_lock", "context_union_mask", "strict_paste_alpha", "hard_protection", "mask_preview", "mask_report")
    FUNCTION="prepare"
    CATEGORY=CATEGORY

    def prepare(self, base_image, edit_mask, original_image, protect_mask, mode,
                inward_feather_px=8, protection_padding_px=0):
        check_image(base_image, "base photograph"); check_image(original_image, "original protection photograph")
        if base_image.shape[:3] != original_image.shape[:3]:
            raise ValueError("Original/protection photo and base must have identical pixel dimensions and framing. Do not resize one independently.")
        if mode not in ("first_pass_same_scene", "refine_accepted_candidate"):
            raise ValueError("Unknown mask-preparation mode.")
        if not 0 <= inward_feather_px <= 128 or not 0 <= protection_padding_px <= 64:
            raise ValueError("Mask feather/padding is out of range.")
        base=base_image[..., :3]
        original=original_image[..., :3].to(device=base.device, dtype=base.dtype)
        if mode=="first_pass_same_scene" and (base-original).abs().max().item()>1.01/255:
            raise ValueError("For the first pass, load the SAME original photograph in both scene loaders. Paint MASK, not RGB strokes. The two images differ.")
        check_mask(edit_mask, base_image, "EDIT mask")
        check_mask(protect_mask, original_image, "PROTECT exposed abdomen / navel mask")
        if (protect_mask > 0).all().item():
            raise ValueError("Protection covers the whole image. Protect the existing exposed abdomen, not the entire photograph.")
        edit=edit_mask.to(device=base.device, dtype=base.dtype)
        locked=(protect_mask > 0).detach().cpu().numpy()[0]
        if protection_padding_px:
            locked=maximum_filter(locked, size=2*int(protection_padding_px)+1, mode="constant", cval=0)
        protect=torch.from_numpy(locked.copy()).unsqueeze(0).to(base.device)
        allowed=(edit>0)&(~protect)
        if not allowed.any().item():
            raise ValueError("No editable pixels remain outside protection. Protection always wins; repaint the masks.")
        if inward_feather_px:
            a=allowed[0].detach().cpu().numpy()
            distance=distance_transform_edt(np.pad(a,1))[1:-1,1:-1]
            t=np.clip((distance-1.0)/float(inward_feather_px), 0, 1)
            ramp=torch.from_numpy((t*t*(3-2*t)).astype(np.float32)).unsqueeze(0).to(device=base.device, dtype=base.dtype)
            alpha=edit*ramp
        else:
            alpha=edit.clone()
        alpha=torch.where(allowed, alpha, torch.zeros_like(alpha))
        if alpha.max().item()<=0:
            raise ValueError("The editable area is thinner than the inward feather. Reduce inward_feather_px or enlarge the edit mask.")
        base_locked=torch.where(protect.unsqueeze(-1), original, base)
        # Includes original visible landmarks in the crop even though they cannot be pasted over.
        context=((edit>0)|protect).to(dtype=base.dtype)
        preview=base_locked.clone()
        warm=base.new_tensor([1.0,0.50,0.05]); cool=base.new_tensor([0.0,0.72,1.0])
        preview=torch.where(allowed.unsqueeze(-1), preview*0.60+warm*0.40, preview)
        preview=torch.where(protect.unsqueeze(-1), preview*0.55+cool*0.45, preview)
        report={"version":VERSION,"mode":mode,"editable_pixels":int((alpha>0).sum().item()),
                "protected_pixels":int(protect.sum().item()),"edit_protection_overlap_pixels":int(((edit>0)&protect).sum().item()),
                "inward_feather_px":int(inward_feather_px),"protection_padding_px":int(protection_padding_px),
                "mask_legend":"Orange: editable. Cyan: original pixels locked. Neither overlay is fed to Qwen.",
                "warning":"The node cannot identify anatomy. Only the pixels you paint as protection are locked."}
        return base_locked, context, alpha, protect.to(base.dtype), preview, json.dumps(report,indent=2)


def make_prompt(task, garment_description, extra_instructions, roles):
    garment=next((i for i,role in roles if role=="GARMENT DESIGN ONLY"),None)
    support=[i for i,role in roles if role=="SUPPORTING TORSO ANGLE"]
    if task=="replace_crop_top":
        if garment is None:
            job="Replace the woman's crop top with this garment: "+garment_description.strip()+"."
        else:
            job=(f"Replace the woman's crop top with the sports bra shown in <image{garment}>. "
                 "Use that image for garment construction, coverage, straps, band, fabric and color only, "
                 "not the garment model's anatomy, skin or identity. Fit it to the woman in <image1>.")
        job+=" Reconstruct only the upper-abdominal skin newly revealed below the shorter garment. Do not replace the whole torso."
    elif task=="repair_skin_seam":
        job=("Keep the accepted sports bra design and fit unchanged. Repair only the newly exposed upper-abdominal "
             "skin and its local seam, shading or texture problem. Do not redesign the garment or replace the whole torso.")
    else:
        raise ValueError("Unknown torso-edit task.")
    prompt=("Use <image1> as the base photograph of the same woman. "+job+"\n\n"
        "<image1> is authoritative for identity, pose, torso rotation, perspective, torso length, body outline, "
        "waist position and all already-visible anatomy. Preserve its existing exposed abdomen and belly button "
        "at their current locations. Do not move, reshape or duplicate the navel. Continue the newly revealed skin "
        "into that existing anatomy rather than repositioning anatomy to match a reference.\n\n"
        "Use <image2> as primary photographic evidence of this same woman's actual upper-abdominal anatomy "
        "and visible skin characteristics. Adapt only relevant details to <image1>'s viewpoint and posture. "
        "Do not transfer the reference's apparent scale, torso length, pose, camera angle, clothing or background. ")
    for i in support:
        prompt+=f"<image{i}> is a supporting torso view of the same woman; consult it only where it clarifies details missing from <image2>. "
    prompt+=("Do not average the references into a different physique. Do not invent unseen distinctive marks.\n\n"
        "Render new skin under <image1>'s lighting, exposure, white balance and local shading. Match the adjacent "
        "existing skin without importing the reference photograph's brightness, color cast or shadows. Retain "
        "natural texture and believable garment-contact shadows, not flat painted color.\n\n"
        "Keep face, hair, expression, arms, lower clothing and background unchanged. No slimming, elongation, "
        "added muscle, exaggerated curves, retouching or beautification. Output one natural photograph, no labels or marks.")
    if extra_instructions.strip():
        prompt+="\n\nSpecific local instruction: "+extra_instructions.strip()
    mentioned={int(n) for n in re.findall(r"<image(\d+)>", prompt)}
    if not mentioned.issubset({i for i,_ in roles}):
        raise ValueError("Extra instructions mention an unconnected image number. Use role names instead.")
    return prompt


class Q21TTorsoReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base_crop": ("IMAGE",), "primary_torso": ("IMAGE",),
            "task": (["replace_crop_top", "repair_skin_seam"],),
            "garment_description": ("STRING", {"multiline":True,"default":"a plain black, full-coverage athletic sports bra with natural fit, no added padding or push-up reshaping"}),
            "extra_instructions": ("STRING", {"multiline":True,"default":""}),
            "reference_megapixels": ("FLOAT", {"default":1.0,"min":0.05,"max":2.0,"step":0.05})},
            "optional":{"secondary_torso":("IMAGE",),"third_torso":("IMAGE",),"garment":("IMAGE",)}}
    RETURN_TYPES=("Q21T_REFERENCES","STRING","STRING")
    RETURN_NAMES=("ordered_images","compiled_prompt","reference_roles")
    FUNCTION="plan"
    CATEGORY=CATEGORY

    def plan(self,base_crop,primary_torso,task,garment_description,extra_instructions,reference_megapixels=1.0,
             secondary_torso=None,third_torso=None,garment=None):
        check_image(base_crop,"context crop")
        if base_crop.shape[1]%32 or base_crop.shape[2]%32 or base_crop.shape[1]*base_crop.shape[2]>4_500_000:
            raise ValueError("Use an AusBoss crop with Multiple=32 and target at most 4 MP. The full scene stays at original size.")
        images=[base_crop,reference_size(primary_torso,reference_megapixels)]
        roles=[(1,"BASE SCENE AND EXISTING ANATOMY"),(2,"PRIMARY GENUINE TORSO REFERENCE")]
        for image in (secondary_torso,third_torso):
            if image is not None:
                images.append(reference_size(image,reference_megapixels));roles.append((len(images),"SUPPORTING TORSO ANGLE"))
        if garment is not None and task=="replace_crop_top":
            images.append(reference_size(garment,reference_megapixels));roles.append((len(images),"GARMENT DESIGN ONLY"))
        if garment is None and task=="replace_crop_top" and not garment_description.strip():
            raise ValueError("Supply a garment photo or describe the sports bra.")
        prompt=make_prompt(task,garment_description,extra_instructions,roles)
        return {f"image_{i+1}":im for i,im in enumerate(images)},prompt,"\n".join(f"<image{i}> = {role}" for i,role in roles)


class Q21TNativeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"clip":("CLIP",),"vae":("VAE",),"ordered_images":("Q21T_REFERENCES",),"prompt":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES=("positive","negative","matching_empty_latent")
    FUNCTION="encode"
    CATEGORY=CATEGORY
    def encode(self,clip,vae,ordered_images,prompt):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        result=TextEncodeQwenImage21.execute(clip=clip,vae=vae,prompt=prompt,negative_prompt="",resolution=0,images=ordered_images)
        return tuple(result.result)


class Q21TOpaqueCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"generated":("IMAGE",),"source_crop":("IMAGE",)}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="flatten"
    CATEGORY=CATEGORY
    def flatten(self,generated,source_crop):
        check_image(generated);check_image(source_crop)
        if generated.shape[:3]!=source_crop.shape[:3]:
            raise ValueError("Generated crop size differs from source. Use the encoder's own latent, not a separate Empty Latent.")
        if generated.shape[-1]==3:
            return (generated,)
        alpha=generated[...,3:4].clamp(0,1)
        return (generated[...,:3]*alpha+source_crop[...,:3].to(generated)*(1-alpha),)


class Q21TFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"base_locked":("IMAGE",),"original_image":("IMAGE",),"candidate":("IMAGE",),
            "paste_alpha":("MASK",),"hard_protection":("MASK",),"mask_report":("STRING",{"forceInput":True})},
            "optional":{"reference_roles":("STRING",{"forceInput":True}),"compiled_prompt":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("IMAGE","IMAGE","IMAGE","STRING")
    RETURN_NAMES=("VERIFIED_FINAL","final_comparison","actual_edit_matte","audit_json")
    FUNCTION="finish"
    CATEGORY=CATEGORY
    def finish(self,base_locked,original_image,candidate,paste_alpha,hard_protection,mask_report,reference_roles="",compiled_prompt=""):
        for name,value in (("locked base",base_locked),("original",original_image),("stitched candidate",candidate)):
            check_image(value,name)
            if value.shape[:3]!=base_locked.shape[:3]:
                raise ValueError("Final paste requires aligned images on the identical original-size canvas. No automatic perspective alignment is performed.")
        check_mask(paste_alpha,base_locked,"actual paste alpha");check_mask(hard_protection,base_locked,"hard protection")
        base=base_locked[...,:3];original=original_image[...,:3].to(base);new=candidate[...,:3].to(base)
        pa=paste_alpha.to(base)
        a=pa.unsqueeze(-1)
        keep=(hard_protection.to(base.device)>0).unsqueeze(-1)
        final=torch.where(a<=0,base,torch.where(a>=1,new,base+(new-base)*a))
        final=torch.where(keep,original,final)
        effective=torch.where(keep.squeeze(-1),torch.zeros_like(pa),pa)
        outside=effective.to(base.device)==0
        # The locked base must already contain original protection pixels.
        if not torch.equal(base[keep.expand_as(base)],original[keep.expand_as(base)]):
            raise ValueError("Locked base has different protection pixels. Use Prepare Masks before finalizing.")
        if not torch.equal(final[outside],base[outside]):
            raise RuntimeError("Pixel audit failed: content outside the allowed final paste changed.")
        if not torch.equal(final[keep.expand_as(final)],original[keep.expand_as(original)]):
            raise RuntimeError("Pixel audit failed: protected source pixels changed.")
        report=json.loads(mask_report)
        report.update({"status":"PASS","outside_edit_max_error":0.0,"protected_original_max_error":0.0,
            "width":base.shape[2],"height":base.shape[1],"reference_roles":reference_roles,"compiled_prompt":compiled_prompt,
            "scope":"Exact tensor RGB preservation outside actual paste and inside protection. Not an anatomical, identity, color or duplicate-navel score. PNG exports are 8-bit."})
        tiles=[("BASE WITH ORIGINAL LOCK",base),("VERIFIED FINAL",final)]
        canvas=Image.new("RGB",(1280,830),(30,30,30));draw=ImageDraw.Draw(canvas)
        for i,(label,tensor) in enumerate(tiles):
            tile=ImageOps.contain(image_pil(tensor),(624,776),Image.Resampling.LANCZOS)
            canvas.paste(tile,(i*640+(640-tile.width)//2,40+(776-tile.height)//2))
            draw.text((i*640+12,14),label,fill=(240,240,240))
        matte=effective.unsqueeze(-1).expand(-1,-1,-1,3)
        print("[Q21T] PASS: original protection exact; zero pixel changes outside final edit mask.")
        return final,pil_tensor(canvas),matte,json.dumps(report,indent=2)


class Q21TSaveVerified:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"edit_matte":("IMAGE",),"comparison":("IMAGE",),
            "audit_json":("STRING",{"forceInput":True}),"filename_prefix":("STRING",{"default":"Qwen21_Torso/verified"})},
            "hidden":{"prompt":"PROMPT","extra_pnginfo":"EXTRA_PNGINFO"}}
    RETURN_TYPES=()
    FUNCTION="save"
    OUTPUT_NODE=True
    CATEGORY=CATEGORY
    def save(self,image,edit_matte,comparison,audit_json,filename_prefix,prompt=None,extra_pnginfo=None):
        import folder_paths
        import nodes
        audit=json.loads(audit_json)
        if audit.get("status")!="PASS":
            raise ValueError("Refusing to label an unaudited image as verified.")
        result=nodes.SaveImage().save_images(image,filename_prefix,prompt,extra_pnginfo)
        root=Path(folder_paths.get_output_directory()).resolve()
        for item in result["ui"]["images"]:
            path=(root/item.get("subfolder","")/item["filename"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Output path escaped ComfyUI's output directory.")
            path.with_suffix(".audit.json").write_text(audit_json+"\n",encoding="utf-8")
            image_pil(edit_matte).save(path.with_suffix(".edit_mask.png"))
            image_pil(comparison).save(path.with_suffix(".comparison.jpg"),quality=95,subsampling=0)
        return result


NODE_CLASS_MAPPINGS={cls.__name__:cls for cls in (Q21TOptionalImage,Q21TPrepareMasks,Q21TTorsoReferences,
    Q21TNativeEncode,Q21TOpaqueCrop,Q21TFinalize,Q21TSaveVerified)}
NODE_DISPLAY_NAME_MAPPINGS={
    "Q21TOptionalImage":"Q21T Optional Reference (none = unused)",
    "Q21TPrepareMasks":"Q21T Edit + Original Abdomen Lock",
    "Q21TTorsoReferences":"Q21T Torso / Garment Roles + Prompt",
    "Q21TNativeEncode":"Q21T Native Qwen 2.1 Encode",
    "Q21TOpaqueCrop":"Q21T Flatten Generated Alpha",
    "Q21TFinalize":"Q21T Strict Paste + Original Pixel Audit",
    "Q21TSaveVerified":"Q21T Save VERIFIED Final + Audit",
}

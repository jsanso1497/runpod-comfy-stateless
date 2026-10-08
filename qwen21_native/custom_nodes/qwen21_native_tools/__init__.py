"""Small safety/adaptation nodes for a Qwen 2.1 + AusBoss identity workflow.

No network access, extra models, monkey-patches, or automatic downloads.
Qwen encoding is delegated to the native ComfyUI node, not reimplemented.
"""
from __future__ import annotations
import math
import torch

CATEGORY = "Qwen 2.1 Native Suite"


def budget_dimensions(width: int, height: int, megapixels: float, allow_upscale: bool) -> tuple[int, int]:
    if min(width, height) < 1 or not math.isfinite(megapixels) or megapixels <= 0:
        raise ValueError("Width, height and megapixel budget must be positive.")
    scale = math.sqrt(megapixels * 1_000_000 / (width * height))
    if not allow_upscale:
        scale = min(1.0, scale)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def require_image(image: torch.Tensor, name: str = "image") -> None:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{name}: supply exactly one RGB/RGBA image, not an image batch.")
    if not torch.isfinite(image).all().item():
        raise ValueError(f"{name}: image contains non-finite values.")


class Q21NResizeBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
            "megapixels": ("FLOAT", {"default": 0.59, "min": 0.05, "max": 16.0, "step": 0.01}),
            "allow_upscale": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = CATEGORY
    def resize(self, image, megapixels, allow_upscale):
        import comfy.utils
        require_image(image)
        h, w = image.shape[1:3]
        nw, nh = budget_dimensions(w, h, megapixels, allow_upscale)
        if (w, h) == (nw, nh):
            return image, nw, nh
        method = "area" if nw * nh < w * h else "lanczos"
        result = comfy.utils.common_upscale(image.movedim(-1, 1), nw, nh, method, "disabled").movedim(1, -1)
        return result, nw, nh


class Q21NEncodeReferences:
    """Fixed sockets, contiguous numbering, and the native encoder's matching latent."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "image_1": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""})},
            "optional": {f"image_{i}": ("IMAGE",) for i in range(2,9)}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "matching_empty_latent")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    def encode(self, clip, vae, image_1, prompt, **references):
        import re
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        unknown=set(references)-{f"image_{i}" for i in range(2,9)}
        if unknown:
            raise ValueError(f"Unknown reference slots: {sorted(unknown)}")
        images={"image_1":image_1}; gap=False
        for i in range(2,9):
            image=references.get(f"image_{i}")
            if image is None:
                gap=True
            elif gap:
                raise ValueError(f"Connect reference slots in order: image_{i} follows an empty slot.")
            else:
                images[f"image_{i}"]=image
        for name,image in images.items():
            require_image(image,name)
            if image.shape[1]%32 or image.shape[2]%32:
                raise ValueError(f"{name}: use Native Reference Size or AusBoss Multiple=32.")
        if image_1.shape[1]*image_1.shape[2]>4_500_000:
            raise ValueError("Edit crop exceeds 4.5 MP. Lower the crop target and stitch into the larger scene.")
        if not prompt.strip():
            raise ValueError("Enter an edit instruction before sampling.")
        mentioned={int(n) for n in re.findall(r"<image(\d+)>",prompt)}
        missing=mentioned-set(range(1,len(images)+1))
        if missing:
            raise ValueError(f"Prompt names unconnected image slots: {sorted(missing)}")
        result=TextEncodeQwenImage21.execute(clip=clip,vae=vae,prompt=prompt,
            negative_prompt="",resolution=0,images=images)
        return tuple(result.result)


class Q21NMaskGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",),
            "label": ("STRING", {"default": "Paint the subject mask"})},
            "optional": {"mask_source": ("IMAGE",), "original_scene": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "checked_mask")
    FUNCTION = "check"
    CATEGORY = CATEGORY
    def check(self, image, mask, label, mask_source=None, original_scene=None):
        require_image(image)
        expected = (1, image.shape[1], image.shape[2])
        if tuple(mask.shape) != expected:
            raise ValueError(f"{label}: mask must match the full scene {expected}; got {tuple(mask.shape)}. "
                "Right-click the scene Load Image, open Mask Editor, paint on the MASK layer, and save.")
        if not torch.isfinite(mask).all().item() or mask.min().item() < 0 or mask.max().item() > 1:
            raise ValueError(f"{label}: invalid mask values.")
        if not (mask > 0.01).any().item():
            raise ValueError(f"{label}: no mask painted. White selects the editable region.")
        if (mask_source is None) != (original_scene is None):
            raise ValueError("Supply both mask_source and original_scene for the combined-workflow alignment check.")
        if mask_source is not None:
            require_image(mask_source, "head mask source")
            require_image(original_scene, "original scene")
            if mask_source.shape != original_scene.shape:
                raise ValueError("The head-mask loader must contain the SAME original scene at the SAME dimensions.")
            delta = (mask_source[..., :3] - original_scene[..., :3]).abs().max().item()
            if delta > 1.01 / 255:
                raise ValueError("The two scene loaders do not have matching RGB pixels. Load the same original "
                    "in both; paint masks, not colored RGB marks. Use the staged head workflow after body drift.")
        return image, mask


class Q21NOpaqueImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "background": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flatten"
    CATEGORY = CATEGORY
    def flatten(self, image, background):
        require_image(image)
        require_image(background, "background")
        if background.shape[1:3] != image.shape[1:3]:
            raise ValueError("Decoded image does not match its source crop.")
        rgb = image[..., :3]
        if image.shape[-1] == 3:
            return (rgb,)
        if background.shape[1:3] != image.shape[1:3]:
            raise ValueError("Decoded Qwen image does not match its crop. Check encoder latent wiring.")
        alpha = image[..., 3:4].clamp(0, 1)
        return (rgb * alpha + background[..., :3] * (1 - alpha),)


class Q21NAuditPreservation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited": ("IMAGE",), "blend_mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("verified_image", "audit_report")
    FUNCTION = "audit"
    CATEGORY = CATEGORY
    def audit(self, original, edited, blend_mask):
        require_image(original, "original")
        require_image(edited, "edited")
        if original.shape[1:3] != edited.shape[1:3] or blend_mask.shape != original.shape[:3]:
            raise ValueError("Stitch audit failed: original, edited image and blend mask dimensions differ.")
        if not torch.isfinite(blend_mask).all().item() or blend_mask.min().item()<0 or blend_mask.max().item()>1:
            raise ValueError("Stitch audit: invalid blend mask.")
        protected = blend_mask == 0
        error = (original[..., :3] - edited[..., :3]).abs().amax(dim=-1)
        max_error = error[protected].max().item() if protected.any().item() else 0.0
        if max_error > 0.0:
            raise ValueError(f"Stitch audit failed: pixels outside the actual blend mask changed (max {max_error:.8f}).")
        pct = 100 * protected.float().mean().item()
        report = f"PASS: {pct:.2f}% of original pixels protected; max outside-blend error {max_error:.8f}."
        print("[Q21N Identity] " + report)
        return edited, report


class Q21NMaskUnion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("MASK",), "b": ("MASK",)}}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine"
    CATEGORY = CATEGORY
    def combine(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Cannot combine blend masks from different scene dimensions.")
        return (torch.maximum(a, b),)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (Q21NResizeBudget, Q21NEncodeReferences, Q21NMaskGuard,
    Q21NOpaqueImage, Q21NAuditPreservation, Q21NMaskUnion)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Q21NResizeBudget": "Q21N Reference Size (32px aligned)",
    "Q21NEncodeReferences": "Native Qwen 2.1 Encode (8 fixed image slots)",
    "Q21NMaskGuard": "Q21N Check Mask / Scene Alignment",
    "Q21NOpaqueImage": "Q21N Flatten Generated Alpha",
    "Q21NAuditPreservation": "Q21N Verify Protected Scene Pixels",
    "Q21NMaskUnion": "Q21N Union of Actual Blend Masks",
}

class Q21NSquareCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad"
    CATEGORY = CATEGORY
    def pad(self, image, side):
        import comfy.utils
        require_image(image)
        if side % 32:
            raise ValueError("Square canvas side must be divisible by 32.")
        h, w = image.shape[1:3]
        scale = side / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb = image[..., :3]
        if image.shape[-1] == 4:
            rgb = rgb * image[..., 3:4] + 1 - image[..., 3:4]
        resized = comfy.utils.common_upscale(rgb.movedim(-1, 1), nw, nh, "lanczos", "disabled").movedim(1, -1)
        canvas = torch.ones((1, side, side, 3), dtype=rgb.dtype, device=rgb.device)
        x, y = (side - nw) // 2, (side - nh) // 2
        canvas[:, y:y+nh, x:x+nw] = resized
        return (canvas,)


class Q21NWriteHandoff:
    """Writes only fixed, documented handoff names inside ComfyUI's input directory."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "stage": (["body", "head", "reference", "repair", "expression", "wardrobe", "final"],)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    def save(self, image, stage):
        from pathlib import Path
        import os
        import tempfile
        import numpy as np
        from PIL import Image
        import folder_paths
        require_image(image)
        if stage not in ("body", "head", "reference", "repair", "expression", "wardrobe", "final"):
            raise ValueError("Unknown handoff stage.")
        directory = Path(folder_paths.get_input_directory()) / "qwen21_native_handoff"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stage}_latest.png"
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".handoff-", suffix=".png")
        os.close(fd)
        try:
            pixels = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(pixels).save(temporary, format="PNG")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"[Q21N Identity] Handoff updated: qwen21_native_handoff/{stage}_latest.png")
        return (image,)

NODE_CLASS_MAPPINGS.update({c.__name__: c for c in (Q21NSquareCanvas, Q21NWriteHandoff)})
NODE_DISPLAY_NAME_MAPPINGS.update({"Q21NSquareCanvas": "Q21N Square Source-Prep Canvas",
    "Q21NWriteHandoff": "Q21N Save Stage for Next Workflow"})


class Q21NBlankCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width":("INT",{"default":1152,"min":256,"max":2752,"step":32}),
            "height":("INT",{"default":1728,"min":256,"max":2752,"step":32})}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="create"
    CATEGORY=CATEGORY
    def create(self,width,height):
        if width%32 or height%32 or width*height>4_500_000:
            raise ValueError("Canvas must be 32-aligned and no larger than 4.5 MP.")
        return (torch.full((1,height,width,3),0.75),)


class Q21NProtectRegion:
    """Final pixel lock. White in protection overrides edits and their feathering."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"original":("IMAGE",),"edited":("IMAGE",),
            "blend_mask":("MASK",),"protection":("MASK",)}}
    RETURN_TYPES=("IMAGE","MASK")
    RETURN_NAMES=("protected_image","effective_blend_mask")
    FUNCTION="protect"
    CATEGORY=CATEGORY
    def protect(self,original,edited,blend_mask,protection):
        require_image(original);require_image(edited)
        if original.shape!=edited.shape or tuple(protection.shape)!=tuple(original.shape[:3]) or protection.shape!=blend_mask.shape:
            raise ValueError("Protection mask, original, edit and blend mask must share dimensions.")
        if not torch.isfinite(protection).all().item() or protection.min().item()<0 or protection.max().item()>1:
            raise ValueError("Invalid protection mask.")
        keep=protection>0
        return torch.where(keep.unsqueeze(-1),original,edited),torch.where(keep,torch.zeros_like(blend_mask),blend_mask)


class Q21NDisjointMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"mask_a":("MASK",),"mask_b":("MASK",)}}
    RETURN_TYPES=("MASK","MASK")
    RETURN_NAMES=("mask_a","mask_b")
    FUNCTION="check"
    CATEGORY=CATEGORY
    def check(self,mask_a,mask_b):
        if mask_a.shape!=mask_b.shape:
            raise ValueError("Both masks must use the same original scene dimensions.")
        if ((mask_a>0.01)&(mask_b>0.01)).any().item():
            raise ValueError("The two subject masks overlap. Resolve occlusion ownership or edit separate passes.")
        return mask_a,mask_b


class Q21NComparePanel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"before":("IMAGE",),"after":("IMAGE",),
                "label":("STRING",{"default":"Native candidate"})},
            "optional":{"reference":("IMAGE",)}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="panel"
    CATEGORY=CATEGORY
    def panel(self,before,after,label,reference=None):
        from PIL import Image,ImageDraw,ImageFont,ImageOps
        import numpy as np
        columns=[(before,"SOURCE CROP"),(after,label[:72])]
        if reference is not None:columns.append((reference,"PRIMARY REFERENCE"))
        canvas=Image.new("RGB",(480*len(columns),552),(32,32,32))
        draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
        for i,(tensor,title) in enumerate(columns):
            require_image(tensor)
            array=(tensor[0,...,:3].detach().float().cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
            tile=ImageOps.contain(Image.fromarray(array),(464,500),Image.Resampling.LANCZOS)
            canvas.paste(tile,(i*480+(480-tile.width)//2,40+(500-tile.height)//2))
            draw.text((i*480+12,14),title,font=font,fill=(240,240,240))
        return (torch.from_numpy(np.array(canvas).astype(np.float32)/255.0).unsqueeze(0),)


for _cls in (Q21NBlankCanvas,Q21NProtectRegion,Q21NDisjointMasks,Q21NComparePanel):
    NODE_CLASS_MAPPINGS[_cls.__name__]=_cls
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Q21NBlankCanvas":"Native Qwen - Blank Scene Canvas",
    "Q21NProtectRegion":"Native Qwen - Absolute Pixel Protection",
    "Q21NDisjointMasks":"Native Qwen - Check Separate Subject Masks",
    "Q21NComparePanel":"Native Qwen - Before / After / Reference Panel",
})

"""Small safety/adaptation nodes for a Qwen 2.1 + AusBoss identity workflow.

No network access, extra models, monkey-patches, or automatic downloads.
Qwen encoding is delegated to the native ComfyUI node, not reimplemented.
"""
from __future__ import annotations
import math
import torch

CATEGORY = "Qwen 2.1 Identity Kit"


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


class Q21ResizeBudget:
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


class Q21EncodeReferences:
    """Fixed image sockets avoid depending on frontend Autogrow serialization."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "image_1": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""})},
            "optional": {"image_2": ("IMAGE",), "image_3": ("IMAGE",)}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "matching_empty_latent")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    def encode(self, clip, vae, image_1, prompt, image_2=None, image_3=None):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        if image_3 is not None and image_2 is None:
            raise ValueError("Connect image_2 before image_3 so reference numbering stays unambiguous.")
        images = {}
        for i, image in enumerate((image_1, image_2, image_3), 1):
            if image is None:
                continue
            require_image(image, f"image_{i}")
            if image.shape[1] % 32 or image.shape[2] % 32:
                raise ValueError(f"image_{i}: use Q21 Resize Budget or AusBoss Multiple=32 first.")
            images[f"image_{i}"] = image
        if image_1.shape[1] * image_1.shape[2] > 4_500_000:
            raise ValueError("Qwen edit crop exceeds 4.5 MP. Lower crop target; stitch into the large original instead.")
        if not prompt.strip():
            raise ValueError("Enter an edit instruction before sampling.")
        # Do not replace this empty latent with an SDXL latent or a masked empty latent.
        # The native node supplies Qwen 2.1's 64-channel, matching-size latent.
        result = TextEncodeQwenImage21.execute(clip=clip, vae=vae, prompt=prompt,
            negative_prompt="", resolution=0, images=images)
        return tuple(result.result)


class Q21MaskGuard:
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


class Q21OpaqueImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "background": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flatten"
    CATEGORY = CATEGORY
    def flatten(self, image, background):
        require_image(image)
        require_image(background, "background")
        rgb = image[..., :3]
        if image.shape[-1] == 3:
            return (rgb,)
        if background.shape[1:3] != image.shape[1:3]:
            raise ValueError("Decoded Qwen image does not match its crop. Check encoder latent wiring.")
        alpha = image[..., 3:4].clamp(0, 1)
        return (rgb * alpha + background[..., :3] * (1 - alpha),)


class Q21AuditPreservation:
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
        protected = blend_mask == 0
        error = (original[..., :3] - edited[..., :3]).abs().amax(dim=-1)
        max_error = error[protected].max().item() if protected.any().item() else 0.0
        if max_error > 1e-6:
            raise ValueError(f"Stitch audit failed: pixels outside the actual blend mask changed (max {max_error:.8f}).")
        pct = 100 * protected.float().mean().item()
        report = f"PASS: {pct:.2f}% of original pixels protected; max outside-blend error {max_error:.8f}."
        print("[Q21 Identity] " + report)
        return edited, report


class Q21MaskUnion:
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


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (Q21ResizeBudget, Q21EncodeReferences, Q21MaskGuard,
    Q21OpaqueImage, Q21AuditPreservation, Q21MaskUnion)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Q21ResizeBudget": "Q21 Reference Size (32px aligned)",
    "Q21EncodeReferences": "Q21 Native Encode (fixed image slots)",
    "Q21MaskGuard": "Q21 Check Mask / Scene Alignment",
    "Q21OpaqueImage": "Q21 Flatten Generated Alpha",
    "Q21AuditPreservation": "Q21 Verify Protected Scene Pixels",
    "Q21MaskUnion": "Q21 Union of Actual Blend Masks",
}

class Q21SquareCanvas:
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


class Q21WriteHandoff:
    """Writes only fixed, documented handoff names inside ComfyUI's input directory."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "stage": (["body", "head", "reference", "repair"],)}}
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
        if stage not in ("body", "head", "reference", "repair"):
            raise ValueError("Unknown handoff stage.")
        directory = Path(folder_paths.get_input_directory()) / "qwen21_handoff"
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
        print(f"[Q21 Identity] Handoff updated: qwen21_handoff/{stage}_latest.png")
        return (image,)

NODE_CLASS_MAPPINGS.update({c.__name__: c for c in (Q21SquareCanvas, Q21WriteHandoff)})
NODE_DISPLAY_NAME_MAPPINGS.update({"Q21SquareCanvas": "Q21 Square Source-Prep Canvas",
    "Q21WriteHandoff": "Q21 Save Stage for Next Workflow"})

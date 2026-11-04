"""Small, model-free helpers for the MOSS Mobile native ComfyUI apps."""
from __future__ import annotations

import math
import re
import warnings
import torch
import torch.nn.functional as F

ASPECTS = {
    "Square 1:1": (1, 1),
    "Portrait 4:5": (4, 5),
    "Portrait 2:3": (2, 3),
    "Phone 9:16": (9, 16),
    "Landscape 3:2": (3, 2),
    "Widescreen 16:9": (16, 9),
}
DETAILS = {"Standard (~1 MP)": 1.0, "High (~1.5 MP)": 1.5, "Maximum (~2 MP)": 2.0}


def _image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] < 3 or image.shape[0] != 1:
        raise ValueError("Use a single RGB still image, not a video or image batch.")
    if image.shape[1] < 16 or image.shape[2] < 16:
        raise ValueError("The image must be at least 16 pixels in each dimension.")
    if max(image.shape[1:3]) > 16384:
        raise ValueError("Use a source image no more than 16,384 pixels on either side.")
    if image.shape[1] * image.shape[2] > 32_000_000:
        raise ValueError("Use a source image no larger than 32 megapixels.")
    return image[..., :3]


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[1:3]) == (height, width):
        return image
    return F.interpolate(image.movedim(-1, 1).float(), size=(height, width),
                         mode="bicubic", align_corners=False, antialias=True).movedim(1, -1).clamp(0, 1)


class MossMobileCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shape": (list(ASPECTS), {"default": "Portrait 4:5"}),
            "detail": (list(DETAILS), {"default": "Standard (~1 MP)"}),
        }}
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "make"
    CATEGORY = "MOSS Mobile"

    def make(self, shape, detail):
        a, b = ASPECTS[shape]
        # Exact requested ratio and 16-pixel divisibility, without exceeding the budget.
        units = max(1, math.floor(math.sqrt(DETAILS[detail] * 1024**2 / (a * b * 16**2))))
        return (a * units * 16, b * units * 16)


class MossMobilePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                              "megapixels": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.5}),
                              "upscale_small": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("IMAGE", "INT", "INT", "MOSS_GEOMETRY")
    RETURN_NAMES = ("working_image", "width", "height", "geometry")
    FUNCTION = "prepare"
    CATEGORY = "MOSS Mobile"

    def prepare(self, image, megapixels=1.0, upscale_small=True):
        image = _image(image)
        h, w = map(int, image.shape[1:3])
        scale = math.sqrt(float(megapixels) * 1024**2 / (w * h))
        if not upscale_small:
            scale = min(1.0, scale)
        rh, rw = max(16, round(h * scale)), max(16, round(w * scale))
        ph, pw = math.ceil(rh / 16) * 16, math.ceil(rw / 16) * 16
        fitted = _resize(image, rh, rw)
        # Bottom/right padding keeps the original top-left coordinate fixed.
        padded = F.pad(fitted.movedim(-1, 1), (0, pw-rw, 0, ph-rh), mode="replicate").movedim(1, -1)
        geometry = {"source_h": h, "source_w": w, "content_h": rh, "content_w": rw,
                    "working_h": ph, "working_w": pw}
        return (padded, pw, ph, geometry)


class MossMobileRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generated": ("IMAGE",), "geometry": ("MOSS_GEOMETRY",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "MOSS Mobile"

    def restore(self, generated, geometry):
        generated = _image(generated)
        if tuple(generated.shape[1:3]) != (geometry["working_h"], geometry["working_w"]):
            raise ValueError("Generated dimensions differ from the prepared canvas. Do not change latent width/height independently.")
        cropped = generated[:, :geometry["content_h"], :geometry["content_w"], :]
        return (_resize(cropped, geometry["source_h"], geometry["source_w"]),)


class MossMobileMaskGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask_image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "check"
    CATEGORY = "MOSS Mobile"

    def check(self, image, mask_image):
        image, mask_image = map(_image, (image, mask_image))
        if image.shape[:3] != mask_image.shape[:3]:
            raise ValueError("Use a mask with exactly the source photo's dimensions. White edits; black protects.")
        if not bool(torch.any(mask_image > 0)):
            raise ValueError("The mask is entirely black. Paint the editable area white before running.")
        return (image,)


class MossMobileComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited": ("IMAGE",), "mask_image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "MOSS Mobile"

    def composite(self, original, edited, mask_image):
        original, edited, mask_image = map(_image, (original, edited, mask_image))
        if original.shape != edited.shape or original.shape[:3] != mask_image.shape[:3]:
            raise ValueError("The mask must have exactly the source image's pixel width and height. White edits; black protects.")
        # White/grey RGB mask, independent of alpha convention in LoadImage.
        mask = mask_image.to(device=original.device, dtype=original.dtype).mean(dim=-1, keepdim=True).clamp(0, 1)
        if not bool(torch.any(mask > 0)):
            raise ValueError("The mask is entirely black. Paint the editable area white and leave protected areas black.")
        edited = edited.to(device=original.device, dtype=original.dtype)
        blend = original * (1 - mask) + edited * mask
        # Exact tensor preservation at zero mask, rather than trusting the model.
        return (torch.where(mask == 0, original, blend),)


class MossMobilePromptChoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "expand_prompt": ("BOOLEAN", {"default": True}),
            "original": ("STRING", {"forceInput": True, "lazy": True}),
            "expanded": ("STRING", {"forceInput": True, "lazy": True}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "choose"
    CATEGORY = "MOSS Mobile"

    def check_lazy_status(self, expand_prompt, original=None, expanded=None):
        # Original is also required as a safe fallback; expansion is truly skipped when OFF.
        needed = []
        if original is None:
            needed.append("original")
        if expand_prompt and expanded is None:
            needed.append("expanded")
        return needed

    def choose(self, expand_prompt, original=None, expanded=None):
        if not original or not original.strip():
            raise ValueError("Enter a prompt before running.")
        if not expand_prompt:
            return (original,)
        if not expanded or not expanded.strip():
            warnings.warn("MOSS prompt expansion was empty; using the exact original prompt.")
            return (original,)
        quoted = re.findall(r'"([^"\n]+)"', original)
        if any(text not in expanded for text in quoted):
            warnings.warn("Expansion changed or omitted quoted text; using the exact original prompt.")
            return (original,)
        return (expanded.strip(),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MossMobileCanvas, MossMobilePrepare, MossMobileRestore,
    MossMobileComposite, MossMobileMaskGuard, MossMobilePromptChoice,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MossMobileCanvas": "MOSS Mobile - Image shape and detail",
    "MossMobilePrepare": "MOSS Mobile - Fit and pad, never crop",
    "MossMobileRestore": "MOSS Mobile - Restore original canvas",
    "MossMobileMaskGuard": "MOSS Mobile - Validate mask before sampling",
    "MossMobileComposite": "MOSS Mobile - Protect original pixels",
    "MossMobilePromptChoice": "MOSS Mobile - Optional prompt expansion",
}

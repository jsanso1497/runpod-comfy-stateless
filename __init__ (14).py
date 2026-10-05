"""Two small utility nodes. Generation stays in native ComfyUI/Krea2Edit nodes."""
from __future__ import annotations
import math

ASPECTS = {
    "Match reference": None,
    "Square 1:1": 1.0,
    "Portrait 4:5": 4 / 5,
    "Portrait 2:3": 2 / 3,
    "Landscape 16:9": 16 / 9,
    "Portrait 9:16": 9 / 16,
}


def canvas_size(width: int, height: int, aspect: str, megapixels: float) -> tuple[int, int]:
    if width < 1 or height < 1 or aspect not in ASPECTS or not 0.25 <= megapixels <= 2.0:
        raise ValueError("Use positive image dimensions and a 0.25 to 2.0 MP output")
    ratio = ASPECTS[aspect] or width / height
    if not 1 / 8 <= ratio <= 8:
        raise ValueError("Use a reference with an aspect ratio between 1:8 and 8:1")
    pixels = megapixels * 1024 * 1024
    w, h = math.sqrt(pixels * ratio), math.sqrt(pixels / ratio)
    scale = min(1.0, 2048 / max(w, h))
    # Floor to a valid grid so neither the requested budget nor the 2 MP limit is exceeded.
    return max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)


class EverydayReferenceCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "aspect": (list(ASPECTS), {"default": "Match reference"}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.25}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("reference_image", "width", "height")
    FUNCTION = "prepare"
    CATEGORY = "Everyday/Reference"
    DESCRIPTION = "Limits source processing to 2 MP and chooses an output canvas. It does not preserve identity by itself; Krea2Edit supplies that conditioning."

    def prepare(self, image, aspect, megapixels):
        import comfy.utils
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Use one still reference per Load Image node")
        h, w = image.shape[1:3]
        target_w, target_h = canvas_size(w, h, aspect, megapixels)
        source = image[..., :3]
        scale = min(1.0, math.sqrt((2 * 1024 * 1024) / (w * h)), 4096 / max(w, h))
        if scale < 1.0:
            sw, sh = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
            source = comfy.utils.common_upscale(source.movedim(-1, 1), sw, sh, "lanczos", "disabled").movedim(1, -1)
        return source, target_w, target_h


class EverydayOptionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 0.6, "min": -2.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Everyday/LoRA"
    DESCRIPTION = "Optional model-only LoRA. Off or None is a true pass-through, with no dummy file needed. Select only a LoRA trained for this model family."

    def apply(self, model, enabled, lora_name, strength_model):
        if not enabled or lora_name == "None" or strength_model == 0:
            return (model,)
        import folder_paths
        import comfy.sd
        import comfy.utils
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        patched, _ = comfy.sd.load_lora_for_models(model, None, state, strength_model, 0.0)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "EverydayReferenceCanvas": EverydayReferenceCanvas,
    "EverydayOptionalLoRA": EverydayOptionalLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EverydayReferenceCanvas": "Reference + Output Size",
    "EverydayOptionalLoRA": "Optional LoRA (Off by Default)",
}

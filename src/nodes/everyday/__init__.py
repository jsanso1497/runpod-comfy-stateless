from workbench.registry import lora_choices
"""Optional model-only LoRA. No download or application by default."""
class EverydayOptionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "lora_name": (["None"] + lora_choices('h3'),),
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
        if lora_name not in lora_choices('h3'):raise ValueError('LoRA is not registered for H3.')
        import comfy.sd
        import comfy.utils
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        patched, _ = comfy.sd.load_lora_for_models(model, None, state, strength_model, 0.0)
        return (patched,)
NODE_CLASS_MAPPINGS={"EverydayOptionalLoRA":EverydayOptionalLoRA}

# Additive registration: preserve the pre-existing optional LoRA node unchanged.
from .green_suit import NODE_CLASS_MAPPINGS as _GREEN_SUIT_CLASSES
from .green_suit import NODE_DISPLAY_NAME_MAPPINGS as _GREEN_SUIT_NAMES
NODE_CLASS_MAPPINGS.update(_GREEN_SUIT_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS = dict(_GREEN_SUIT_NAMES)
WEB_DIRECTORY = './web'

"""Shared Qwen toggle and standalone preview, with saved-graph compatibility."""
from workbench.qwen_prompt import Options, enhance_edit_prompt, ui_result

class WBQwenPromptControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'enabled': ('BOOLEAN', {'default': False, 'label_on': 'Enhancer ON', 'label_off': 'Original prompt',
                'tooltip': 'OFF uses your exact original prompt. ON uses the selected local abliterated vision model. Default: Qwen3-VL 32B FP16. Set WB_QWEN_PROMPT_MODEL to switch.'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF, 'control_after_generate': False,
                'tooltip': 'Independent of the image seed. Keep fixed when comparing image seeds.'}),
            'keep_exact': ('STRING', {'default': '', 'multiline': True,
                'tooltip': 'One LoRA trigger or literal phrase per line. Protected when present in that pass. This does not activate a LoRA.'}),
        }}
    RETURN_TYPES = ('WB_QWEN_PROMPT',)
    RETURN_NAMES = ('prompt_enhancer',)
    FUNCTION = 'configure'
    CATEGORY = 'Workbench / Qwen prompt enhancement'
    DESCRIPTION = 'Shared local full-precision vision enhancer. Retries invalid rewrites once, then visibly uses the original prompt. No mask, image order or geometry changes.'

    def configure(self, enabled=False, seed=42, keep_exact=''):
        options = Options.from_value({'enabled': enabled, 'seed': seed, 'keep_exact': keep_exact})
        return ({'enabled': options.enabled, 'seed': options.seed, 'keep_exact': options.keep_exact},)

class WBQwenPromptPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'prompt_enhancer': ('WB_QWEN_PROMPT',), 'image_1': ('IMAGE',),
                             'prompt': ('STRING', {'default': '', 'multiline': True})},
                'optional': {f'image_{i}': ('IMAGE',) for i in range(2, 11)}}
    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('effective_prompt', 'original_prompt', 'enhancement_report')
    OUTPUT_NODE = True
    FUNCTION = 'preview'
    CATEGORY = 'Workbench / Qwen prompt enhancement'
    DESCRIPTION = 'Preview the final prompt without generating an image. Reports selected model, retry and original-prompt fallback. Keep images in target-workflow order.'

    def preview(self, prompt_enhancer, image_1, prompt, **references):
        import json
        images = {'image_1': image_1, **{k: v for k, v in references.items() if v is not None}}
        effective, report = enhance_edit_prompt(prompt, images, prompt_enhancer)
        return ui_result((effective, prompt, json.dumps(report, ensure_ascii=False)), report)

NODE_CLASS_MAPPINGS = {'WBQwenPromptControl': WBQwenPromptControl, 'WBQwenPromptPreview': WBQwenPromptPreview}
NODE_DISPLAY_NAME_MAPPINGS = {'WBQwenPromptControl': 'Qwen | Local Vision Prompt Enhancer ON / OFF',
                              'WBQwenPromptPreview': 'Qwen | Preview Enhanced Prompt Only'}
WEB_DIRECTORY = './web'

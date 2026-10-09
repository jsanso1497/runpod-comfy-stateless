"""Shared Qwen prompt toggle and standalone preview. Nothing loads on import."""
from workbench.qwen_prompt import Options, enhance_edit_prompt, ui_result

class WBQwenPromptControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'enabled': ('BOOLEAN', {'default': False, 'label_on': 'Enhancer ON', 'label_off': 'Original prompt',
                'tooltip': 'OFF uses your exact original prompt without loading the enhancer. ON uses official Qwen I2I BF16.'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF, 'control_after_generate': False,
                'tooltip': 'Independent of the image seed. Keep fixed when comparing image seeds.'}),
            'keep_exact': ('STRING', {'default': '', 'multiline': True,
                'tooltip': 'Optional: one LoRA trigger or literal phrase per line. Protected when present in that pass. This does not activate a LoRA.'}),
        }}
    RETURN_TYPES = ('WB_QWEN_PROMPT',)
    RETURN_NAMES = ('prompt_enhancer',)
    FUNCTION = 'configure'
    CATEGORY = 'Workbench / Qwen prompt enhancement'
    DESCRIPTION = 'One master toggle for all edit passes. Full BF16; no external API, Ollama, quantized fallback or geometry changes.'

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
    DESCRIPTION = 'Preview the rewritten prompt without loading the Qwen image generator. Images must have the same order and roles as the target workflow.'

    def preview(self, prompt_enhancer, image_1, prompt, **references):
        import json
        images = {'image_1': image_1, **{k: v for k, v in references.items() if v is not None}}
        effective, report = enhance_edit_prompt(prompt, images, prompt_enhancer)
        return ui_result((effective, prompt, json.dumps(report, ensure_ascii=False)), report)

NODE_CLASS_MAPPINGS = {'WBQwenPromptControl': WBQwenPromptControl, 'WBQwenPromptPreview': WBQwenPromptPreview}
NODE_DISPLAY_NAME_MAPPINGS = {'WBQwenPromptControl': 'Qwen | I2I Prompt Enhancer ON / OFF (BF16)',
                              'WBQwenPromptPreview': 'Qwen | Preview Enhanced Prompt Only'}
WEB_DIRECTORY = './web'

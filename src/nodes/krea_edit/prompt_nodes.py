"""Backward-compatible Krea prompt nodes backed by the shared Qwen VLM runtime."""
from workbench.krea_prompt import LOCAL, OFFICIAL, enhance as enhance_krea_prompt


class WBKreaPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'editing_inputs': ('WB_KREA_PACKAGE',),
            'instruction': ('STRING', {'default': 'Describe the change to the source photo. Refer to enabled references as R1 to R4.', 'multiline': True}),
            'enhance': ('BOOLEAN', {'default': True}),
            'ollama_model': ('STRING', {'default': 'auto'}),
            'trigger_words': ('STRING', {'default': '', 'multiline': True}),
            'max_tokens': ('INT', {'default': 1536, 'min': 128, 'max': 4096, 'step': 64})},
            'optional': {'backend': ([LOCAL, OFFICIAL], {'default': LOCAL})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('final_edit_instruction',)
    FUNCTION = 'enhance_prompt'
    CATEGORY = 'Workbench/Krea 2'
    STANDARD = False

    def enhance_prompt(self, editing_inputs, instruction, enhance, ollama_model,
                       trigger_words, max_tokens, backend=LOCAL):
        text, report = enhance_krea_prompt(editing_inputs, instruction, enhance,
            ollama_model, trigger_words, max_tokens, backend, self.STANDARD)
        return {'ui': {'text': [text + '\n\n' + report['status']], 'qwen_prompt': [report]}, 'result': (text,)}


class WBKreaStandardPrompt(WBKreaPrompt):
    STANDARD = True

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result['optional']['backend'] = ([LOCAL], {'default': LOCAL})
        return result

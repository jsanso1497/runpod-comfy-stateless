"""Bounds for local Akatz analysis, separate from H3 generation settings."""
from dataclasses import dataclass, asdict

VERSION = '2026-10-27.1'
DEFAULT_OUTPUT_TOKENS = 16384
DEFAULT_CONTEXT_TOKENS = 32768
MAX_PROMPT_CHARS = 160000
DETAIL_LEVELS = ('Detailed', 'Very detailed', 'Exhaustive')
LAYOUTS = ('Single character image', 'Face-left / body-right sheet')


@dataclass(frozen=True)
class AnalysisOptions:
    max_output_tokens: int = DEFAULT_OUTPUT_TOKENS
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    sampled_frames: int = 16
    detail_level: str = 'Exhaustive'
    reference_layout: str = LAYOUTS[0]
    timeout_seconds: int = 1800

    def validate(self):
        for name, low, high in (('max_output_tokens', 1024, 32768),
                                ('sampled_frames', 8, 32), ('timeout_seconds', 120, 3600)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer from {low} to {high}')
        if type(self.context_tokens) is not int or self.context_tokens not in (16384, 32768, 65536):
            raise ValueError('context_tokens must be 16384, 32768, or 65536')
        if self.detail_level not in DETAIL_LEVELS or self.reference_layout not in LAYOUTS:
            raise ValueError('Unsupported detail level or reference layout')
        # Conservative planning reserve, NOT the actual tokenizer count. Vision
        # encodings vary; report Ollama prompt_eval_count after every real request.
        reserve = 2048 + 768 * (self.sampled_frames + 1)
        if self.max_output_tokens + reserve > self.context_tokens:
            raise ValueError('Output budget and sampled images need more context. '
                             'Choose 65536 context, fewer sampled frames, or fewer output tokens.')
        return self

    def record(self):
        return asdict(self)


def validate_target(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 1000 or any(c in value for c in '<>\x00'):
        raise ValueError('Describe the source performer in 1-1000 plain-text characters')
    return ' '.join(value.split())


def validate_prompt(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PROMPT_CHARS or '\x00' in value:
        raise ValueError(f'Prompt must contain 1-{MAX_PROMPT_CHARS:,} characters; it is never silently truncated.')
    return value

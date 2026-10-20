"""Start only the approved Qwen + SAM profile. No private library preparation."""
from __future__ import annotations
import os


def configure_minimal_environment():
    required = {'WB_WORKSPACE': 'qwen', 'WB_APPROVED_ONLY': '1', 'WB_QWEN_ONLY': '1'}
    for key, value in required.items():
        if os.environ.get(key, value) != value:
            raise ValueError(f'This image requires {key}={value}.')
        os.environ[key] = value
    if os.environ.get('WB_TASKS', 'QLOCAL') not in ('', 'QLOCAL'):
        raise ValueError('This image supports only WB_TASKS=QLOCAL.')
    if os.environ.get('WB_TOOLBOX', '') not in ('', 'sam'):
        raise ValueError('This image has no optional model families or prompt enhancers.')
    os.environ['WB_TASKS'] = 'QLOCAL'
    os.environ['WB_TOOLBOX'] = ''  # SAM is already required by the approved task.
    # Do not process inherited private libraries or stale unused secret bindings.
    for key in ('WB_LORA_URLS', 'WB_LORA_FILE', 'WB_CHECKPOINTS', 'WB_CHECKPOINT_FILE',
                'CIVITAI_TOKEN', 'civit_token', 'OLLAMA_MODEL', 'OLLAMA_ANALYSIS_MODEL',
                'WB_QWEN_PROMPT_MODEL', 'WB_QWEN_PROMPT_FAILURE'):
        os.environ.pop(key, None)


def main():
    configure_minimal_environment()
    from .runtime import main as start
    start()


if __name__ == '__main__':
    main()

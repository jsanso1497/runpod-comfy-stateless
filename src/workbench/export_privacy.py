"""Remove user instructions/reference identities from disabled-metadata audits."""
from __future__ import annotations
import copy


_PRIVATE_FIELDS = frozenset(('references', 'instructions', 'keep_exact', 'role',
                             'filename', 'source_filename', 'file_path', 'path',
                             'url', 'source_url', 'trigger_words'))


def public_verification(value):
    """Retain numeric geometry/quality checks without embedded prompt text.

    Q21PhotoStitch stores effective prompts and reference filenames inside its
    verification report. Disabling the outer prompt/workflow metadata alone does
    not remove those nested values. This is a copy; the local UI is unaffected.
    """
    if isinstance(value, dict):
        return {key: public_verification(item) for key, item in value.items()
                if str(key).casefold() not in _PRIVATE_FIELDS
                and 'prompt' not in str(key).casefold()}
    if isinstance(value, list):
        return [public_verification(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_verification(item) for item in value)
    return copy.deepcopy(value)

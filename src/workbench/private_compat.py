"""Metadata-assisted Krea registration without pretending filenames prove compatibility."""
from __future__ import annotations
import re


def krea_metadata(value):
    normalized = re.sub(r'[^a-z0-9]', '', str(value).lower())
    return normalized in {'krea2', 'krea2raw', 'krea2turbo', 'kreakrea2raw', 'kreakrea2turbo'}


def normalize_entry(entry, provider_row):
    """Keep non-Krea rules unchanged; allow exact Krea metadata to fill its declaration.

    This is only a download registration. The existing Krea loader checks the
    instantiated model and requires an explicit Raw/Turbo/Custom sampling choice
    for custom checkpoints. SD/FLUX/other Qwen checkpoints are never converted.
    """
    result = dict(entry)
    metadata = str(provider_row.get('reported_base_model') or '').strip()
    reported_krea = krea_metadata(metadata)
    if reported_krea:
        if result.get('family') not in (None, 'krea'):
            raise ValueError('Provider reports Krea 2, but the entry declares another family.')
        result['family'] = 'krea'
    if result.get('family') != 'krea':
        return result
    if metadata and not reported_krea and metadata.casefold() not in {'other', 'unknown', 'custom'}:
        raise ValueError('Provider base model is not Krea 2. A family label cannot convert another model family.')
    kind = result.get('kind', 'diffusion_models')
    if kind in {'checkpoints', 'diffusion_models'}:
        architecture = result.get('architecture')
        if architecture not in (None, 'krea2'):
            raise ValueError('Krea checkpoint architecture must be krea2.')
        if architecture is None and not reported_krea:
            raise ValueError('Krea metadata is inconclusive. Use a JSON entry with family krea and architecture krea2.')
        result['architecture'] = 'krea2'
        # Civitai calls diffusion-model fine-tunes "Checkpoints". Krea's native
        # UNETLoader reads them from diffusion_models, not CheckpointLoaderSimple.
        result['kind'] = 'diffusion_models'
    return result

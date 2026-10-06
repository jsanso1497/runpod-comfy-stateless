"""Model-family contracts for the repository's pinned KREA and H3 stacks."""
from __future__ import annotations
import copy
import hashlib
import json
import uuid

KREA_DIFFUSION = ('krea2_turbo_bf16.safetensors', 'krea2_raw_bf16.safetensors')
KREA_ENCODER = 'qwen3vl_4b_bf16.safetensors'
KREA_VAE = 'qwen_image_vae.safetensors'
H3_DIFFUSION = ('minimax_h3_ref2va_bf16.safetensors', 'minimax_h3_ref2va_int8_convrot.safetensors')
H3_ENCODERS = ('qwen3vl_32b_minimax_h3_bf16.safetensors', 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
H3_VIDEO_VAE = 'minimax_h3_video_vae_fp16.safetensors'
H3_AUDIO_VAE = 'minimax_h3_audio_vae_fp32.safetensors'

# native class, model field, file category, allowed filenames, encoder type
SPECS = {
    'SafeKreaUNETLoader': ('UNETLoader', 'unet_name', 'diffusion_models', KREA_DIFFUSION, None),
    'SafeKreaCLIPLoader': ('CLIPLoader', 'clip_name', 'text_encoders', (KREA_ENCODER,), 'krea2'),
    'SafeKreaVAELoader': ('VAELoader', 'vae_name', 'vae', (KREA_VAE,), None),
    'SafeH3UNETLoader': ('UNETLoader', 'unet_name', 'diffusion_models', H3_DIFFUSION, None),
    'SafeH3CLIPLoader': ('CLIPLoader', 'clip_name', 'text_encoders', H3_ENCODERS, 'minimax'),
    'SafeH3VideoVAELoader': ('VAELoader', 'vae_name', 'vae', (H3_VIDEO_VAE,), None),
    'SafeH3AudioVAELoader': ('VAELoader', 'vae_name', 'vae', (H3_AUDIO_VAE,), None),
}
NATIVE_FIELDS = {'UNETLoader': ('unet_name', 'weight_dtype'),
                 'CLIPLoader': ('clip_name', 'type', 'device'), 'VAELoader': ('vae_name',)}


def check_selection(safe_type, values):
    native, field, category, allowed, encoder_type = SPECS[safe_type]
    if values.get(field) not in allowed:
        raise ValueError(f'{safe_type}: {field} must be one of {", ".join(allowed)}. '
                         'H3, KREA image models and Ollama prompt models are not interchangeable.')
    if encoder_type and values.get('type') != encoder_type:
        raise ValueError(f'{safe_type}: set type={encoder_type}; do not reuse the other workflow\'s encoder.')
    if native == 'UNETLoader' and values.get('weight_dtype', 'default') != 'default':
        raise ValueError('This protected loader preserves the selected checkpoint precision; use weight_dtype=default.')
    if native == 'CLIPLoader' and values.get('device', 'default') not in ('default', 'cpu'):
        raise ValueError('Text encoder device must be default or cpu.')
    return category, values[field]


def native_view(graph):
    """Expose equivalent native schemas to existing graph validators, without edits."""
    graph = copy.deepcopy(graph)
    for node in graph.get('nodes', []):
        if node['type'] in SPECS:
            node['type'] = SPECS[node['type']][0]
    return graph


def protect_graph(graph):
    """Only harden valid source graphs. Never guess a repair from a node title."""
    out = copy.deepcopy(graph)
    for n in out.get('nodes', []):
        if n['type'] in SPECS:
            native = SPECS[n['type']][0]
        elif n['type'] in NATIVE_FIELDS:
            native = n['type']
        else:
            continue
        fields = NATIVE_FIELDS[native]
        vals = dict(zip(fields, n.get('widgets_values', [])))
        named = n.get('widgets_values_named')
        if named and any(named.get(k) != v for k, v in vals.items()):
            raise ValueError(f'Node {n["id"]}: positional and named model widgets disagree.')
        matches = [t for t, spec in SPECS.items() if spec[0] == native and vals.get(spec[1]) in spec[3]]
        if not matches:
            # Unrelated models in the general template are not rewritten.
            continue
        safe = matches[0]
        check_selection(safe, vals)
        n['type'] = safe
        family = 'KREA 2' if safe.startswith('SafeKrea') else 'H3'
        role = {'UNETLoader': 'diffusion model', 'CLIPLoader': 'text encoder', 'VAELoader': 'VAE'}[native]
        if safe == 'SafeH3AudioVAELoader': role = 'AUDIO VAE'
        if safe == 'SafeH3VideoVAELoader': role = 'VIDEO VAE'
        n['title'] = f'{family} ONLY | {role}'
        n.setdefault('properties', {})['Node name for S&R'] = safe
        for k in ('cnr_id', 'ver', 'models'):
            n['properties'].pop(k, None)
        n['widgets_values_named'] = vals
    if out.get('extra', {}).get('model_safety') != '1.5.3':
        # Distinct UI identity prevents an old open workflow from masking the new copy.
        signature = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
        out['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-model-safety-1.5.3:' + signature))
    out.setdefault('extra', {})['model_safety'] = '1.5.3'
    return out

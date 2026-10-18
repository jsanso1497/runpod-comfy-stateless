"""Narrow CUDA-12 fallback for the pinned ComfyUI 0.41 attention paths.

This is an opt-out compatibility shim, not a driver installer. Full precision
weights are unchanged. It must run once, before any workflow loads its models.
"""
from __future__ import annotations
import ctypes
import functools
import logging
import os

VERSION = 'audit-20261018.1'
_INSTALLED = False


def needs_fallback(cuda_version, driver_api_version=None, policy='auto'):
    if policy not in ('auto', 'safe', 'off'):
        raise ValueError('WB_CUDA_COMPAT must be auto, safe, or off.')
    if policy == 'off' or cuda_version is None:
        return False
    if policy == 'safe':
        return True
    try:
        major = int(str(cuda_version).split('.')[0])
    except (TypeError, ValueError):
        return True
    # The installed kitchen wheels target CUDA 13. Do not probe their kernels
    # on a known-incompatible host: a failed launch can poison a CUDA context.
    return major < 13 or driver_api_version is None or driver_api_version < 13000


def driver_api_version():
    try:
        lib = ctypes.CDLL('libcuda.so.1')
        value = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(value)) == 0:
            return value.value
    except (OSError, AttributeError):
        pass
    return None


def fixed_cache_sdpa(cache, q, _k=None, _v=None):
    """Same causal visibility and flattened output as FixedKVCache.attend.

    MTP can submit several query tokens at once. Using is_causal=True on a
    cached sequence would mask the wrong positions; reuse the core cache mask.
    """
    import torch.nn.functional as F
    k, v = cache.key, cache.value
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('Unexpected FixedKVCache tensor layout.')
    if q.shape[1] % k.shape[1]:
        raise ValueError('Query heads must be a multiple of KV heads.')
    count = q.shape[1] // k.shape[1]
    if count != 1:
        k = k.repeat_interleave(count, dim=1)
        v = v.repeat_interleave(count, dim=1)
    mask = cache.mask(q.shape[2]).to(device=q.device, dtype=q.dtype)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=0.0, is_causal=False)
    return result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def install_compat(*, modules=None, cuda_version=None, driver_version=None, policy=None):
    """Patch only the unavailable kitchen paths, once per ComfyUI process."""
    global _INSTALLED
    if _INSTALLED:
        return {'version': VERSION, 'active': True, 'already_installed': True}
    if modules is None:
        import torch
        cuda_version = torch.version.cuda
        driver_version = driver_api_version() if cuda_version else None
    policy = policy or os.getenv('WB_CUDA_COMPAT', 'auto')
    active = needs_fallback(cuda_version, driver_version, policy)
    report = {'version': VERSION, 'active': active, 'policy': policy,
              'torch_cuda': cuda_version, 'driver_api_version': driver_version}
    if not active:
        return report
    if modules is None:
        import comfy_kitchen as ck
        import comfy.ldm.modules.attention as attention
        import comfy.text_encoders.llama as llama
        import comfy.text_encoders.qwen35 as qwen35
        import comfy.model_management as management
    else:
        ck, attention, llama, qwen35, management = modules
    required = [(attention, 'ComfyAttention'), (attention, 'attention_pytorch'),
                (llama, 'FixedKVCache'), (qwen35, 'GatedAttention')]
    if any(not hasattr(obj, key) for obj, key in required):
        raise RuntimeError('Unsupported ComfyUI attention API for this pinned compatibility shim.')
    if not hasattr(llama.FixedKVCache, 'mask'):
        raise RuntimeError('FixedKVCache causal mask API changed.')
    if hasattr(management, 'args') and getattr(management.args, 'use_ck_attention', False):
        raise RuntimeError('Remove --use-ck-attention when WB_CUDA_COMPAT is active.')
    original = attention.ComfyAttention._load_from_state_dict

    @functools.wraps(original)
    def load_attention(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # Ignore only the kitchen hint in a checkpoint, not arbitrary model data.
        # The original saved config is retained for checkpoint round-tripping.
        if self.method in ('comfy_kitchen_int8', 'comfy_kitchen_sol'):
            self.method, self.function = None, None

    attention.ComfyAttention._load_from_state_dict = load_attention
    attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False
    for name in ('int8_attention_is_available', 'flash_attention_decode_is_available',
                 'sol_attn_is_available', 'int8_decode_is_available'):
        if hasattr(ck, name):
            setattr(ck, name, lambda *args, **kwargs: False)
    management.comfy_kitchen_attention_enabled = lambda: False
    llama.FixedKVCache.attend = fixed_cache_sdpa
    qwen35.GatedAttention.kv_cache_type = lambda self, device, dtype: llama.FixedKVCache
    _INSTALLED = True
    logging.warning('[WB CUDA COMPAT] CUDA-12 safe attention enabled: '
                    'MOSS checkpoint hints and Qwen3.5 token decoding use PyTorch. '
                    'No driver, weight, sampler, or precision changes.')
    return report

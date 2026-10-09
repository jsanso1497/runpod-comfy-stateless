"""Optional official Qwen I2I prompt enhancement using native ComfyUI inference.

No imports of inference libraries, network access, or model loads occur on the
OFF path. The input images are read-only and retain their downstream ordering.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import gc
import hashlib
import json
import re
import threading
from .config import model_root, read_catalog, safe_path

MODEL_ID = 'Qwen/Qwen-Image-2.1-PE-I2I'
REVISION = '72927bc08afc99b7888ceb7d7d51a12db3700bbd'
GROUP = 'qwen_pe_i2i'
MAX_NEW_TOKENS = 24000
_LOCK = threading.RLock()
_CACHE: OrderedDict[str, tuple[str, dict]] = OrderedDict()
_VERIFIED: set[tuple] = set()

# This is an integration constraint, not a replacement for Qwen's system prompt.
INTEGRATION_RULES = '''\n\nWorkbench integration requirements:
The user's image tags and explicit reference-role assignments are authoritative.
The provided images are exactly those used in this edit pass, in the same order.
Some are working crops; do not describe missing full-scene context as visible.
Preserve all explicit identity, wardrobe, pose, perspective and preservation
constraints. Never add a global beautification or style change unless requested.
Preserve quoted text, LoRA trigger strings, and the listed keep-exact terms
verbatim when they occur in the user's instruction. Clarify only the requested
changes. Do not invent a new edit, subject, reference, or additional operation.
The workflow owns mask geometry, pixel protection, size and aspect ratio.
Do not attempt to control those settings through rewritten prose. Output Qwen's
three-field JSON object. Do not include these integration instructions as image
content.''' 


class EnhancementError(ValueError):
    """An actionable error that does not expose prompts or model reasoning."""


@dataclass(frozen=True)
class Options:
    enabled: bool = False
    seed: int = 42
    keep_exact: str = ''

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise EnhancementError('Connect the Qwen prompt enhancer control to this input.')
        if type(value.get('enabled', False)) is not bool:
            raise EnhancementError('Prompt enhancement must be ON or OFF.')
        seed = value.get('seed', 42)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise EnhancementError('Enhancer seed must be an unsigned 64-bit integer.')
        terms = value.get('keep_exact', '')
        if not isinstance(terms, str) or len(terms) > 16000:
            raise EnhancementError('Keep-exact terms must be a text list under 16,000 characters.')
        return cls(value.get('enabled', False), seed, terms)


def parse_response(raw: str, original: str, image_count: int, keep_exact: str = '') -> dict:
    """Extract only the final JSON. Reject truncation, missing locks and bad tags."""
    if not isinstance(raw, str):
        raise EnhancementError('The enhancer returned no text. Turn it OFF or retry.')
    if '</think>' in raw:
        answer = raw.rsplit('</think>', 1)[1].strip()
    elif '<think>' in raw:
        raise EnhancementError('Enhancer thinking was truncated before its final answer. Retry or turn it OFF.')
    else:
        answer = raw.strip()
    answer = re.sub(r'^```(?:json)?\s*', '', answer, flags=re.I)
    answer = re.sub(r'\s*```$', '', answer)
    try:
        result = json.loads(answer)
    except (ValueError, TypeError):
        raise EnhancementError('The enhancer did not return complete valid JSON. Retry or turn it OFF; no image was generated.') from None
    fields = {'rewritten_prompt', 'wh_ratio', 'ratio_follow'}
    if not isinstance(result, dict) or set(result) != fields or any(not isinstance(result[k], str) for k in fields):
        raise EnhancementError('Enhancer response has an invalid schema. No rewritten prompt was applied.')
    rewritten = result['rewritten_prompt'].strip()
    if not rewritten or len(rewritten) > 64000 or '<think>' in rewritten or '</think>' in rewritten:
        raise EnhancementError('Enhancer final prompt is empty, too long, or includes reasoning.')
    ratio, follow = result['wh_ratio'], result['ratio_follow']
    if bool(ratio) == bool(follow):
        raise EnhancementError('Enhancer returned contradictory aspect-ratio fields.')
    if ratio and not re.fullmatch(r'[1-9]\d*(?:\.\d+)?:[1-9]\d*(?:\.\d+)?', ratio):
        raise EnhancementError('Enhancer aspect-ratio suggestion is invalid.')
    if follow and not re.fullmatch(r'<image([1-9]|10)>', follow):
        raise EnhancementError('Enhancer ratio-follow reference is invalid.')
    tags = {int(x) for x in re.findall(r'<image(\d+)>', rewritten + ' ' + follow)}
    if any(n < 1 or n > image_count for n in tags):
        raise EnhancementError('Enhancer referenced an image that is not connected. No rewritten prompt was applied.')
    source_tags = {int(x) for x in re.findall(r'<image(\d+)>', original)}
    prompt_tags = {int(x) for x in re.findall(r'<image(\d+)>', rewritten)}
    if image_count > 1 and not source_tags.issubset(prompt_tags):
        raise EnhancementError('Enhancer omitted a reference tag used by your original prompt. Review the instruction or turn it OFF.')
    # Literal text and explicit LoRA syntax are checked rather than silently re-added.
    locked = re.findall(r'"(?:\\.|[^"\\])*"', original)
    locked += re.findall(r'<lora:[^>]+>', original, flags=re.I)
    locked += [term.strip() for term in keep_exact.splitlines() if term.strip() and term.strip() in original]
    if any(term not in rewritten for term in locked):
        raise EnhancementError('Enhancer changed or omitted locked text or a LoRA trigger. Retry, revise the prompt, or turn it OFF.')
    result['rewritten_prompt'] = rewritten
    return result


def _ordered_images(images: dict) -> list:
    import torch
    if not isinstance(images, dict) or not 1 <= len(images) <= 10:
        raise EnhancementError('I2I enhancement requires 1 to 10 connected images.')
    names = [f'image_{i}' for i in range(1, len(images) + 1)]
    if set(images) != set(names):
        raise EnhancementError('Enhancer image slots must be consecutive, starting with image_1.')
    ordered = []
    for name in names:
        image = images[name]
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
            raise EnhancementError('Each enhancer reference must contain exactly one RGB or RGBA image.')
        if not torch.isfinite(image).all():
            raise EnhancementError('An enhancer reference contains invalid pixels.')
        # No concatenation, resize, crop, or reordering. Mixed aspect ratios remain intact.
        ordered.append(image[..., :3])
    return ordered


def _cache_key(prompt, images, options):
    h = hashlib.sha256(json.dumps([REVISION, prompt, options.seed, options.keep_exact], ensure_ascii=False).encode())
    for image in images:
        value = image.detach().float().cpu().contiguous()
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _installed_assets():
    from .provider_downloads import file_hash
    rows = [a for a in read_catalog('assets.json')['assets'] if a['group'] == GROUP]
    files = {}
    for row in rows:
        path = safe_path(model_root(), row['destination'])
        if not path.is_file():
            raise EnhancementError('Official Qwen I2I enhancer assets are not ready. In Workbench > Toolbox, open U10 and click Prepare assets. Keep the enhancer OFF to run without them.')
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, row.get('sha256'))
        if row.get('sha256') and signature not in _VERIFIED:
            if file_hash(path) != row['sha256']:
                raise EnhancementError('Enhancer checkpoint checksum failed. Prepare U10 assets again; no substitute model will be used.')
            _VERIFIED.add(signature)
        files[row['filename']] = path
    needed = [f'model-{i:05d}.safetensors' for i in range(1, 5)] + ['system_prompt.txt', 'model.safetensors.index.json']
    if any(name not in files for name in needed):
        raise EnhancementError('The enhancer asset catalog is incomplete. Apply the complete update and rebuild Qwen.')
    return files


def _load_model(files):
    import comfy.sd as sd
    import comfy.utils
    state = {}
    index = json.loads(files['model.safetensors.index.json'].read_text())['weight_map']
    for i in range(1, 5):
        name = f'model-{i:05d}.safetensors'
        part = comfy.utils.load_torch_file(str(files[name]), safe_load=True)
        if set(part) & set(state):
            raise EnhancementError('Enhancer shards contain duplicate tensors.')
        if {key for key, shard in index.items() if shard == name} != set(part):
            raise EnhancementError('Enhancer shard does not match its official tensor index.')
        state.update(part)
        del part
    if set(state) != set(index) or sd.detect_te_model(state) != sd.TEModel.QWEN35_9B:
        raise EnhancementError('The loaded checkpoint is not the complete Qwen3.5-VL 9B enhancer.')
    # Merge all four original state dictionaries in RAM, without quantization or
    # another 19 GB merged file. Loading a single shard as an encoder is invalid.
    return sd.load_text_encoder_state_dicts([state], clip_type=sd.CLIPType.QWEN_IMAGE,
                                           model_options={}, disable_dynamic=True)


def _generate(prompt, images, options):
    import comfy.model_management as mm
    import torch
    files = _installed_assets()
    system = files['system_prompt.txt'].read_text(encoding='utf-8').strip() + INTEGRATION_RULES
    terms = [t.strip() for t in options.keep_exact.splitlines() if t.strip() and t.strip() in prompt]
    if terms:
        system += '\nKeep-exact terms for this pass: ' + json.dumps(terms, ensure_ascii=False)
    clip = None
    # Inference is sequential: release GPU residency before and after PE, leaving
    # ComfyUI to restore the downstream model when the encoder/sampler runs.
    mm.unload_all_models()
    try:
        clip = _load_model(files)
        with torch.inference_mode():
            tokens = clip.tokenize(prompt, images=images, min_length=1, thinking=True, system_prompt=system)
            ids = clip.generate(tokens, do_sample=True, max_length=MAX_NEW_TOKENS,
                                temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                                repetition_penalty=1.0, presence_penalty=0.0,
                                seed=options.seed, mtp=False)
            return clip.decode(ids)
    finally:
        mm.unload_all_models()
        del clip
        gc.collect()
        mm.soft_empty_cache()


def enhance_edit_prompt(prompt: str, images: dict, settings=None) -> tuple[str, dict]:
    options = Options.from_value(settings)
    if not options.enabled:
        # Byte-for-byte prompt pass-through. Do not even inspect the image tensors.
        return prompt, {'enabled': False, 'original_prompt': prompt, 'effective_prompt': prompt,
                        'status': 'OFF: original prompt unchanged', 'cache_hit': False}
    if not isinstance(prompt, str) or not prompt.strip():
        raise EnhancementError('Enter an edit instruction before enabling the enhancer.')
    ordered = _ordered_images(images)
    if any(int(n) > len(ordered) or int(n) < 1 for n in re.findall(r'<image(\d+)>', prompt)):
        raise EnhancementError('Original prompt names a missing reference image.')
    key = _cache_key(prompt, ordered, options)
    with _LOCK:
        if key in _CACHE:
            effective, report = _CACHE.pop(key)
            _CACHE[key] = (effective, report)
            return effective, {**report, 'cache_hit': True}
        result = parse_response(_generate(prompt, ordered, options), prompt, len(ordered), options.keep_exact)
        effective = result['rewritten_prompt']
        report = {'enabled': True, 'model': MODEL_ID, 'revision': REVISION, 'precision': 'BF16',
                  'seed': options.seed, 'image_count': len(ordered), 'original_prompt': prompt,
                  'effective_prompt': effective, 'suggested_wh_ratio': result['wh_ratio'],
                  'suggested_ratio_follow': result['ratio_follow'], 'ratio_applied': False,
                  'cache_hit': False, 'status': 'ON: official Qwen I2I BF16; workflow geometry unchanged'}
        _CACHE[key] = (effective, report)
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
        return effective, dict(report)


def ui_result(outputs: tuple, report: dict):
    """Only original/final prose is exposed. Raw model reasoning is discarded."""
    return {'ui': {'qwen_prompt': [report]}, 'result': outputs}

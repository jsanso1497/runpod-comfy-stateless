"""Shared prompt enhancement for existing Qwen image workflows.

ON uses an explicitly selected vision backend: local abliterated Qwen3.8 27B
(default) or the official Qwen Image 2.1 PE-I2I BF16. OFF is exact pass-through
and does not inspect images, resolve settings or access an inference service.
Backend errors never switch to a different model silently.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import re
import threading
from .config import model_root, read_catalog, safe_path

BACKEND_LOCAL = "Qwen3.8 27B (Ollama vision)"
BACKEND_OFFICIAL = "Official Qwen Image 2.1 PE-I2I (BF16)"
BACKEND_CHOICES = (BACKEND_LOCAL, BACKEND_OFFICIAL)

# Legacy official asset identities, retained for cached-asset verification only.
MODEL_ID = 'Qwen/Qwen-Image-2.1-PE-I2I'
REVISION = '72927bc08afc99b7888ceb7d7d51a12db3700bbd'
GROUP = 'qwen_pe_i2i'
OFFICIAL_ASSET_DIR = 'prompt_enhancers/qwen_image_2_1_i2i'
MAX_NEW_TOKENS = 24000
_LOCK = threading.RLock()
_CACHE: OrderedDict[str, tuple[str, dict]] = OrderedDict()
_VERIFIED: set[tuple] = set()
INTEGRATION_RULES = '''\nWorkbench requirements: preserve image tags, reference-role assignments,
identity, wardrobe, pose, perspective, exclusions and literal/LoRA trigger text.
Only clarify requested changes. The workflow owns geometry. Output the official
three-field JSON without reasoning.'''

class EnhancementError(ValueError):
    """An actionable error without private prompts or model reasoning."""

@dataclass(frozen=True)
class Options:
    enabled: bool = False
    seed: int = 42
    keep_exact: str = ''
    repair_error: str = ''
    backend: str = BACKEND_LOCAL

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
        backend = value.get('backend', BACKEND_LOCAL)
        if backend not in BACKEND_CHOICES:
            raise EnhancementError('Choose Qwen3.8 Ollama vision or the official Qwen Image 2.1 I2I prompt enhancer.')
        return cls(value.get('enabled', False), seed, terms, '', backend)


def parse_response(raw: str, original: str, image_count: int, keep_exact: str = '') -> dict:
    """Strict validator. The caller retries invalid output, never relaxes locks."""
    if not isinstance(raw, str):
        raise EnhancementError('The enhancer returned no text.')
    if '</think>' in raw:
        answer = raw.rsplit('</think>', 1)[1].strip()
    elif '<think>' in raw:
        raise EnhancementError('Enhancer thinking was truncated before its final answer.')
    else:
        answer = raw.strip()
    answer = re.sub(r'^```(?:json)?\s*', '', answer, flags=re.I)
    answer = re.sub(r'\s*```$', '', answer)
    try:
        result = json.loads(answer)
    except (ValueError, TypeError):
        raise EnhancementError('The enhancer did not return complete valid JSON.') from None
    fields = {'rewritten_prompt', 'wh_ratio', 'ratio_follow'}
    if not isinstance(result, dict) or set(result) != fields or any(not isinstance(result[k], str) for k in fields):
        raise EnhancementError('Enhancer response has an invalid schema.')
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
        raise EnhancementError('Enhancer referenced an image that is not connected.')
    source_tags = {int(x) for x in re.findall(r'<image(\d+)>', original)}
    prompt_tags = {int(x) for x in re.findall(r'<image(\d+)>', rewritten)}
    if image_count > 1 and not source_tags.issubset(prompt_tags):
        raise EnhancementError('Enhancer omitted a reference tag used by the original prompt.')
    locked = re.findall(r'"(?:\\.|[^"\\])*"', original)
    locked += re.findall(r'<lora:[^>]+>', original, flags=re.I)
    locked += [term.strip() for term in keep_exact.splitlines() if term.strip() and term.strip() in original]
    if any(term not in rewritten for term in locked):
        raise EnhancementError('Enhancer changed or omitted locked text or a LoRA trigger.')
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
        ordered.append(image[..., :3])
    return ordered


def _backend_identity(options):
    if options.backend == BACKEND_LOCAL:
        from .qwen_prompt_local import cache_identity
        return cache_identity()
    # Cached official replies must not conceal missing/replaced official weights.
    # The hash changes when any asset changes, disappears, or is reinstalled.
    root = model_root()
    components = [MODEL_ID, REVISION, BACKEND_OFFICIAL]
    # Exact known six-file official layout; no model load or catalog access
    # solely to compute the cache identity.
    for name in ([f'model-{i:05d}.safetensors' for i in range(1, 5)]
                 + ['system_prompt.txt', 'model.safetensors.index.json']):
        path = safe_path(root, OFFICIAL_ASSET_DIR + '/' + name)
        try:
            st = path.stat()
            components.append([name, st.st_size, st.st_mtime_ns, st.st_ctime_ns])
        except OSError:
            components.append([name, 'missing'])
    return json.dumps(components)


def _cache_key(prompt, images, options):
    h = hashlib.sha256(json.dumps([_backend_identity(options), options.backend,
                                   prompt, options.seed, options.keep_exact], ensure_ascii=False).encode())
    for image in images:
        value = image.detach().float().cpu().contiguous()
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class BackendSetupError(EnhancementError):
    """Missing/corrupt weights must never trigger the unchanged-prompt fallback."""


def _generate(prompt, images, options):
    if options.backend == BACKEND_OFFICIAL:
        try:
            return _generate_official(prompt, images, options)
        except EnhancementError as exc:
            raise BackendSetupError(str(exc)) from None
    from .qwen_prompt_local import generate
    return generate(prompt, images, options)


def enhance_edit_prompt(prompt: str, images: dict, settings=None) -> tuple[str, dict]:
    from .qwen_only import enabled as qwen_only_enabled, direct_prompt_result
    if qwen_only_enabled():
        return direct_prompt_result(prompt)
    options = Options.from_value(settings)
    if not options.enabled:
        return prompt, {'enabled': False, 'original_prompt': prompt, 'effective_prompt': prompt,
                        'status': 'OFF: original prompt unchanged', 'cache_hit': False, 'fallback_used': False}
    if not isinstance(prompt, str) or not prompt.strip():
        raise EnhancementError('Enter an edit instruction before enabling the enhancer.')
    ordered = _ordered_images(images)
    if any(int(n) > len(ordered) or int(n) < 1 for n in re.findall(r'<image(\d+)>', prompt)):
        raise EnhancementError('Original prompt names a missing reference image.')
    from . import qwen_prompt_local as local
    key = _cache_key(prompt, ordered, options)
    with _LOCK:
        if key in _CACHE:
            effective, report = _CACHE.pop(key)
            _CACHE[key] = (effective, report)
            return effective, {**report, 'cache_hit': True}
        result = None
        validation_error = ''
        attempts = 0
        for attempt in range(2):
            attempts += 1
            attempt_options = replace(options, repair_error=validation_error)
            try:
                raw = _generate(prompt, ordered, attempt_options)
                result = parse_response(raw, prompt, len(ordered), options.keep_exact)
                break
            except BackendSetupError:
                raise
            except (EnhancementError, local.InvalidReply) as exc:
                validation_error = str(exc)
            except local.LocalServiceError as exc:
                # A wrong/missing/text-only/quantized model or a service/memory
                # problem is not a rewrite error. Keep it explicit, not hidden.
                raise EnhancementError(str(exc)) from None
        if result is None and local.failure_mode() == 'error':
            raise EnhancementError('Prompt rewrite failed validation twice. ' + validation_error)
        fallback = result is None
        effective = prompt if fallback else result['rewritten_prompt']
        is_official = options.backend == BACKEND_OFFICIAL
        model = MODEL_ID if is_official else local.selected_model()
        model_label = BACKEND_OFFICIAL if is_official else local.PRESETS[model]['label']
        status = ('WARNING: rewrite failed validation twice; ORIGINAL prompt used unchanged.' if fallback
                  else 'ON: ' + model_label + ('; repaired rewrite' if attempts == 2 else ''))
        report = {'enabled': True, 'backend': 'official' if is_official else 'ollama',
                  'model': model, 'revision': REVISION if is_official else local.VERSION,
                  'precision': 'BF16' if is_official else local.PRESETS[model]['precision'], 'seed': options.seed,
                  'image_count': len(ordered), 'original_prompt': prompt, 'effective_prompt': effective,
                  'suggested_wh_ratio': '' if fallback else result['wh_ratio'],
                  'suggested_ratio_follow': '' if fallback else result['ratio_follow'],
                  'ratio_applied': False, 'cache_hit': False, 'status': status,
                  'fallback_used': fallback, 'attempts': attempts, 'retry_count': attempts - 1,
                  'validation_warning': validation_error if fallback else ''}
        if fallback:
            # No prompt text, image, URL or reasoning is logged. A failed rewrite
            # is not cached, so the next queue can recover rather than stick.
            print('[Qwen prompt enhancer] ' + status, flush=True)
        else:
            _CACHE[key] = (effective, report)
            while len(_CACHE) > 8:
                _CACHE.popitem(last=False)
        return effective, dict(report)


def ui_result(outputs: tuple, report: dict):
    return {'ui': {'qwen_prompt': [report]}, 'result': outputs}


# Legacy native utilities are retained for verifying existing official assets.
# They are NOT selected by the ON path and NOT used when Ollama fails.
def _installed_assets():
    from .provider_downloads import file_hash
    rows = [a for a in read_catalog('assets.json')['assets'] if a['group'] == GROUP]
    files = {}
    for row in rows:
        path = safe_path(model_root(), row['destination'])
        if not path.is_file():
            raise EnhancementError('Official PE-I2I weights are not installed. Prepare Toolbox qwen_pe_i2i in Workbench or set WB_TOOLBOX=qwen_pe_i2i and restart. You can also select Qwen3.8 or turn enhancement OFF.')
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, row.get('sha256'))
        if row.get('sha256') and signature not in _VERIFIED:
            if file_hash(path) != row['sha256']:
                raise EnhancementError('Enhancer checkpoint checksum failed.')
            _VERIFIED.add(signature)
        files[row['filename']] = path
    needed = [f'model-{i:05d}.safetensors' for i in range(1, 5)] + ['system_prompt.txt', 'model.safetensors.index.json']
    if any(name not in files for name in needed):
        raise EnhancementError('Official PE-I2I asset catalog is incomplete. Restore catalog/assets.json before preparing qwen_pe_i2i.')
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
    return sd.load_text_encoder_state_dicts([state], clip_type=sd.CLIPType.QWEN_IMAGE,
                                           model_options={}, disable_dynamic=True)


def _generate_official(prompt, images, options):
    import comfy.model_management as mm
    import torch
    files = _installed_assets()
    system = files['system_prompt.txt'].read_text(encoding='utf-8').strip() + INTEGRATION_RULES
    terms = [t.strip() for t in options.keep_exact.splitlines() if t.strip() and t.strip() in prompt]
    if terms:
        system += '\nKeep-exact terms for this pass: ' + json.dumps(terms, ensure_ascii=False)
    if options.repair_error:
        system += ('\nPrevious rewritten prompt failed strict validation: '
                   + options.repair_error
                   + '. Retry from the original prompt; preserve all original tags, quoted literals and triggers.')
    clip = None
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

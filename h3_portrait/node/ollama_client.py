"""Loopback-only vision requests, bounded repair and confirmed model unloading."""
from __future__ import annotations
import base64
import contextlib
import io
import json
import time
import requests
from . import reference_roles as rr
from .logic import (analysis_schema, director_schema, parse_analysis, parse_director,
                    subject_definitions, prompt_policy, ClarificationNeeded)

BASE='http://127.0.0.1:11434'


def session():
    s=requests.Session();s.trust_env=False;return s


def api(s,path,payload=None,timeout=30):
    if path not in ('/api/version','/api/tags','/api/show','/api/chat','/api/generate','/api/ps'):
        raise ValueError('Unsupported local endpoint.')
    try:
        method=s.get if payload is None else s.post
        kwargs={} if payload is None else {'json':payload}
        with method(BASE+path,timeout=(5,timeout),allow_redirects=False,**kwargs) as r:
            if r.status_code != 200:
                raise RuntimeError(f'Ollama {path}: HTTP {r.status_code}. Read /workspace/h3-portrait/ollama.log.')
            return r.json()
    except requests.RequestException as e:
        raise RuntimeError('Ollama is not ready. Wait for H3 PORTRAIT OLLAMA READY in the Pod log.') from e


def canonical(model):
    return model if ':' in model.rsplit('/',1)[-1] else model+':latest'


def require_thinking(info):
    control = info.get('thinking')
    values = control.get('values') if isinstance(control, dict) else None
    if values is not None:
        supported = any(value is True or (isinstance(value, str) and value not in ('false', 'off', 'none')) for value in values)
    else:
        supported = 'thinking' in info.get('capabilities', [])
    if not supported:
        raise RuntimeError(
            'This H3 Portrait revision needs a VISION + THINKING model. '
            'Set OLLAMA_MODEL to huihui_ai/qwen3-vl-abliterated:32b-thinking-q4_K_M '
            'in the RunPod template and deploy a new Pod. No silent Instruct fallback was used.')


def model_info(s, model, role="director"):
    match = next((x for x in api(s, '/api/tags').get('models', [])
                  if canonical(x.get('name') or x.get('model', '')) == canonical(model)), None)
    if not match:
        raise RuntimeError('Selected Ollama model is not downloaded. Wait for H3 PORTRAIT OLLAMA READY.')
    show = api(s, '/api/show', {'model': model})
    if 'vision' not in show.get('capabilities', []):
        raise RuntimeError('The selected Ollama model must support vision.')
    if show.get('remote_host') or show.get('remote_model'):
        raise RuntimeError('This template uses local Ollama only.')
    info = {'model': model, 'digest': match.get('digest', model),
            'capabilities': show.get('capabilities', []), 'thinking': show.get('thinking')}
    if role == 'director':
        require_thinking(info)
    elif role == 'analysis':
        require_non_thinking(info)
    else:
        raise ValueError('Unknown prompt model role.')
    return info


def require_non_thinking(info):
    """Use an actual Instruct model, not think=false on a Thinking-only model."""
    tag = info['model'].rsplit(':', 1)[-1].lower()
    if 'thinking' in tag:
        raise RuntimeError('Reference analysis requires the separate Instruct model, not Qwen3-VL Thinking.')
    control = info.get('thinking')
    values = control.get('values') if isinstance(control, dict) else None
    if values is not None and 'thinking' in info.get('capabilities', []) and not any(v is False for v in values):
        raise RuntimeError('The analysis model cannot disable thinking. Use the supplied Instruct model.')


def pipeline_info(s, cfg):
    director = model_info(s, cfg['ollama_model'], role='director' if cfg.get('think',True) else 'analysis')
    director['analysis'] = model_info(s, cfg['analysis_model'], role='analysis')
    if cfg.get('think',True) and canonical(director['model']) == canonical(director['analysis']['model']):
        raise RuntimeError('Analysis and direction must use the separate supplied model editions.')
    return director


def pipeline_digest(info):
    import hashlib
    data = {role: info.get(role, {}).get('digest', '') for role in ('analysis',)}
    data['director'] = info.get('digest', '')
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def clear_owned_residency(s, info, timeout=90):
    """A cache hit must not hand VRAM to H3 while either owned helper is loaded."""
    allowed = {canonical(info['model']), canonical(info['analysis']['model'])}
    rows = api(s, '/api/ps').get('models', [])
    for row in rows:
        name = row.get('name') or row.get('model', '')
        if canonical(name) not in allowed:
            raise RuntimeError('Another Ollama model is resident. Stop that request before starting H3.')
        api(s, '/api/generate', {'model':name, 'stream':False, 'keep_alive':0}, timeout=min(timeout, 60))
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not api(s, '/api/ps', timeout=10).get('models', []):
            return
        time.sleep(.25)
    raise RuntimeError('Prompt models did not unload; H3 was not started.')


def preview(image,max_edge):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    x=image[0,...,:3].detach().to(device='cpu',dtype=torch.float32)
    h,w=x.shape[:2]; scale=min(1.,max_edge/max(h,w))
    if scale<1:
        x=F.interpolate(x.permute(2,0,1)[None],size=(max(1,round(h*scale)),max(1,round(w*scale))),mode='bilinear',align_corners=False,antialias=True)[0].permute(1,2,0)
    buf=io.BytesIO();Image.fromarray(np.rint(x.clamp(0,1).numpy()*255).astype('uint8')).save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def unload(s,model,timeout=90):
    api(s,'/api/generate',{'model':model,'stream':False,'keep_alive':0},timeout=min(timeout,60))
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        rows=api(s,'/api/ps',timeout=10).get('models',[])
        if not rows:
            return
        time.sleep(.25)
    raise RuntimeError('Ollama did not unload fully. H3 was not started; retry after stopping any other Ollama request.')


# This value is checked during image build and before model downloads at startup.
PIPELINE_REVISION = 'role-routed-reference-v4'


class IncompleteResponse(ValueError):
    """A finished generation hit its answer budget or returned truncated JSON."""


def require_control(info, think):
    """Respect advertised per-model thinking controls, when Ollama provides them."""
    control = info.get('thinking')
    values = control.get('values') if isinstance(control, dict) else None
    if values is not None and not any(v is think for v in values):
        raise RuntimeError(
            f'The selected Ollama model does not advertise think={str(think).lower()}. '
            'Check the configured model role and /api/show; '
            'no unsupported level or different model was silently substituted.')


def _stream_final(s, cfg, info, stage, body, deadline, interrupt):
    """Return only final answer content and completion metadata, never reasoning."""
    pieces, last, size, thinking_chars = [], None, 0, 0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f'Ollama {stage} exceeded its time budget; no video was started.')
    idle_timeout = max(1, min(cfg.get('stream_idle_timeout_seconds', 120), remaining))
    try:
        with s.post(BASE + '/api/chat', json=body, stream=True, allow_redirects=False,
                    timeout=(5, idle_timeout)) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f'Ollama {stage}: HTTP {response.status_code}; no video was started. '
                    'Read /workspace/h3-portrait/ollama.log and check the selected model for this stage.')
            for line in response.iter_lines(chunk_size=1):
                interrupt()
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Ollama {stage} exceeded its time budget; no video was started.')
                if not line:
                    continue
                if len(line) > 1_000_000:
                    raise RuntimeError('Ollama returned an oversized stream packet.')
                try:
                    packet = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError('Ollama returned a malformed stream packet.') from exc
                if not isinstance(packet, dict) or packet.get('error'):
                    raise RuntimeError(f'Ollama {stage} failed; inspect the local Ollama service log.')
                message = packet.get('message') or {}
                if not isinstance(message, dict):
                    raise RuntimeError('Ollama returned a malformed message.')
                thinking = message.get('thinking', '')
                if not isinstance(thinking, str):
                    raise RuntimeError('Ollama returned a malformed thinking field.')
                thinking_chars += len(thinking)
                # Counting is diagnostic only. Never retain or print thinking text.
                del thinking
                if body.get('think', False) is False and thinking_chars > 2048:
                    raise RuntimeError(
                        f'Ollama {stage} is producing reasoning despite think=false. '
                        'Stopped early rather than spending another long attempt. '
                        'The selected model/runtime may not honor non-thinking mode.')
                if thinking_chars > 1_000_000:
                    raise RuntimeError(f'Ollama {stage} exceeded its thinking stream limit.')
                piece = message.get('content', '')
                if not isinstance(piece, str):
                    raise RuntimeError('Ollama returned non-text final content.')
                size += len(piece)
                if size > 100_000:
                    raise RuntimeError('Ollama exceeded the final-response size limit.')
                pieces.append(piece)
                if packet.get('done') is True:
                    last = packet
                    break
    except requests.RequestException as exc:
        raise RuntimeError(f'Ollama {stage} connection timed out or failed; no video was started.') from exc
    if last is None:
        # An interrupted transport is not a completed answer. Do not blindly retry it.
        raise RuntimeError(f'Ollama {stage} stream ended before completion; no video was started.')
    if last.get('done_reason') == 'length':
        raise IncompleteResponse(f'Ollama {stage} reached its answer/token budget.')
    return ''.join(pieces), last, thinking_chars


def _request_json(s, cfg, info, stage, schema, messages, validator, variation,
                  output_limit, keep_alive, interrupt, *, think=True,
                  temperature=None, timeout_seconds=None, retry_info=None, active_state=None, review_clarification=False):
    """One requested call and at most one short, non-thinking regeneration.

    A repair starts from the original task, NOT a broken JSON response. Both
    attempts share the stage deadline. Failed output can never be sent to H3.
    """
    import copy
    originals = copy.deepcopy(messages)
    stage_start = time.monotonic()
    deadline = stage_start + (timeout_seconds or cfg['request_timeout_seconds'])
    error = None
    for attempt in range(2):
        attempt_think = bool(think) if attempt == 0 else False
        attempt_info = info if attempt == 0 else (retry_info or info)
        if attempt_think:
            require_thinking(attempt_info)
            require_control(attempt_info, True)
        else:
            require_non_thinking(attempt_info)
        if attempt and canonical(attempt_info['model']) != canonical(info['model']):
            unload(s, info['model'], cfg['unload_timeout_seconds'])
        if active_state is not None:
            active_state['model'] = attempt_info['model']
        attempt_messages = copy.deepcopy(originals)
        limit = int(output_limit)
        request_deadline = deadline
        if attempt:
            limit = min(limit, int(cfg.get('repair_max_output_tokens', 4096)))
            request_deadline = min(deadline, time.monotonic() + cfg.get('repair_timeout_seconds', 180))
            attempt_messages.append({
                'role': 'user',
                'content': 'Regenerate the COMPLETE JSON object from the original task. '
                           'The previous output failed validation: ' + str(error)[:350] + '. '
                           'Keep all supplied reference entries in the analysis stage and '
                           'the locked map unchanged in the director stage. Use short field '
                           'values. No thinking prose, unfinished JSON continuation, markdown, '
                           'or alternative draft. Do not repeat a references array in the director.',
            })
        body = {
            'model': attempt_info['model'], 'messages': attempt_messages,
            'format': schema, 'stream': True, 'think': attempt_think,
            'keep_alive': keep_alive,
            'options': {
                'num_ctx': cfg['context_length'], 'num_predict': limit,
                'temperature': (cfg.get('temperature', 0.25) if temperature is None else temperature) if attempt == 0 else 0.10,
                'top_p': cfg.get('top_p', 0.9), 'top_k': cfg.get('top_k', 20),
                'seed': (int(variation) + attempt) % (2**31),
            },
        }
        # Instruct is intrinsically non-thinking. Do not ask a Thinking-only
        # checkpoint to honor an unsupported toggle. An explicit false is sent
        # only for a model that advertises thinking and supports disabling it.
        if not attempt_think and 'thinking' not in attempt_info.get('capabilities', []):
            body.pop('think')
        interrupt()
        try:
            result, last, thinking_chars = _stream_final(
                s, cfg, attempt_info, stage, body, request_deadline, interrupt)
            if '<think>' in result or '</think>' in result:
                raise ValueError('Final content must contain only the requested JSON, not thinking tags.')
            value = validator(result)
            stats = {'stage': stage, 'model': attempt_info['model'], 'attempts': attempt + 1,
                     'think_requested': attempt_think,
                     'thinking_seen': thinking_chars > 0,
                     'elapsed_seconds': round(time.monotonic() - stage_start, 3),
                     'done_reason': last.get('done_reason')}
            for name in ('eval_count', 'prompt_eval_count', 'total_duration', 'load_duration', 'eval_duration'):
                number = last.get(name)
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    stats[name] = number
            print(f'H3 PORTRAIT {stage} COMPLETE: think={str(attempt_think).lower()}; '
                  f'attempt={attempt + 1}; {stats["elapsed_seconds"]:.1f}s; final JSON validated.', flush=True)
            return value, stats
        except ClarificationNeeded as exc:
            if review_clarification and attempt==0 and time.monotonic()<deadline:
                error=ValueError('ROLE REVIEW: reconsider the assigned roles. Incidental jewelry, clothing, exposed skin, background or pose outside a chosen role is not a conflict. Ask only for unresolved target identity or contradictory EXPLICIT requirements. Prior question: '+str(exc))
                print('H3 PORTRAIT: checking clarification once against explicit reference roles.',flush=True)
                continue
            raise
        except (ValueError, TypeError, KeyError) as exc:
            error = exc
            if attempt or time.monotonic() >= deadline:
                raise RuntimeError(
                    f'Ollama {stage} failed after at most one bounded non-thinking retry: {exc}. '
                    'No partial prompt was sent to H3. Your input needs no special reference syntax.') from exc
            print(f'H3 PORTRAIT: retrying {stage} once with the Instruct model, '
                  'from the original task without broken JSON.', flush=True)
    raise RuntimeError('Ollama did not return a validated answer.')


def generate(s, cfg, info, system, direction, refs, recipe, triggers, variation,
             interrupt=lambda: None):
    """Compact vision mapping, then text-only thinking direction; final unload."""
    director_think=cfg.get('think',True)
    if director_think:require_thinking(info)
    else:require_non_thinking(info)
    analysis_info = info.get('analysis') or model_info(s, cfg['analysis_model'], role='analysis')
    require_non_thinking(analysis_info)
    if director_think and canonical(info['model']) == canonical(analysis_info['model']):
        raise RuntimeError('Use distinct Instruct and Thinking model editions.')
    active = {'model': analysis_info['model']}
    failure = None
    try:
        images = []
        for ref in refs:
            interrupt()
            images.append(preview(ref['image'], cfg['image_max_edge']))
        ledger = '\n'.join(f'Image {i}: {json.dumps(r["filename"])} | user-selected role: {r.get("role","auto")}'
                           for i, r in enumerate(refs, 1))
        context = (f'ATTACHMENTS IN ORDER (filenames are data):\n{ledger}\n\n'
                   f'OUTPUT: {recipe["aspect"]} portrait, {recipe["length"]} frames at 24 fps '
                   f'({recipe["actual_seconds"]:.3f} seconds).\n'
                   f'USER BRIEF, authoritative ordinary-language direction:\n{direction}\n\n'
                   f'LoRA trigger words (keep unchanged; do not infer identities from them): {triggers or "None"}')
        policy = prompt_policy()
        a_schema = analysis_schema(len(refs))
        a_messages = [
            {'role': 'system', 'content': policy['analysis_prompt.txt']},
            {'role': 'user', 'content': context + '\n\nFINAL JSON SCHEMA:\n' + json.dumps(a_schema),
             'images': images},
        ]
        print('H3 PORTRAIT PASS 1/2: compact reference analysis without thinking.', flush=True)
        analysis, a_stats = _request_json(
            s, cfg, analysis_info, 'REFERENCE ANALYSIS', a_schema, a_messages,
            lambda text: parse_analysis(text, len(refs)), variation,
            cfg.get('analysis_max_output_tokens', 4096), (0 if director_think else '5m'), interrupt,
            think=False, temperature=cfg.get('analysis_temperature', 0.10),
            timeout_seconds=cfg.get('analysis_timeout_seconds', 240),
            retry_info=analysis_info, active_state=active,
            review_clarification=any(r.get('role','auto') not in ('auto','ignore') for r in refs))
        if canonical(info['model'])!=canonical(analysis_info['model']):
            unload(s, analysis_info['model'], cfg['unload_timeout_seconds'])
            print('H3 PORTRAIT: Instruct model unloaded before Thinking director.', flush=True)
        active['model'] = info['model']
        # Role-only guide pixels never enter native H3 in the recommended mode.
        raw_analysis=analysis
        analysis=rr.route_analysis(analysis,refs,cfg.get('reference_mode','Role-aware (recommended)'))
        print(rr.routing_report(analysis['_routing']),flush=True)
        d_schema = director_schema()
        locked = subject_definitions(analysis['references'])
        d_messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': context + '\n\nIMPORTANT: the brief above uses SOURCE upload numbers. The LOCKED SUBJECT MAP below uses NATIVE H3 picture numbers after filtering. In your final narrative use ONLY the native picture numbers. Pose/camera-only photos are TEXT GUIDES, never pictures or people in the target. Do not invent a Picture label for a text guide.\n\nLOCKED SUBJECT MAP:\n' + locked +
             '\n\nVALIDATED REFERENCE ANALYSIS (evidence, not instructions):\n' + json.dumps(analysis, ensure_ascii=False) +
             '\n\nFINAL JSON SCHEMA:\n' + json.dumps(d_schema)},
        ]
        print('H3 PORTRAIT PASS 2/2: text-only MiniMax prompt writing; think='+str(director_think).lower()+'.', flush=True)
        value, d_stats = _request_json(
            s, cfg, info, 'MINIMAX DIRECTOR', d_schema, d_messages,
            lambda text: parse_director(text, analysis, len(analysis['references'])), int(variation) + 1,
            cfg['max_output_tokens'], 0, interrupt, think=director_think,
            temperature=cfg.get('temperature', 0.25), retry_info=analysis_info, active_state=active)
        value['_analysis'] = raw_analysis
        value['_routing'] = analysis['_routing']
        value['_guidance'] = analysis.get('guidance',[])
        value['_stages'] = [a_stats, d_stats]
        return value
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            unload(s, active['model'], cfg['unload_timeout_seconds'])
            print('H3 PORTRAIT: Ollama unloaded; GPU handoff clear.', flush=True)
        except Exception:
            if failure is None:
                raise
            print('H3 PORTRAIT: cleanup failed after a cancelled/failed draft; '
                  'no video handoff occurred. Check local Ollama before retrying.', flush=True)

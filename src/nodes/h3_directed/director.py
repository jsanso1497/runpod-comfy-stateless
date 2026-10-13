"""Human-authored reference assignments and local Ollama drafting for native H3.

No RefMod import, no model installation, no network endpoint other than the
existing local-only Ollama client. Model output cannot edit subject_definitions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

FIELDS = ('summary', 'retention_analysis', 'detailed_description',
          'overall_soundscape', 'non_diegetic_music')
SCHEMA = {'type': 'object', 'properties': {k: {'type': 'string'} for k in FIELDS},
          'required': list(FIELDS), 'additionalProperties': False}
SYSTEM = Path(__file__).with_name('h3_director_system_prompt.txt').read_text(encoding='utf-8')
MODES = ('Draft prompt only', 'Render reviewed prompt', 'Write prompt and render')
HEADER = re.compile(r'^\[H3_DIRECT_INPUTS ([0-9a-f]{64})\]\n')
SENTINEL = 'HUMAN DIRECTION (verbatim; takes priority):\n'


def text(value, name, limit=4000):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} is required. You define this, not Ollama.')
    out = value.strip()
    if len(out) > limit or '\x00' in out:
        raise ValueError(f'{name}: use at most {limit} characters and no NUL characters.')
    if HEADER.search(out) or SENTINEL in out:
        raise ValueError(f'{name}: reserved prompt-header text is not allowed here.')
    return out


def owner_name(value):
    out = ' '.join(text(value, 'belongs_to', 80).split())
    if any(ch in out for ch in '<>[]{}'):
        raise ValueError('Use a simple belongs_to label, such as Person A, not H3 tags.')
    return out


def validate_image(image):
    import torch
    if (not isinstance(image, torch.Tensor) or image.ndim != 4 or
            image.shape[0] != 1 or image.shape[-1] != 3 or min(image.shape[1:3]) < 32):
        raise ValueError('Each reference must be ONE still RGB image, at least 32 pixels per side.')
    if not image.isfinite().all():
        raise ValueError('The reference contains invalid pixel values.')
    return image


def image_digest(image):
    import torch
    # Exact decoded pixels, not a thumbnail. This is provenance, not face analysis.
    im = image.detach().to(device='cpu', dtype=torch.float32).contiguous()
    h = hashlib.sha256(str(tuple(im.shape)).encode('ascii'))
    h.update(memoryview(im.numpy()).cast('B'))
    return h.hexdigest()


def make_reference(image, belongs_to, contains, use_for, filename=''):
    validate_image(image)
    return {'image': image, 'belongs_to': owner_name(belongs_to),
            'contains': text(contains, 'what_image_contains', 2000),
            'use_for': text(use_for, 'use_this_reference_for', 2000),
            'image_sha256': image_digest(image), 'filename': str(filename)}


def collect(slots):
    expected = {f'reference_{i}' for i in range(1, 10)}
    if set(slots) - expected:
        raise ValueError('Unknown reference slot.')
    refs, owners, gap = [], {}, False
    for i in range(1, 10):
        r = slots.get(f'reference_{i}')
        if r is None:
            gap = True
            continue
        if gap:
            raise ValueError('Fill Reference 1 onward without gaps. Leave only the trailing unused references at (none).')
        if not isinstance(r, dict) or any(k not in r for k in ('image', 'belongs_to', 'contains', 'use_for', 'image_sha256')):
            raise ValueError('Connect a User-Defined Reference node to each active slot.')
        key = owner_name(r['belongs_to']).casefold()
        if key not in owners:
            owners[key] = {'number': len(owners) + 1, 'name': r['belongs_to']}
        refs.append({**r, 'picture': i, 'subject': owners[key]['number'],
                     'belongs_to': owners[key]['name']})
    if not refs:
        raise ValueError('Upload and describe at least Reference 1.')
    for r in refs:
        # Do not silently treat manual labels in descriptions as other assets.
        validate_labels(r['contains'] + '\n' + r['use_for'], len(refs), len(owners))
    return {'references': refs, 'subjects': list(owners.values())}


def definitions(refs):
    lines = []
    for s in refs['subjects']:
        rows = [r for r in refs['references'] if r['subject'] == s['number']]
        pics = ', '.join(f'<Picture {r["picture"]}>' for r in rows)
        lines.append(f'<Subject {s["number"]}> is the user-defined content "{s["name"]}" from {pics}. '
                     'These references belong to the SAME content unit, not separate copies of it.')
        for r in rows:
            # Human text stays verbatim. Label assignment is code, not an LLM decision.
            lines.append(f'  Source <Picture {r["picture"]}>: {r["contains"]}\n'
                         f'  Human-defined role for <Picture {r["picture"]}>: {r["use_for"]}')
    return '\n'.join(lines)


def ledger(refs):
    rows = ['YOUR REFERENCE MAP (not inferred from faces):']
    for r in refs['references']:
        rows.append(f'Reference {r["picture"]} = <Picture {r["picture"]}> -> '
                    f'<Subject {r["subject"]}> ({r["belongs_to"]})\n'
                    f'Contains: {r["contains"]}\nUse for: {r["use_for"]}')
    rows.append('Repeated belongs_to labels share one subject; different labels stay separate. '
                'No automatic headshot, body, clothing or scene priority is assigned.')
    return '\n\n'.join(rows)


def validate_labels(value, pictures, subjects, generated=False):
    for kind, raw in re.findall(r'<\s*(Picture|Subject|Video|Audio)\s+(\d+)\s*>', value, re.I):
        kind, n = kind.lower(), int(raw)
        if kind in ('video', 'audio'):
            raise ValueError('No source video/audio references were supplied; remove those asset tags.')
        limit = pictures if kind == 'picture' else subjects
        if not 1 <= n <= limit:
            raise ValueError(f'Unknown {kind} reference {n}; the supplied map has {limit}.')
        if generated and kind == 'picture':
            raise ValueError('Ollama tried to restate picture assignments. Retry the draft; the human map must own these.')
    if re.search(r'<\s*/?think\s*>|```', value, re.I):
        raise ValueError('Remove reasoning markers or Markdown code fences from the H3 text.')


def check_request(refs, actions, scene_camera_audio, preservation_rules, length, width, height):
    actions = text(actions, 'actions', 6000)
    scene = text(scene_camera_audio, 'scene_camera_audio', 4000)
    preserve = text(preservation_rules, 'preservation_rules', 4000)
    if not isinstance(length, int) or not 124 <= length <= 362 or length % 17 != 5:
        raise ValueError('Use 124-362 frames on H3\'s 17k+5 frame grid.')
    if any(not isinstance(v, int) or not 256 <= v <= 4096 or v % 32 for v in (width, height)):
        raise ValueError('Width/height must be multiples of 32 between 256 and 4096.')
    for field in (actions, scene, preserve):
        validate_labels(field, len(refs['references']), len(refs['subjects']))
    facts = {'reference_map': [{k: r[k] for k in ('picture','subject','belongs_to','contains','use_for','image_sha256')}
                               for r in refs['references']],
             'actions': actions, 'scene_camera_audio': scene, 'preservation_rules': preserve,
             'length': length, 'width': width, 'height': height}
    signature = hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    exact = (SENTINEL + 'Actions:\n' + actions + '\nScene / camera / audio:\n' + scene +
             '\nPreservation rules:\n' + preserve +
             f'\nDuration: {length} frames at 24 fps ({length/24:.3f} seconds). '
             f'Canvas: {width}x{height}.')
    return facts, signature, exact


def format_response(content, refs, exact_direction):
    try:
        obj = json.loads(content)
    except (TypeError, ValueError) as e:
        raise ValueError('Ollama returned invalid prompt JSON. Retry with a different variation_seed.') from e
    if not isinstance(obj, dict) or set(obj) != set(FIELDS):
        raise ValueError('Ollama must return exactly the five writable fields, not its own subject definitions.')
    for key in FIELDS:
        obj[key] = text(obj[key], 'Ollama field ' + key, 20000)
        validate_labels(obj[key], len(refs['references']), len(refs['subjects']), generated=True)
        # Embedded section headings can masquerade as a second reference map.
        if re.search(r'^\s*(?:subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*:', obj[key], re.M):
            raise ValueError('Ollama returned embedded section headings. Retry the draft.')
    if not obj['summary'].startswith('[reference generation]'):
        raise ValueError('The summary must start with [reference generation].')
    if '[Shot 1]' not in obj['detailed_description']:
        raise ValueError('The draft is missing [Shot 1].')
    present = {int(n) for n in re.findall(r'<Subject (\d+)>', '\n'.join(obj.values()))}
    if present != set(range(1, len(refs['subjects'])+1)):
        raise ValueError('Ollama omitted a user-defined subject. Retry or make the requested role clearer.')
    obj['detailed_description'] += '\n\n' + exact_direction
    return ('subject_definitions:\n' + definitions(refs) + '\n\n' +
            '\n\n'.join(k + ':\n' + obj[k] for k in FIELDS))


def review_copy(prompt, signature):
    return f'[H3_DIRECT_INPUTS {signature}]\n' + prompt


def read_review(review, refs, signature, exact_direction):
    review = review.strip()
    m = HEADER.match(review)
    if not m or m.group(1) != signature:
        raise ValueError('Paste the complete current draft, including its first H3_DIRECT_INPUTS line. '
                         'If images, their descriptions, directions or output dimensions changed, draft again.')
    prompt = review[m.end():]
    prefix = 'subject_definitions:\n' + definitions(refs) + '\n\nsummary:\n'
    if not prompt.startswith(prefix):
        raise ValueError('The reference assignments were edited. Change the reference cards and draft again instead.')
    if exact_direction not in prompt:
        raise ValueError('Keep the HUMAN DIRECTION block unchanged. Edit the actions/scene/rules fields and redraft to change it.')
    headings = re.findall(r'^(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):\s*$', prompt, re.M)
    if headings != ['subject_definitions', *FIELDS]:
        raise ValueError('The reviewed draft must retain its six section headings once, in order.')
    validate_labels(prompt, len(refs['references']), len(refs['subjects']))
    if '[Shot 1]' not in prompt:
        raise ValueError('Keep [Shot 1] in the reviewed draft.')
    return prompt


def generate(client, cfg, refs, facts, exact_direction, system_prompt, seed, temperature,
             include_images, interrupt=lambda: None):
    """Reuse the deployed local-only client; preserve unload behavior on all requests."""
    if not cfg['enabled']:
        raise RuntimeError('ENABLE_OLLAMA must be 1 for drafting. Reviewed-prompt rendering can run without it.')
    if not math.isfinite(temperature) or not 0 <= temperature <= 1:
        raise ValueError('Temperature must be between 0 and 1.')
    system_prompt = text(system_prompt, 'system_prompt', 20000)
    instruction = ('HUMAN REFERENCE MAP (read-only):\n' + definitions(refs) + '\n\n' + exact_direction +
                   '\n\nDo not rewrite the ownership map. Use its <Subject N> labels for the action prose. '
                   'Only the five writable JSON fields are requested.')
    with client.session() as session:
        info = client.model_info(session, cfg['model'], cfg['expected_model_digest'])
        message = {'role': 'user', 'content': instruction}
        if include_images:
            message['images'] = [client.encode_image(r['image'], cfg['image_max_edge'])[0]
                                 for r in refs['references']]
            message['content'] += '\nSupplemental images are in exact Picture order. Human descriptions always take priority.'
        else:
            message['content'] += '\nNo images are attached to this drafting request. Do not infer missing visual details.'
        body = {'model': cfg['model'], 'messages': [{'role':'system','content':system_prompt}, message],
                'format': SCHEMA, 'stream': True, 'keep_alive': 0,
                'options': {'num_ctx':32768, 'num_predict':cfg['max_output_tokens'],
                            'temperature':float(temperature), 'seed':int(seed)}}
        if 'thinking' in info.get('capabilities', []):
            body['think'] = False
        error = None
        try:
            interrupt()
            pieces, final, size = [], None, 0
            with session.post(client.BASE_URL + '/api/chat', json=body, stream=True,
                              allow_redirects=False, timeout=(5,cfg['request_timeout_seconds'])) as response:
                if response.status_code != 200:
                    raise RuntimeError(f'Local Ollama chat returned HTTP {response.status_code}. See /workspace/ollama/service.log.')
                for line in response.iter_lines(chunk_size=1):
                    interrupt()
                    if not line:
                        continue
                    packet = json.loads(line)
                    if packet.get('error'):
                        raise RuntimeError('Ollama failed during drafting. No video is authorized from this request.')
                    piece = packet.get('message', {}).get('content', '')
                    if not isinstance(piece, str):
                        raise ValueError('Invalid Ollama text packet.')
                    size += len(piece)
                    if size > 64000:
                        raise ValueError('Ollama draft exceeded the text limit.')
                    pieces.append(piece)
                    if packet.get('done'):
                        final = packet
                        break
            if final is None or final.get('done_reason') == 'length':
                raise ValueError('Draft truncated. Shorten the directions or increase the configured output-token budget.')
            prompt = format_response(''.join(pieces), refs, exact_direction)
            return prompt, {**info, 'seed':seed, 'temperature':temperature,
                            'images_sent_to_ollama':bool(include_images), 'human_inputs':facts,
                            'total_duration_ns':final.get('total_duration')}
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                client.unload_and_wait(session, cfg['model'], cfg['unload_timeout_seconds'])
            except Exception:
                if error is None:
                    raise
                print('[DirectedH3] Unload was not confirmed after a failed/cancelled request; no H3 handoff returned.', flush=True)

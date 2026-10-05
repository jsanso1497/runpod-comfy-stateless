"""Pure configuration, geometry and internal Ollama-output validation.
There is deliberately NO parser for the user's natural-language direction.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re

MAX_REFERENCES = 9
ROOT = Path(os.environ.get('H3_PORTRAIT_CONFIG', '/workspace/h3-portrait/config'))
DEFAULT_MODEL = 'huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M'
PRESETS = ('Standard', 'Preview', 'High fidelity')
ASPECTS = ('9:16', '2:3')


def settings():
    path = ROOT / 'settings.json'
    if not path.is_file():
        path = Path('/opt/h3-portrait/settings.json')
    data = json.loads(path.read_text())
    data['ollama_model'] = os.environ.get('OLLAMA_MODEL', '').strip() or data['ollama_model']
    name = data['ollama_model']
    if not re.fullmatch(r'[A-Za-z0-9_.:/-]+', name) or '://' in name or name.endswith('-cloud'):
        raise ValueError('OLLAMA_MODEL must be a local downloadable Ollama tag.')
    return data


def geometry(aspect, quality, seconds):
    if aspect not in ASPECTS or quality not in PRESETS:
        raise ValueError('Choose aspect 9:16 or 2:3 and a supported quality preset.')
    seconds = int(seconds)
    if not 3 <= seconds <= 15:
        raise ValueError('Duration must be 3 through 15 seconds.')
    n = max(5, round(seconds * 24))
    frames = n + (5 - n % 17) % 17
    # Exact 9:16 full-quality output: crop 6 pixels from each side of the
    # model-compatible 768x1344 canvas. No AI upscale and no stretched people.
    if quality == 'Preview':
        w,h = (576,1024) if aspect == '9:16' else (576,864)
        ow,oh,steps,ref_size = w,h,12,'match'
    else:
        w,h = (768,1344) if aspect == '9:16' else (768,1152)
        ow,oh = (756,1344) if aspect == '9:16' else (768,1152)
        steps = 20 if quality == 'Standard' else 25
        ref_size = 'match' if quality == 'Standard' else 'max'
    return dict(aspect=aspect,quality=quality,width=w,height=h,output_width=ow,
                output_height=oh,steps=steps,ref_image_size=ref_size,length=frames,
                fps=24,actual_seconds=frames/24,sampler='res_multistep',scheduler='normal')


def output_schema(count):
    text = {'type':'string'}
    return {'type':'object','additionalProperties':False,
        'properties':{
          'references':{'type':'array','minItems':count,'maxItems':count,'items':{
            'type':'object','additionalProperties':False,
            'properties':{'image':{'type':'integer','minimum':1,'maximum':count},
                          'subject':text,'contains':text,'use_for':text},
            'required':['image','subject','contains','use_for']}},
          'summary':text,'retention':text,'shot':text,'sound':text,'music':text,
          'clarification':text},
        'required':['references','summary','retention','shot','sound','music','clarification']}


def parse_llm(content, count):
    """Validate the MODEL's JSON, never impose a grammar on user notes."""
    text = content.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I)
    obj = json.loads(text)
    if not isinstance(obj,dict):
        raise ValueError('Return a JSON object, not a list.')
    if set(obj) != set(output_schema(count)['properties']):
        raise ValueError('Return exactly the fields in the supplied schema.')
    refs = obj['references']
    if not isinstance(refs,list) or len(refs) != count:
        raise ValueError(f'The reference array must contain {count} entries.')
    if any(not isinstance(r,dict) or set(r) != {'image','subject','contains','use_for'} for r in refs):
        raise ValueError('Every reference needs image, subject, contains and use_for.')
    if any(type(r['image']) is not int for r in refs):
        raise ValueError('Image numbers must be integers.')
    if sorted(r['image'] for r in refs) != list(range(1,count+1)):
        raise ValueError('Return every supplied image number exactly once, without extra images.')
    refs.sort(key=lambda r:r['image'])
    for r in refs:
        for k in ('subject','contains','use_for'):
            if not isinstance(r[k],str) or not r[k].strip() or len(r[k]) > 2000:
                raise ValueError(f'Reference {r["image"]}: {k} must be a short nonempty string.')
            r[k]=r[k].strip()
    for k in ('summary','retention','shot','sound','music','clarification'):
        if not isinstance(obj[k],str) or len(obj[k]) > 16000:
            raise ValueError(f'{k} must be a string.')
    if obj['clarification'].strip():
        # This is semantic ambiguity, not a malformed response. No video model loads.
        raise ClarificationNeeded(obj['clarification'].strip())
    if not obj['summary'].strip() or not obj['shot'].strip():
        raise ValueError('Summary and shot must describe the requested video.')
    labels = []
    for r in refs:
        if r['subject'].casefold() not in [x.casefold() for x in labels]:
            labels.append(r['subject'])
    narrative = '\n'.join(str(obj[k]) for k in ('summary','retention','shot','sound','music'))
    # Normal picture mentions are valid. Only references to nonexistent assets fail.
    for kind, number in re.findall(r'<\s*(Picture|Subject|Video|Audio)\s+(\d+)\s*>',narrative,re.I):
        limit = count if kind.lower()=='picture' else len(labels) if kind.lower()=='subject' else 0
        if not 1 <= int(number) <= limit:
            raise ValueError(f'Unknown reference <{kind} {number}>. Use only attached picture numbers.')
    return obj


class ClarificationNeeded(ValueError):
    pass


def format_prompt(obj, user_direction, recipe, trigger_words=''):
    refs=obj['references']
    subjects={}
    for r in refs:
        key=r['subject'].casefold()
        if key not in subjects:
            subjects[key]={'number':len(subjects)+1,'label':r['subject'],'refs':[]}
        subjects[key]['refs'].append(r)
    lines=[]
    for s in subjects.values():
        tags=', '.join(f'<Picture {r["image"]}>' for r in s['refs'])
        lines.append(f'<Subject {s["number"]}> ({s["label"]}): references {tags} define this SAME content unit, not extra people.')
        for r in s['refs']:
            lines.append(f'  <Picture {r["image"]}>: {r["contains"]} Use for: {r["use_for"]}')
    shot = obj['shot'].strip()
    if '[Shot 1]' not in shot:
        shot = '[Shot 1] ' + shot
    summary = obj['summary'].strip()
    if not summary.startswith('[reference generation]'):
        summary='[reference generation] '+summary
    text='\n\n'.join([
        'subject_definitions:\n'+'\n'.join(lines),
        'summary:\n'+summary,
        'retention_analysis:\n'+(obj['retention'].strip() or 'Preserve the requested identities and reference roles.'),
        'detailed_description:\n'+shot,
        'overall_soundscape:\n'+(obj['sound'].strip() or 'No dialogue; restrained natural ambience.'),
        'non_diegetic_music:\n'+(obj['music'].strip() or 'N/A'),
        f'DELIVERY: portrait {recipe["aspect"]}; {recipe["actual_seconds"]:.3f} seconds. Keep important subjects clear of frame edges.',
        'AUTHORITATIVE USER DIRECTION:\n'+user_direction.strip()
    ])
    if trigger_words.strip():
        text+='\n\nLoRA trigger words:\n'+trigger_words.strip()
    return text


def cache_key(hashes, instruction, recipe, model_digest, cfg, triggers, variation):
    payload={'v':1,'images':hashes,'instruction':instruction,'aspect':recipe['aspect'],
             'length':recipe['length'],'model':model_digest,'preview_edge':cfg['image_max_edge'],
             'context':cfg['context_length'],'triggers':triggers,'variation':int(variation)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def active_loras(use_loras=True, selections=(), library_root=None):
    """Select explicitly from the shared library; never apply all downloads."""
    if not use_loras:
        return []
    root=Path(library_root or '/workspace/ComfyUI/models/loras')
    index=root/'link-library.json'
    data=json.loads(index.read_text()) if index.is_file() else {}
    by_name={r['lora_name']:r for r in data.get('loras',[])}
    selected=[]
    seen=set()
    for name,strength in selections:
        if name in ('','(none)','None',None):
            continue
        if name in seen:
            raise ValueError('The same LoRA was selected twice. Select it in only one slot.')
        seen.add(name)
        strength=float(strength)
        if not math.isfinite(strength) or not -2<=strength<=2:
            raise ValueError('LoRA strength must be finite and between -2 and 2.')
        if strength==0:
            continue
        row=dict(by_name.get(name,{}))
        if not row:
            # ComfyUI also exposes files a user uploaded by hand. Full key checks
            # still run against H3 when the model loads.
            row={'name':Path(name).name,'lora_name':name,'trigger_words':''}
        row['strength']=strength
        selected.append(row)
    return selected

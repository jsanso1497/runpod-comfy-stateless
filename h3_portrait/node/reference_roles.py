"""Role-aware routing. Guidance photographs never become identity pixels by default.

The original upload order remains the user-facing ledger. Native H3 pictures are
renumbered after filtering; this module records the mapping explicitly.
"""
from __future__ import annotations
import copy
import re

ROLES = ('auto', 'identity', 'face', 'body', 'hair', 'wardrobe', 'pose_camera', 'expression', 'scene', 'ignore')
TEXT_ONLY = {'pose_camera', 'expression'}
STILL_TEXT_ONLY = {'pose_camera', 'expression', 'scene'}
ROLE_SCOPE = {
    'identity': 'subject identity and natural appearance; not a first frame',
    'face': 'facial identity and facial features only; no clothing, jewelry, background or pose transfer',
    'body': 'body proportions and build only; no clothing, undress, jewelry or background transfer',
    'hair': 'hairstyle, length, texture and color only',
    'wardrobe': 'requested clothing and accessories only; do not copy the source wearer identity',
    'pose_camera': 'pose, posture, framing, camera position and angle only; no source-person appearance',
    'expression': 'facial expression only; no source identity or styling',
    'scene': 'environment and lighting only; do not introduce people from the source background',
    'ignore': 'not used',
}


def role_of(ref, override='auto'):
    if override not in ROLES:
        raise ValueError('Unknown reference role. Choose one of the supplied role labels.')
    if override != 'auto':
        return override
    role = ref.get('role', 'auto')
    if role not in ROLES:
        raise ValueError('Ollama returned an unsupported reference role.')
    if role != 'auto':
        return role
    # Backward-compatible reading of previous cached/reference-map formats.
    text = ref.get('use_for', '').lower()
    if any(word in text for word in ('pose', 'camera', 'composition', 'framing', 'viewpoint')) and not any(word in text for word in ('identity', 'face identity', 'body proportions', 'wardrobe')):
        return 'pose_camera'
    return 'identity'


def remap_text(text, mapping):
    """Replace ordinary numeric/ordinal image aliases with the actual H3 ledger."""
    def label(number):
        if number not in mapping:
            raise ValueError('The prompt refers to an image that was not uploaded.')
        target = mapping[number]
        return f'<Picture {target}>' if target is not None else f'text-only guide from source image {number}'
    # Protect replacements from later passes by using sentinels.
    tokens = []
    def token(number):
        tokens.append(label(number)); return f'@@H3REF{len(tokens)-1}@@'
    def group_alias(match):
        parts = re.split(r'\s*(?:,|and|&)\s*', match.group(1), flags=re.I)
        numbers=[]
        for part in parts:
            bounds=re.split(r'\s*(?:-|to)\s*',part,flags=re.I)
            if len(bounds)==2:
                lo,hi=map(int,bounds)
                if not 1<=lo<=hi<=9:raise ValueError('Invalid image range.')
                numbers.extend(range(lo,hi+1))
            else:numbers.append(int(part))
        return ' and '.join(token(n) for n in numbers)
    text=re.sub(r'\b(?:Images|Pictures|Photos|References)\s+((?:\d+)(?:\s*(?:-|to|,|and|&)\s*\d+)+)\b',group_alias,text,flags=re.I)
    text = re.sub(r'<\s*Picture\s+(\d+)\s*>|\b(?:Image|Picture|Photo|Reference)\s*#?\s*(\d+)\b',
                  lambda m: token(int(m.group(1) or m.group(2))), text, flags=re.I)
    ordinals = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9}
    text = re.sub(r'\b('+'|'.join(ordinals)+r')\s+(?:image|picture|photo|reference)\b',
                  lambda m: token(ordinals[m.group(1).lower()]), text, flags=re.I)
    for i, value in enumerate(tokens): text = text.replace(f'@@H3REF{i}@@', value)
    return text


def route_analysis(analysis, uploads, mode='Role-aware (recommended)'):
    allowed = ('Role-aware (recommended)', 'All images visual (comparison)',
               'Still safe swap (pose/scene text-only)')
    if mode not in allowed:
        raise ValueError('Unknown reference-routing mode.')
    text_only_roles = STILL_TEXT_ONLY if mode == 'Still safe swap (pose/scene text-only)' else TEXT_ONLY
    if len(analysis['references']) != len(uploads):
        raise ValueError('Reference map and uploads do not agree.')
    routed = copy.deepcopy(analysis)
    native, guides, ledger, indices, mapping = [], [], [], [], {}
    for source, (entry, upload) in enumerate(zip(analysis['references'], uploads), 1):
        if entry['image'] != source:
            raise ValueError('Reference map is not in upload order.')
        role = role_of(entry, upload.get('role', 'auto'))
        visual = role != 'ignore' and (role not in text_only_roles or mode == 'All images visual (comparison)')
        number = len(native) + 1 if visual else None
        mapping[source] = number
        ledger.append({'source_image': source, 'filename': upload.get('filename',''), 'role': role,
                       'h3_picture': number, 'delivery': 'visual' if visual else 'ignored' if role == 'ignore' else 'text_only'})
        if visual:
            item = {k: entry[k] for k in ('image', 'subject', 'contains', 'use_for')}
            item['image'] = number
            item['role'] = role
            native.append(item); indices.append(source - 1)
        elif role != 'ignore':
            # The analysis policy permits only geometric/expression observations
            # for these roles, not physical appearance of the source performer.
            guides.append({'source_image': source, 'role': role, 'target_subject': entry['subject'],
                           'description': entry['use_for'], 'scope': ROLE_SCOPE[role]})
    if not native:
        raise ValueError('Keep at least one subject/appearance image as a visual reference. Pose-only or ignored images cannot define an H3 portrait identity.')
    for item in native:
        for field in ('contains', 'use_for'):
            item[field] = remap_text(item[field], mapping)
    labels={r['subject'].casefold():r['subject'] for r in native}
    for guide in guides:
        target=guide['target_subject'].casefold()
        if target in labels:guide['target_subject']=labels[target]
        elif len(labels)==1:guide['target_subject']=next(iter(labels.values()))
        else:raise ValueError('Specify which subject the pose/scene/expression guide applies to. The guide performer is not a new subject.')
        guide['description'] = remap_text(guide['description'], mapping)
    routed['references'] = native
    routed['priorities'] = remap_text(routed.get('priorities',''), mapping)
    routed['conflicts'] = [remap_text(x,mapping) for x in routed.get('conflicts',[])]
    routed['guidance'] = guides
    routed['_routing'] = {'mode': mode, 'ledger': ledger, 'native_source_indices': indices, 'source_to_h3': mapping}
    return routed


def routing_report(routing):
    lines = ['REFERENCE ROUTING (source upload -> H3 input)']
    for row in routing['ledger']:
        destination = f'<Picture {row["h3_picture"]}>' if row['h3_picture'] else row['delivery'].replace('_',' ')
        lines.append(f'Image {row["source_image"]}: {row["role"]} -> {destination}')
    lines.append('Text-only guide photos are analyzed by Ollama but their pixels are NOT sent to H3.')
    return '\n'.join(lines)

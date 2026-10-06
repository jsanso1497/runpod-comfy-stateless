#!/usr/bin/env python3
"""Add workflow 08 without rewriting existing working Krea workflows."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('krea_text_scene_builder_base', HERE/'make_multi_workflows.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def make():
    g = base.Graph('described_scene_two_references')
    a = g.add('LoadImage', 'REFERENCE A - primary face / person A', (-1300,-400))
    b = g.add('LoadImage', 'REFERENCE B - body or second view / person B', (-1300,100))
    a['size'] = b['size'] = [360,420]
    prep = g.add('KreaIdentityTextSceneTwoRefs', 'START HERE - describe the NEW scene', (-870,-530))
    prep['size'] = [620,820]
    g.link(a,0,prep,'reference_a'); g.link(b,0,prep,'reference_b')
    models = base.models(g, -200,-1400)
    result = base.stage(g,prep,models,'Two references / new scene')
    # Both inputs are references now. Do NOT retain the old Scene=1 label/boost.
    stage = g.stages[0]
    nodes = {n['id']:n for n in g.nodes}
    patch = nodes[stage['patch']]
    patch['title'] = 'Reference A=4 / Reference B=4 / FIT (neither is a scene)'
    patch['widgets_values_named']['ref_boost_a'] = 4.0
    patch['widgets_values'] = list(patch['widgets_values_named'].values())
    nodes[stage['empty']]['title'] = 'Output aspect from YOUR scene settings, not either reference'
    for n in g.nodes:
        if n['type']=='VAEEncode':
            n['title'] = 'Original reference A appearance' if n['title'].endswith(' / scene appearance') else 'Original reference B appearance'
    save = g.add('SaveImage','Save native Krea image / use 04 for optional upscale',(2090,-520),{'filename_prefix':'KreaIdentity/TwoRefs_DescribedScene'})
    g.link(result,0,save,'images')
    base.note(g,'NO SCENE PHOTO REQUIRED',
        'Upload exactly two original reference photographs. Select One person for complementary face/body or multi-view references, '
        'or Two people for a man and woman or another pair. Describe where they are, clothing, pose, lighting and camera. '
        'For Two people: change reference A use to Person A identity and reference B use to Person B identity; describe placement explicitly. '
        'Both photos stay separate in semantic AND appearance conditioning; no collage and no blank scene proxy. '
        'A=4 / B=4 is an unbenchmarked balanced starting point, not a quality guarantee. Lower one when its pose/background leaks. '
        'Keep 12 steps, CFG 1, FIT, grounding 1024 and the fixed seed for A/B comparisons. Rebalancer starts OFF. '
        'Landscape, portrait and square are selectable; working generation stays at or below 2 MP.',
        -1300,660,1050,310)
    filename = 'Krea_Identity_08_Two_References_Described_Scene_v1_2.json'
    data = g.save(filename)
    data['extra']['krea_identity']['version']='1.2.0'
    (HERE/'workflows'/filename).write_text(json.dumps(data,indent=2)+'\n')
    return data

if __name__=='__main__': make()

#!/usr/bin/env python3
"""Verify H3 Portrait source and release wiring before any model download.

This release deliberately keeps the Lite still workflow on the same MiniMax H3
Ref2VA stack as video. No separate image-generation checkpoint is required.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

EXPECTED_VERSION='1.5.2'
EXPECTED_PIPELINE='role-routed-reference-still-local-refvideo-v5_2'
HERE=Path(__file__).resolve().parent
NODE_FILES=(
    '__init__.py','logic.py','ollama_client.py','analysis_prompt.txt','system_prompt.txt',
    'still_portrait.py','still_system_prompt.txt','reference_roles.py','video_export.py',
    'native_helpers.py','reference_video.py','web/references.js','web/director_controls.js',
)
WORKFLOW_FILES=(
    'workflows/H3_Portrait_Auto.json',
    'workflows/H3_Ref2VA_Standard.json',
    'workflows/H3_Portrait_Image_Lite.json',
    'workflows/H3_Reference_Video_Swap_Local.json',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _settings(source,relative='settings.json'):
    return json.loads((Path(source)/relative).read_text())


def verify_lite_profile(source):
    source=Path(source)
    cfg=_settings(source,'profiles/lite/settings.json')
    if cfg.get('version')!=EXPECTED_VERSION or cfg.get('profile')!='lite':
        raise ValueError('Lite profile settings belong to an older or invalid release.')
    if cfg.get('analysis_think') is not False or cfg.get('think') is not False:
        raise ValueError('Lite must use non-thinking Instruct direction for both prompt stages.')
    if cfg.get('analysis_model')!=cfg.get('ollama_model') or '8b-instruct' not in cfg.get('ollama_model','').lower():
        raise ValueError('Lite uses one 8B Instruct Ollama helper for both prompt stages.')
    files=cfg.get('model_files') or {}
    if set(files)!={'diffusion','encoder','video_vae','audio_vae'}:
        raise ValueError('Lite must contain only the four MiniMax H3 model-file roles.')
    if 'pruned' in files.get('diffusion','').lower() or 'int8_convrot' not in files.get('diffusion',''):
        raise ValueError('Lite must retain the full Ref2VA architecture using INT8 weights.')
    if 'image_files' in cfg:
        raise ValueError('Lite must not require a separate still-image model stack.')
    rows=json.loads((source/'profiles/lite/models.json').read_text())
    if not rows or any(r.get('profile')!='h3' for r in rows):
        raise ValueError('Lite asset catalog must contain H3 assets only.')
    destinations={Path(r.get('destination','')).name for r in rows}
    if {Path(v).name for v in files.values()} != destinations:
        raise ValueError('Lite H3 settings and asset catalog disagree.')
    return cfg


def verify_source(source):
    source=Path(source)
    required=(
        'VERSION','settings.json','models.json','runtime.json','start.sh','prepare_assets.py',
        'ollama_service.py','smoke_check.py','install_workflows.py','patch_refpack_local.py','profiles/lite/settings.json',
        'profiles/lite/models.json',*WORKFLOW_FILES,
    )
    for relative in required+tuple('node/'+x for x in NODE_FILES):
        if not (source/relative).is_file():
            raise ValueError(f'Incomplete portrait upload: missing h3_portrait/{relative}')
    version=(source/'VERSION').read_text().strip();cfg=_settings(source)
    if version!=EXPECTED_VERSION or cfg.get('version')!=EXPECTED_VERSION:
        raise ValueError('Portrait VERSION/settings disagree or belong to an older release.')
    lite=cfg.get('profile')=='lite'
    verify_lite_profile(source)
    if cfg.get('analysis_think') is not False or cfg.get('think') is not (not lite):
        raise ValueError('Prompt-stage thinking controls disagree with the selected profile.')
    if 'instruct' not in cfg.get('analysis_model','').lower():
        raise ValueError('An Instruct reference model is required; a Thinking toggle is insufficient.')
    if lite:
        if cfg.get('analysis_model')!=cfg.get('ollama_model') or '8b-instruct' not in cfg.get('ollama_model','').lower():
            raise ValueError('Lite active profile must use one 8B Instruct helper.')
        if 'image_files' in cfg:
            raise ValueError('Lite active settings must not contain a separate still-image model mapping.')
    elif cfg.get('analysis_model')==cfg.get('ollama_model') or '32b-thinking' not in cfg.get('ollama_model','').lower():
        raise ValueError('Full uses separate Instruct analysis and 32B Thinking direction.')

    client=(source/'node/ollama_client.py').read_text()
    module=ast.parse(client)
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in module.body
                 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                 and isinstance(n.value,ast.Constant)}
    if assignments.get('PIPELINE_REVISION')!=EXPECTED_PIPELINE:
        raise ValueError('Older ollama_client.py detected. Upload the complete h3_portrait/node folder.')
    if 'generate_image' in {n.name for n in module.body if isinstance(n,ast.FunctionDef)}:
        raise ValueError('Older separate-image-model Ollama path detected.')
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_request_json']
    calls.sort(key=lambda n:n.lineno)
    controls=[next((ast.unparse(k.value) for k in c.keywords if k.arg=='think'),None) for c in calls]
    if controls!=['False','director_think']:
        raise ValueError('Expected compact analysis followed by profile-controlled direction.')
    if 'rr.route_analysis(' not in client:
        raise ValueError('Missing reference-role routing.')

    init=(source/'node/__init__.py').read_text();init_module=ast.parse(init)
    init_version=next((ast.literal_eval(n.value) for n in init_module.body if isinstance(n,ast.Assign)
                       and isinstance(n.value,ast.Constant)
                       and any(isinstance(t,ast.Name) and t.id=='PACKAGE_VERSION' for t in n.targets)),None)
    if init_version!=EXPECTED_VERSION:
        raise ValueError('Older node/__init__.py detected. Upload the complete H3 Portrait 1.5.2 node folder.')
    classes={n.name for n in init_module.body if isinstance(n,ast.ClassDef)}
    for name in ('H3PortraitDirector','H3PortraitStillDirector','H3PortraitStillOutput','H3PortraitSaveLastFrame'):
        if name not in classes and name not in init:
            raise ValueError('Missing H3 Portrait 1.5 node: '+name)

    still=(source/'node/still_portrait.py').read_text()
    for marker in ('Still safe swap (pose/scene text-only)','length\': 5','res_multistep','Best stable quality'):
        if marker not in still:
            raise ValueError('Still-image H3 Ref2VA source is incomplete: '+marker)
    roles=(source/'node/reference_roles.py').read_text()
    if "STILL_TEXT_ONLY = {'pose_camera', 'expression', 'scene'}" not in roles:
        raise ValueError('Still safe-swap routing is missing.')

    runtime=json.loads((source/'runtime.json').read_text())
    if set(runtime.get('profiles',{}))!={'h3'} or any(v!='h3' for v in runtime.get('workflow_profiles',{}).values()):
        raise ValueError('Runtime must route video and still workflows through the same H3 profile.')
    still_graph=json.loads((source/'workflows/H3_Portrait_Image_Lite.json').read_text())
    types={n['type'] for n in still_graph.get('nodes',[])}
    required_types={'H3PortraitReferences','H3PortraitStillDirector','H3PortraitModels','H3PortraitConditioning','H3PortraitSampler','VAEDecode','H3PortraitStillOutput','SaveImage'}
    if not required_types.issubset(types):
        raise ValueError('Lite still workflow is missing native H3 Ref2VA stages.')
    if any('QwenGenerate' in t or 'QwenModels' in t or t=='H3PortraitImageDirector' for t in types):
        raise ValueError('Lite still workflow must not use the removed separate image-model path.')

    ref_graph=json.loads((source/'workflows/H3_Reference_Video_Swap_Local.json').read_text())
    ref_types={n['type'] for n in ref_graph.get('nodes',[])}
    required_ref={'MiniMaxH3ReferencePack','MiniMaxH3ReferenceToVideo','H3ReferenceVideoSettings','H3ReferenceVideoDraftGate','H3ReferenceVideoCrop','H3PortraitExportVideo','H3PortraitSaveLastFrame'}
    if not required_ref.issubset(ref_types):
        raise ValueError('Local reference-video workflow is missing required Hearmeman/H3 stages.')
    pack=next(n for n in ref_graph['nodes'] if n['type']=='MiniMaxH3ReferencePack')
    vals=pack.get('widgets_values',[])
    if len(vals)<14 or vals[3]!='local' or vals[4] or vals[7]!='http://127.0.0.1:11434/v1' or vals[9]!='replacement':
        raise ValueError('Reference Pack workflow must default to local Ollama replacement mode with no API key.')

    for f in (source/'node').rglob('*.py'):
        ast.parse(f.read_text(),filename=str(f))
    print(f'H3 PORTRAIT SOURCE VERIFIED: {version} | {EXPECTED_PIPELINE}',flush=True)
    return cfg


def node_hashes(path):
    path=Path(path);values={}
    for name in NODE_FILES:
        f=path/name
        if not f.is_file():raise ValueError(f'Missing installed portrait node file: {f}')
        values[name]=digest(f)
    return values


def compare_node(source,node):
    expected=node_hashes(Path(source)/'node');observed=node_hashes(node)
    mismatched=[name for name in expected if expected[name]!=observed[name]]
    if mismatched:raise ValueError('Portrait node copy does not match this source: '+', '.join(mismatched))


def asset_hashes(source):
    source=Path(source)
    names=('settings.json','models.json','runtime.json','install_workflows.py',*WORKFLOW_FILES)
    return {name:digest(source/name) for name in names}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=HERE)
    p.add_argument('--node-copy',action='append',type=Path,default=[])
    p.add_argument('--write-manifest',type=Path);p.add_argument('--check-manifest',type=Path)
    p.add_argument('--source-revision',default=os.environ.get('H3_SOURCE_REVISION','local-uncommitted'))
    a=p.parse_args();verify_source(a.source)
    if not (a.source_revision=='local-uncommitted' or re.fullmatch(r'[0-9a-f]{40}',a.source_revision)):
        raise ValueError('Build source revision must be a full Git commit or local-uncommitted.')
    hashes=node_hashes(a.source/'node')
    for copy in a.node_copy:compare_node(a.source,copy)
    assets=asset_hashes(a.source)
    if a.check_manifest:
        saved=json.loads(a.check_manifest.read_text())
        if saved.get('version')!=EXPECTED_VERSION or saved.get('node_sha256')!=hashes:
            raise ValueError('Image release manifest does not match the installed portrait source.')
        if saved.get('source_revision')!=a.source_revision:raise ValueError('Image source revision and release manifest disagree.')
        if saved.get('assets_sha256')!=assets:raise ValueError('Workflow/catalog/source assets do not match the build manifest.')
        if saved.get('settings_sha256')!=digest(a.source/'settings.json'):raise ValueError('Image settings do not match the build manifest.')
    if a.write_manifest:
        a.write_manifest.write_text(json.dumps({'version':EXPECTED_VERSION,'pipeline':EXPECTED_PIPELINE,
            'source_revision':a.source_revision,'node_sha256':hashes,
            'settings_sha256':digest(a.source/'settings.json'),'assets_sha256':assets},indent=2)+'\n')
    print('H3 PORTRAIT RELEASE VERIFIED: '+EXPECTED_VERSION+' | source='+a.source_revision,flush=True)

if __name__=='__main__':main()

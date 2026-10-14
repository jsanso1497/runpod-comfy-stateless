#!/usr/bin/env python3
"""Prepare separate Qwen 2.1 / Krea 2 developer edit recipes on a ComfyUI Pod.

Only optional LoRAs and pinned developer workflows are downloaded. Existing base
models, custom nodes, personal workflows, secrets and running servers are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

VERSION = '1.0.0'
QREPO = 'ausboss/Qwen-Image-2.1-Consistency-LoRA'
OREPO = 'ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA'
KREPO = 'conradlocke/krea2-identity-edit'
QREV = '8f05b0fa027d517fa396fb31b71e0eaf48110e89'
OREV = '226c0a4edcee08328f2369f42f14b305e9671d9c'
KREV = '89e9e7a09ee2e5c9331e952063d79b1b8a703280'
KNODE = '86f886dac23013d88996e3a2e99093ba44d322fb'
ANODE = 'ade11ddf3d6fbd9fab22cd9e215fa9682fb9b3cb'
Q1500 = 'qwen-image-2.1-consistency.safetensors'
Q2000 = 'qwen-image-2.1-consistency-2000.safetensors'
OUTFIT = 'qwen-image-2.1-outfit-swap.safetensors'
KEDIT = 'krea2_identity_edit_v1_2.safetensors'
WEIGHTS = {
    'qwen': [(QREPO, QREV, Q1500, '4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63'),
             (QREPO, QREV, Q2000, 'a0bf043edc4695b1661a0e768e49a7fdff9a656b7d4313203d590e0262b64beb'),
             (OREPO, OREV, OUTFIT, '4ddfeac5695adaed1f0f78b23fda40919fd2ed6a4af790ced6c1e043dc4ae1f0')],
    'krea': [(KREPO, KREV, KEDIT, '6adf9a69cc9502d286db7b69964d37da7e9cfe4b05b4d004bc275f087d3fd3cf')],
}
BASE = {
    'qwen': ['diffusion_models/qwen_image_2.1_bf16.safetensors',
             'text_encoders/qwen3vl_8b_bf16.safetensors', 'vae/qwen_image_2.1_vae_bf16.safetensors'],
    'krea': ['diffusion_models/krea2_turbo_bf16.safetensors', 'diffusion_models/krea2_raw_bf16.safetensors',
             'text_encoders/qwen3vl_4b_bf16.safetensors', 'vae/qwen_image_vae.safetensors'],
}
PACKS = {
    'qwen': ('ausboss/ComfyUI-AusBoss', 'ComfyUI-AusBoss', ANODE),
    'krea': ('lbouaraba/comfyui-krea2edit', 'comfyui-krea2edit', KNODE),
}


def hf_url(repo, revision, filename):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A pinned source revision is required.')
    return f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'


def recipes(family):
    if family == 'krea':
        return {'editor': f'https://raw.githubusercontent.com/lbouaraba/comfyui-krea2edit/{KNODE}/workflows/krea2_identity_edit.json'}
    return {key: hf_url(repo, rev, 'workflows/' + filename) for key, repo, rev, filename in [
        ('consistency', QREPO, QREV, 'qwen-image-2.1-edit-consistency.json'),
        ('realign', QREPO, QREV, 'qwen-image-2.1-edit-realign.json'),
        ('outfit', OREPO, OREV, 'qwen-image-2.1-outfit-swap.json')]}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path, expected=None, limit=8 * 1024**2):
    """Publish only complete, verified downloads; never overwrite existing files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing a symlinked destination: {path}')
    if path.exists():
        if expected and digest(path) != expected:
            raise ValueError(f'Existing file has a different hash; left untouched: {path}')
        return
    for attempt in range(3):
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temp = Path(stream.name)
                req = Request(url, headers={'User-Agent': 'Workbench-Identity-Setup/1.0'})
                with urlopen(req, timeout=120) as response:
                    size = 0
                    while block := response.read(1024**2):
                        size += len(block)
                        if size > limit:
                            raise ValueError('Download exceeds the expected size limit.')
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if expected and digest(temp) != expected:
                raise ValueError(f'Download hash mismatch: {path.name}')
            temp.chmod(0o644)
            try:
                os.link(temp, path)
            except FileExistsError:
                if path.is_symlink() or digest(path) != digest(temp):
                    raise ValueError(f'Concurrent conflicting file left untouched: {path}')
            return
        except OSError:
            if attempt == 2:
                raise RuntimeError(f'Download failed: {path.name}. Check provider/network access.') from None
            time.sleep(attempt + 1)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def publish(path, data):
    """Keep user changes and write a separate factory copy on a conflict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(data).hexdigest()
    if path.is_symlink():
        raise ValueError(f'Refusing a symlinked file: {path}')
    if path.exists() and digest(path) != sha:
        path = path.with_name(path.stem + '_factory-' + sha[:12] + path.suffix)
    if path.is_symlink():
        raise ValueError(f'Refusing a symlinked factory file: {path}')
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(data)
    elif digest(path) != sha:
        raise ValueError(f'Conflicting factory copy left untouched: {path}')
    return path


def containers(value):
    if isinstance(value, dict):
        if isinstance(value.get('nodes'), list):
            yield value
        for child in value.values():
            yield from containers(child)
    elif isinstance(value, list):
        for child in value:
            yield from containers(child)


def nodes(graph):
    for part in containers(graph):
        yield from part['nodes']


def required_types(graph):
    parts = list(containers(graph))
    subgraphs = {str(g['id']) for g in parts[1:] if 'id' in g}
    return {n['type'] for n in nodes(graph)} - subgraphs - {'Note', 'MarkdownNote', 'PrimitiveNode', 'Reroute'}


def widget(node, name, index, value):
    values = node.get('widgets_values')
    if not isinstance(values, list) or len(values) <= index:
        raise ValueError(f'Unexpected upstream widget layout: {node.get("type")} / {name}')
    values[index] = value
    if 'widgets_values_named' in node:
        node['widgets_values_named'][name] = value


def replace_strings(value, replacements):
    if isinstance(value, dict):
        return {k: replace_strings(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_strings(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def adapt_qwen(original, selected):
    graph = replace_strings(original, {
        'qwen_image_2.1_int8_convrot.safetensors': Path(BASE['qwen'][0]).name,
        'qwen3vl_8b_int8_convrot.safetensors': Path(BASE['qwen'][1]).name,
        'Qwen-Image-2.1-Outfit-Swap-LoRA/': 'Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA/'})
    active = 0
    for n in nodes(graph):
        kind = n['type']
        if kind == 'UNETLoader':
            widget(n, 'unet_name', 0, Path(BASE['qwen'][0]).name)
            widget(n, 'weight_dtype', 1, 'default')
        elif kind == 'CLIPLoader':
            widget(n, 'clip_name', 0, Path(BASE['qwen'][1]).name)
        elif kind == 'VAELoader':
            widget(n, 'vae_name', 0, Path(BASE['qwen'][2]).name)
        elif kind == 'AUSBOSS_NODES_LoraLoader':
            rows = json.loads(n['widgets_values'][0])
            if not isinstance(rows, list) or any(not isinstance(r, dict) or 'name' not in r for r in rows):
                raise ValueError('Unexpected AusBoss LoRA format.')
            related = any('qwen-image-2.1-consistency' in r['name'] or 'qwen-image-2.1-outfit-swap' in r['name'] for r in rows)
            for row in rows:
                row['enabled'] = False
            if selected and related:
                if not any(r['name'] == selected for r in rows):
                    rows.append({'name': selected, 'strength': 1.0, 'triggers': ''})
                for row in rows:
                    if row['name'] == selected:
                        row.update(enabled=True, strength=1.0)
                        active += 1
            widget(n, 'loras', 0, json.dumps(rows))
        elif kind == 'LoraLoaderModelOnly':
            if selected is None:
                widget(n, 'strength_model', 1, 0.0)
                n['mode'] = 4
            else:
                widget(n, 'lora_name', 0, selected)
                widget(n, 'strength_model', 1, 1.0)
                active += 1
        elif kind == 'AUSBOSS_NODES_WorkflowNote':
            note = json.loads(n['widgets_values'][0])
            note['body'] = 'Workbench BF16 adaptation. One task LoRA at a time; speed LoRAs OFF. Original JSON is archived.\n\n' + note.get('body', '')
            for model in note.get('models', []):
                if model.get('name') == Path(BASE['qwen'][0]).name:
                    model['size'] = 'about 14.2 GB'
                elif model.get('name') == Path(BASE['qwen'][1]).name:
                    model['size'] = 'about 17.5 GB'
            widget(n, 'note', 0, json.dumps(note))
    if selected and active != 1:
        raise ValueError(f'Expected exactly one active Qwen task LoRA, found {active}.')
    needed = {'UNETLoader', 'CLIPLoader', 'VAELoader', 'TextEncodeQwenImage21', 'KSampler'}
    if needed - required_types(graph):
        raise ValueError('Unexpected Qwen developer graph; missing core nodes.')
    return graph


def adapt_krea(original, raw=False, two=False):
    model = 'krea2_raw_bf16.safetensors' if raw else 'krea2_turbo_bf16.safetensors'
    graph = replace_strings(original, {'krea2_turbo_fp8_scaled': Path(model).stem,
        'qwen3vl_4b_fp8_scaled': 'qwen3vl_4b_bf16', 'Krea2/' + KEDIT: KEDIT})
    by_id = {n['id']: n for n in graph['nodes']}
    expected = {53: 'KSampler', 55: 'UNETLoader', 56: 'CLIPLoader', 71: 'LoraLoaderModelOnly',
                79: 'Krea2EditModelPatch', 82: 'EmptySD3LatentImage', 84: 'Krea2EditGroundedEncode',
                85: 'Krea2EditGroundedEncode', 90: 'LoadImage', 92: 'VAEEncode'}
    if any(by_id.get(i, {}).get('type') != t for i, t in expected.items()):
        raise ValueError('Pinned Krea developer workflow differs from the expected layout.')
    widget(by_id[55], 'unet_name', 0, model)
    widget(by_id[55], 'weight_dtype', 1, 'default')
    widget(by_id[56], 'clip_name', 0, 'qwen3vl_4b_bf16.safetensors')
    widget(by_id[71], 'lora_name', 0, KEDIT)
    widget(by_id[71], 'strength_model', 1, 1.0)
    widget(by_id[53], 'steps', 2, 20 if raw else 10)
    widget(by_id[53], 'cfg', 3, 3.0 if raw else 1.0)
    by_id[90]['mode'] = by_id[92]['mode'] = 0 if two else 4
    if two:
        widget(by_id[84], 'prompt', 0, 'Replace the person in image 1 with the person in image 2. Keep the framing, pose, wardrobe, background and lighting from image 1. Use the face and hair from image 2.')
    elif raw:
        widget(by_id[84], 'prompt', 0, 'Remove the unwanted object. Reconstruct the background naturally and preserve the rest of the image.')
    widget(by_id[85], 'prompt', 0, '')
    patch_inputs = {p['name']: p.get('link') for p in by_id[79]['inputs']}
    sampler_inputs = {p['name']: p.get('link') for p in by_id[53]['inputs']}
    links = {r[0]: r for r in graph['links']}
    if (patch_inputs.get('target_latent') not in links or
        sampler_inputs.get('latent_image') not in links or
        links[patch_inputs['target_latent']][1:3] != links[sampler_inputs['latent_image']][1:3]):
        raise ValueError('Krea target_latent must come from the same output as sampler latent_image.')
    if by_id[79]['widgets_values'] != [4, 1, 'fit']:
        raise ValueError('Unexpected Krea reference settings.')
    for n in graph['nodes']:
        if n['type'] == 'Note' and n.get('title') == 'Sampling':
            n['widgets_values'][0] = ('Workbench BF16 preset: Raw, 20 steps, CFG 3. Empty negative is image-grounded. The old workflow note said 40 steps; this preset follows the current developer README.' if raw else 'Workbench BF16 preset: Turbo, 10 steps, CFG 1. Developer range: 8-12 steps.')
        if n['type'] == 'Note' and n.get('title') == 'Enable for two-image edits' and two:
            n['widgets_values'][0] = 'Second input is ENABLED in this preset. Image 1: scene/base. Image 2: identity reference. Both references also feed the empty negative.'
    return graph


def make_presets(family, originals):
    if family == 'qwen':
        result = [(name, adapt_qwen(originals[key], lora)) for name, key, lora in [
            ('01_Qwen_Consistency_1500_BF16', 'consistency', Q1500),
            ('02_Qwen_Consistency_2000_BF16', 'consistency', Q2000),
            ('03_Qwen_Realign_No_LoRA_BF16', 'realign', None),
            ('04_Qwen_Outfit_Swap_BF16', 'outfit', OUTFIT)]]
    else:
        result = [(f'{i:02d}_Krea_{mode}_{refs}_BF16', adapt_krea(originals['editor'], mode == 'Raw', refs == 'Scene_Person'))
                  for i, (mode, refs) in enumerate([('Turbo', 'Single_Image'), ('Turbo', 'Scene_Person'),
                                                    ('Raw', 'Single_Image'), ('Raw', 'Scene_Person')], 1)]
    for name, graph in result:
        graph.setdefault('extra', {})['workbench_identity'] = {'version': VERSION, 'name': name,
            'adapted_from_developer': True, 'gpu_tested': False}
    return result


def ensure_pack(comfy, family):
    repo, directory, revision = PACKS[family]
    root = comfy / 'custom_nodes'
    root.mkdir(parents=True, exist_ok=True)
    normalized = lambda s: re.sub(r'[-_]', '', s.lower())
    matches = [p for p in root.iterdir() if normalized(p.name) == normalized(directory)]
    if len(matches) > 1:
        raise ValueError('Duplicate companion node folders. Resolve these before setup: ' + ', '.join(map(str, matches)))
    target = matches[0] if matches else root / directory
    if target.exists():
        try:
            rev = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            # Workbench strips .git when building images; use its recorded source pin.
            provenance = Path(os.getenv('WB_ROOT', '/opt/workbench')) / 'build-provenance.json'
            key = 'ausboss' if family == 'qwen' else 'krea2edit'
            try:
                record = json.loads(provenance.read_text())['external_nodes'][key]
                if record['directory'] != directory or record['url'] != f'https://github.com/{repo}.git':
                    raise ValueError('Unexpected node provenance.')
                rev = record['commit']
            except (OSError, KeyError, ValueError):
                raise ValueError(f'Cannot verify existing {directory}. Left untouched; use the current Workbench image.') from None
        if not (target / '__init__.py').is_file():
            raise ValueError(f'Incomplete companion node pack: {target}')
        if rev != revision:
            raise ValueError(f'{directory} is not the verified revision. Left untouched; rebuild the current workspace or update the pack deliberately.')
        print('Reusing companion nodes at ' + rev[:12])
        return False
    temp = Path(tempfile.mkdtemp(prefix='.identity-nodes-', dir=root))
    try:
        for args in (['init', '-q', str(temp)], ['-C', str(temp), 'remote', 'add', 'origin', f'https://github.com/{repo}.git'],
                     ['-C', str(temp), 'fetch', '-q', '--depth', '1', 'origin', revision],
                     ['-C', str(temp), 'checkout', '-q', '--detach', 'FETCH_HEAD']):
            subprocess.run(['git', *args], check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp.rename(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['qwen', 'krea'])
    parser.add_argument('--comfyui', type=Path, default=Path('/opt/ComfyUI'))
    parser.add_argument('--data-root', type=Path, default=Path(os.getenv('WB_DATA_ROOT', '/workspace')))
    parser.add_argument('--workflow-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--require-live', action='store_true')
    args = parser.parse_args(argv)
    family = args.family
    if os.getenv('WB_WORKSPACE', family) != family:
        parser.error('Run each setup in its matching workspace. Do not mix Qwen and Krea.')
    if args.dry_run:
        print(json.dumps({'family': family, 'base_models_reused': BASE[family],
                          'weights': WEIGHTS[family], 'recipes': recipes(family), 'node_pack': PACKS[family]}, indent=2))
        return 0
    comfy, root = args.comfyui.expanduser().resolve(), args.data_root.expanduser().resolve()
    if not (comfy / 'main.py').is_file():
        raise ValueError(f'ComfyUI not found at {comfy}. Set --comfyui to the active installation.')
    missing = [p for p in BASE[family] if not (comfy / 'models' / p).is_file()]
    if missing:
        raise ValueError('Prepare the normal workspace base models first: ' + ', '.join(missing))
    output = args.workflow_dir or root / f'comfy-user/default/workflows/Identity Developer Presets/{family}/{VERSION}'
    archive = root / 'workbench/identity-developer-sources' / family / VERSION
    report = {'family': family, 'gpu_tested': False, 'upstream': [], 'weights': [], 'workflows': []}
    originals = {}
    for key, url in recipes(family).items():
        path = archive / 'originals' / (key + '.json')
        download(url, path)
        originals[key] = json.loads(path.read_text(encoding='utf-8'))
        report['upstream'].append({'url': url, 'path': str(path), 'sha256': digest(path)})
    prepared = make_presets(family, originals)
    installed = ensure_pack(comfy, family)
    for repo, rev, filename, sha in WEIGHTS[family]:
        print('Preparing ' + filename, flush=True)
        path = comfy / 'models/loras' / filename
        download(hf_url(repo, rev, filename), path, sha, 2 * 1024**3)
        report['weights'].append({'file': filename, 'sha256': sha, 'verified': True})
    schema = None
    try:
        with urlopen('http://127.0.0.1:8189/object_info', timeout=10) as response:
            schema = json.loads(response.read(32 * 1024**2))
        if not isinstance(schema, dict) or not schema:
            schema = None
    except (OSError, ValueError):
        pass
    absent = set()
    for name, graph in prepared:
        path = publish(output / (name + '.json'), (json.dumps(graph, indent=2) + '\n').encode())
        missing_nodes = sorted(required_types(graph) - set(schema)) if schema else None
        absent.update(missing_nodes or [])
        report['workflows'].append({'path': str(path), 'sha256': digest(path), 'missing_nodes': missing_nodes})
    report.update(node_types_checked_live=schema is not None, restart_required=installed or bool(absent))
    publish(archive / 'installation-report.json', (json.dumps(report, indent=2) + '\n').encode())
    print('Prepared workflows: ' + str(output))
    print('Unmodified developer originals and report: ' + str(archive))
    print('Select your own images in Load Image. Refresh the model lists and workflow browser.')
    if installed or absent:
        print('Restart the existing ComfyUI service, then refresh. Do NOT launch a second server.')
    if absent:
        print('Missing node types: ' + ', '.join(sorted(absent)))
    if schema is None:
        print('Live node check unavailable on internal port 8189. Check in ComfyUI before generation.')
    print('File preparation is not a GPU inference or quality test.')
    return 2 if args.require_live and (schema is None or absent) else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print('SETUP ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)

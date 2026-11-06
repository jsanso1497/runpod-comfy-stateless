"""Portable copies of the pinned developer graphs; no rewritten sampling graph."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import uuid

PACKAGE = Path(__file__).resolve().parent
PRESETS = (
    ('h3_still_text_to_image-8MP-NoTurboVersion.json', '00_H3_Still_8MP_NoTurbo.json', 'hq'),
    ('h3_still_text_to_image.json', '01_H3_Still_Turbo_2_5MP.json', 'turbo'),
    ('Edit_Workflow_example.json', '02_H3_Still_Edit_README_2_5MP.json', 'edit_readme'),
    ('Edit_Workflow_example.json', '03_H3_Still_Edit_Original_2MP.json', 'edit_original'),
    ('Edit_Workflow_example.json', '04_H3_Still_Two_References_CORE.json', 'edit_two'),
)
LOADER_FIELDS = {
    'UNETLoader': ('unet_name', 'h3_fl2va_int8'),
    'CLIPLoader': ('clip_name', 'h3_text_encoder'),
    'VAELoader': ('vae_name', 'h3_video_vae'),
    'LoraLoaderModelOnly': ('lora_name', 'h3_turbo_v4_600_ema'),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def get_node(graph: dict, kind: str) -> dict:
    nodes = [n for n in graph['nodes'] if n['type'] == kind]
    if len(nodes) != 1:
        raise ValueError(f'Expected exactly one {kind}, found {len(nodes)}.')
    return nodes[0]


def set_widget(node: dict, name: str, value) -> None:
    # These upstream files store named values in the same order as the widget array.
    # Refuse unexpected layouts rather than silently altering the wrong parameter.
    named = node.get('widgets_values_named', {})
    if name not in named:
        raise ValueError(f'Missing named widget {node["type"]}.{name}.')
    index = list(named).index(name)
    positional = node.get('widgets_values', [])
    if index >= len(positional) or positional[index] != named[name]:
        raise ValueError(f'Inconsistent widget serialization: {node["type"]}.{name}.')
    named[name] = value
    positional[index] = value


def portable_graph(source: dict, output_name: str, profile: str, models: list[dict]) -> dict:
    graph = copy.deepcopy(source)
    assets = {row['id']: row for row in models}
    changes = []
    for kind, (field, asset_id) in LOADER_FIELDS.items():
        node = get_node(graph, kind)
        asset = assets[asset_id]
        filename = Path(asset['destination']).name
        old = node['widgets_values_named'][field]
        if old.replace('\\', '/').split('/')[-1] != filename:
            raise ValueError(f'Unexpected developer model selection: {old}.')
        set_widget(node, field, filename)
        node.setdefault('properties', {})['models'] = [{
            'name': filename,
            'url': f'https://huggingface.co/{asset["repo"]}/resolve/{asset["revision"]}/{asset["file"]}',
            'directory': asset['destination'].split('/')[0],
        }]
        if old != filename:
            changes.append(f'{kind}: normalized Windows subfolder to {filename}.')
    changes.append('Loader download metadata now matches the selected pinned model files.')
    if profile == 'hq':
        selector = get_node(graph, 'ResolutionSelector')
        set_widget(selector, 'multiple', 32)
        changes.append('ResolutionSelector multiple 16 -> 32, as required by the developer README.')
    if profile in ('edit_readme', 'edit_two'):
        set_widget(get_node(graph, 'AILab_ImageResize'), 'megapixels', 2.5)
        changes.append('Resize 2 -> 2.5 MP to follow README; separate original-2-MP preset retained.')
    graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-still-2026-11-05.1/' + output_name))
    graph.setdefault('extra', {})['h3_still_workspace'] = {
        'release': '2026-11-05.1',
        'profile': profile,
        'source_commit': '10d5171f6e45b0757a03df977f53a660e060c33d',
        'changes': changes,
        'prompt_enhancer': False,
        'gpu_tested': False,
    }
    from core_workflows import native_resize, add_second_reference, finalize
    if profile.startswith('edit'):
        graph = native_resize(graph)
        changes.append('Use built-in ImageScaleToTotalPixels; no padding or color-node dependency.')
        if profile == 'edit_two':
            graph = add_second_reference(graph)
            changes.append('Second reference has a separate LoadImage and native resize path.')
    graph = finalize(graph, output_name)
    validate_graph(graph, profile)
    return graph


def validate_graph(graph: dict, profile: str) -> None:
    if profile not in {p[2] for p in PRESETS}:
        raise ValueError('Unknown preset profile.')
    nodes = {n['id']: n for n in graph['nodes']}
    if len(nodes) != len(graph['nodes']):
        raise ValueError('Duplicate node IDs.')
    links = {link[0]: link for link in graph['links']}
    if len(links) != len(graph['links']):
        raise ValueError('Duplicate link IDs.')
    for link in links.values():
        if len(link) != 6:
            raise ValueError('Unsupported link layout.')
        lid, source, output_index, target, input_index, _ = link
        if source not in nodes or target not in nodes:
            raise ValueError('Dangling graph link.')
        outputs, inputs = nodes[source].get('outputs', []), nodes[target].get('inputs', [])
        if output_index >= len(outputs) or input_index >= len(inputs):
            raise ValueError('Graph link references a missing port.')
        if inputs[input_index].get('link') != lid or lid not in (outputs[output_index].get('links') or []):
            raise ValueError('Graph link has inconsistent reciprocal references.')
    for node in nodes.values():
        for port in node.get('inputs', []):
            if port.get('link') is not None and port['link'] not in links:
                raise ValueError('Input references a missing link.')
        for port in node.get('outputs', []):
            if any(lid not in links for lid in (port.get('links') or [])):
                raise ValueError('Output references a missing link.')

    def origin(node, name):
        port = next(p for p in node['inputs'] if p['name'] == name)
        if port.get('link') not in links:
            raise ValueError(f'Missing required connection {name}.')
        edge = links[port['link']]
        return nodes[edge[1]], edge[2]

    still = get_node(graph, 'FizgigH3StillLatent')
    sampler = get_node(graph, 'SamplerCustomAdvanced')
    decoder = get_node(graph, 'FizgigH3StillDecode')
    save = get_node(graph, 'SaveImage')
    guider = get_node(graph, 'BasicGuider')
    condition_kind = 'MiniMaxH3ReferenceToVideo' if profile.startswith('edit') else 'MiniMaxH3ImageToVideo'
    conditioning = get_node(graph, condition_kind)
    if origin(sampler, 'latent_image') != (still, 0):
        raise ValueError('Sampler must use the Fizgig single-frame latent.')
    if origin(save, 'images') != (decoder, 0):
        raise ValueError('Saved image must come from Fizgig Still Decode.')
    if origin(decoder, 'samples') != (sampler, 0):
        raise ValueError('Still decoder must receive the sampled latent.')
    if origin(guider, 'conditioning') != (conditioning, 0):
        raise ValueError('Native H3 conditioning must feed BasicGuider.')
    for port in conditioning.get('outputs', []):
        if port.get('type') == 'LATENT' and port.get('links'):
            raise ValueError('The native video latent must stay disconnected.')
    for dimension in ('width', 'height'):
        if origin(still, dimension) != origin(conditioning, dimension):
            raise ValueError('Still and conditioning dimensions must share the same source.')
    if get_node(graph, 'KSamplerSelect')['widgets_values_named']['sampler_name'] != 'er_sde':
        raise ValueError('Unexpected sampler.')
    schedule = get_node(graph, 'BasicScheduler')['widgets_values_named']
    expected_steps = 50 if profile == 'hq' else 20
    if (schedule['scheduler'], schedule['steps'], schedule['denoise']) != ('simple', expected_steps, 1):
        raise ValueError('Sampling schedule differs from the developer preset.')
    lora = get_node(graph, 'LoraLoaderModelOnly')['widgets_values_named']
    if lora['strength_model'] != (0 if profile == 'hq' else 0.38):
        raise ValueError('Turbo strength differs from the developer preset.')
    if profile.startswith('edit'):
        resizers = [n for n in graph['nodes'] if n['type'] == 'ImageScaleToTotalPixels']
        if len(resizers) != (2 if profile == 'edit_two' else 1):
            raise ValueError('Incorrect native reference resize count.')
        expected_mp = 2 if profile == 'edit_original' else 2.5
        for n in resizers:
            resize = n['widgets_values_named']
            if resize != {'upscale_method':'lanczos','megapixels':expected_mp,'resolution_steps':32}:
                raise ValueError('Incorrect native edit resize settings.')
        if any(n['type'].startswith('AILab_') for n in graph['nodes']):
            raise ValueError('Legacy resize/color dependency must not be present.')
        reference_ports = [p for p in conditioning['inputs']
                           if p['name'].startswith('ref_images.') and p.get('link') is not None]
        if len(reference_ports) != len(resizers):
            raise ValueError('Reference image count does not match the resizers.')
        if '<Picture 1>' not in conditioning['widgets_values_named']['prompt']:
            raise ValueError('Reference prompt must retain the Picture 1 marker.')
    else:
        size = get_node(graph, 'ResolutionSelector')['widgets_values_named']
        if size['megapixels'] != (8 if profile == 'hq' else 2.5) or size['multiple'] != 32:
            raise ValueError('Incorrect text-to-image resolution settings.')


def build_workflows(source_dir: Path, output_dir: Path, models: list[dict]) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for source_name, destination, profile in PRESETS:
        source_bytes = (source_dir / source_name).read_bytes()
        graph = portable_graph(json.loads(source_bytes), destination, profile, models)
        output = json.dumps(graph, indent=2, ensure_ascii=False).encode('utf-8') + b'\n'
        (output_dir / destination).write_bytes(output)
        report.append({'source': source_name, 'source_git_blob': blob_sha(source_bytes),
                       'file': destination, 'profile': profile, 'sha256': hashlib.sha256(output).hexdigest(),
                       'changes': graph['extra']['h3_still_workspace']['changes']})
    return report

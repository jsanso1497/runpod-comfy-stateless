"""Editable workflow records and safe metadata. Never reads environment secrets.

A chain export restores its controller, not a fake native execution canvas.
Internal job.api.json remains the exact worker graph for diagnostics. Input media
and model files are not embedded. These records do not certify identical output.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import uuid
from .config import ROOT, atomic_json, boolean, data_root

LOG = logging.getLogger(__name__)
MAX_METADATA_BYTES = 16 * 1024 * 1024
SIMPLE_PROMPT = ('Replace the selected person with the reference character. '
                 'Preserve source performance, scene and audio timing.')


def enabled():
    return boolean('H3_SWAP_SAVE_METADATA', True)


def json_text(value):
    text = json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    if len(text.encode('utf-8')) > MAX_METADATA_BYTES:
        raise ValueError('Workflow record exceeds the 16 MiB metadata limit')
    return text



def prompt_for_metadata(prompt):
    """Copy an API prompt without execution-only cache fingerprints.

    ComfyUI intentionally uses NaN in node-level ``is_changed`` values to
    force loop/cache reevaluation. Those values are not workflow inputs and
    must remain untouched in the live execution graph. Drop them only from
    the export copy. Do not replace arbitrary NaNs with null: a nonfinite
    actual input is still an error under the strict JSON serializer.
    """
    exported = copy.deepcopy(prompt)
    for node in exported.values():
        if isinstance(node, dict) and 'class_type' in node:
            node.pop('is_changed', None)
    return exported


def media_metadata(prompt=None, extra_pnginfo=None):
    if not enabled():
        return {}
    result = {}
    workflow = extra_pnginfo.get('workflow') if isinstance(extra_pnginfo, dict) else None
    if isinstance(workflow, dict) and isinstance(workflow.get('nodes'), list):
        result['workflow'] = copy.deepcopy(workflow)
    if isinstance(prompt, dict):
        result['prompt'] = prompt_for_metadata(prompt)
    json_text(result)
    return result


def sidecars(media_path, metadata):
    """Use filename.ext.workflow.json so different containers cannot collide."""
    media_path = Path(media_path)
    for key, suffix in [('workflow', '.workflow.json'), ('prompt', '.api.json')]:
        if key in metadata:
            atomic_json(Path(str(media_path) + suffix), metadata[key])


def _input_name(value, config):
    path, root = Path(value).resolve(), Path(config['input_root']).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Never silently map a different input to the same basename. Public
        # control nodes use input-root-relative file selectors.
        raise ValueError('Reference is outside the input folder. Copy it to input and '
                         'reselect it before creating a reusable controller workflow.')


def control_workflow(config, *, resume=False, factory_root=None):
    """Reconstruct the public controller from immutable saved run configuration."""
    root = Path(factory_root or ROOT) / 'workflows'
    dual = config.get('reference_mode', 'single') == 'face_body'
    filename = '09_H3_LowMemory_Face_Body_Chain.json' if dual else '06_H3_LowMemory_Chain_Control.json'
    graph = json.loads((root / filename).read_text())
    classname = 'H3SwapDiskChainFaceBody' if dual else 'H3SwapDiskChain'
    matches = [n for n in graph['nodes'] if n['type'] == classname]
    if len(matches) != 1:
        raise ValueError('Cannot locate the chain controller in its factory workflow')
    node = matches[0]
    values = node['widgets_values_named']
    new_name = config['run_name'][:46] + '_copy_' + uuid.uuid4().hex[:8]
    changes = dict(action='Resume' if resume else 'Start new run',
        run_name=config['run_name'] if resume else new_name,
        source_video=_input_name(config['source'], config),
        duration_seconds=config['duration_seconds'], start_seconds=config['start_seconds'],
        resolution=config['resolution'], window_frames=str(config['window_frames']),
        target_person=config['target_person'], selection=config['selection'],
        selection_index=config['selection_index'], mask_expansion=config['mask_expansion'],
        seed=config['seed'], review_mode=config['review_mode'], max_retries=config['max_retries'])
    if dual:
        changes.update(face_image=_input_name(config['reference'], config),
                       body_image=_input_name(config['body_reference'], config),
                       additional_instructions=config['prompt'])
    else:
        changes.update(character_image=_input_name(config['reference'], config), prompt=config['prompt'])
    if not changes.keys() <= values.keys():
        raise ValueError('Controller schema changed; refusing an incomplete recovery')
    values.update(changes)
    node['widgets_values'] = list(values.values())
    node['title'] = ('RESUME SAVED RUN' if resume else 'REPEAT WITH NEW RUN NAME') + ': ' + config['run_name']
    note = ('Recovered controller for '+config['run_name']+'. Precision profile: '+config.get('model_profile','baseline')+'. Use the matching image. '+
            ('Resume retains accepted chunks. It does not rerender a completed run.' if resume else
             'Start new run uses a new name. Confirm that your input files and models are still the intended versions.')+
            ' This is a settings recovery, not embedded media or guaranteed pixel-identical reproduction. '
            'The exact worker graph is retained separately in each attempt/job.api.json. '
            'Do not apply changed settings to Resume; use the repeat workflow instead.')
    for n in graph['nodes']:
        if n['type'] in ('Note', 'MarkdownNote'):
            n['widgets_values'] = [note]
            n['widgets_values_named'] = {'text': note}
    graph['id'] = str(uuid.uuid4())
    graph.setdefault('extra', {})['h3_recovery'] = {
        'source_run': config['run_name'], 'kind': 'resume' if resume else 'repeat',
        'source_sha256': config.get('source_sha256'),
        'reference_sha256': config.get('reference_sha256'),
        'body_reference_sha256': config.get('body_reference_sha256'),
        'worker_recipe_sha256': config.get('workflow_digest'),
        'model_profile': config.get('model_profile','baseline'),
        'assets_embedded': False}
    # Controller prompt matches the editable graph; not the internal worker job.
    api = {str(node['id']): {'class_type': classname, 'inputs': dict(values),
                            '_meta': {'title': node['title']}}}
    return graph, api


def export_run(run, *, factory_root=None):
    """Add sidecars only. Never edits config, state, checkpoints or video pixels."""
    if not enabled():
        return {}
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    config_hash = hashlib.sha256(json_text(config).encode()).hexdigest()
    receipt = run/'recovery.manifest.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('config_sha256') != config_hash:
        raise ValueError('Saved run configuration changed since workflow export; refusing stale recovery')
    result = {}
    for mode in ('resume', 'repeat'):
        wf_path, api_path = run / (mode + '.workflow.json'), run / (mode + '.api.json')
        if not wf_path.exists() or not api_path.exists():
            graph, api = control_workflow(config, resume=mode == 'resume', factory_root=factory_root)
            atomic_json(wf_path, graph)
            atomic_json(api_path, api)
        result[mode] = {'workflow': json.loads(wf_path.read_text()),
                        'prompt': json.loads(api_path.read_text())}
    atomic_json(receipt, {'config_sha256': config_hash, 'run_name': config['run_name'],
                         'type': 'controller settings, not a raw native sampling graph'})
    return result


def ffmetadata(path, metadata):
    """FFmetadata escaping, avoiding huge/shell-interpreted command arguments."""
    def escape(text):
        return (text.replace('\\', '\\\\').replace('\n', '\\\n')
                .replace('=', '\\=').replace(';', '\\;').replace('#', '\\#'))
    body = ';FFMETADATA1\n'
    for key in ('workflow', 'prompt'):
        if key in metadata:
            body += key + '=' + escape(json_text(metadata[key])) + '\n'
    path = Path(path)
    path.write_text(body, encoding='utf-8')
    return path


def journal_request(data):
    """On-prompt observer. No graph rewriting, sampling or execution side effects."""
    if not enabled() or not isinstance(data, dict):
        return data
    prompt = data.get('prompt')
    if not isinstance(prompt, dict) or not any(
        isinstance(n, dict) and ('H3' in str(n.get('class_type', '')) or 'MiniMax' in str(n.get('class_type', '')))
        for n in prompt.values()):
        return data
    extra = data.get('extra_data')
    extra = extra if isinstance(extra, dict) else {}
    info=extra.get('extra_pnginfo')
    wf=info.get('workflow') if isinstance(info,dict) else None
    restored=((wf or {}).get('extra') or {}).get('h3_recovery') if isinstance(wf,dict) else None
    if isinstance(restored,dict) and 'model_profile' in restored:
        from .precision import profile
        if restored['model_profile']!=profile():
            raise ValueError('Recovered workflow belongs to a different model precision image. Open the matching image or start a new factory workflow.')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = data_root() / 'output/h3_character_swap/workflow_history' / (stamp + '_' + uuid.uuid4().hex[:8])
    try:
        metadata = media_metadata(prompt, extra.get('extra_pnginfo'))
        folder.mkdir(parents=True, exist_ok=False)
        sidecars(folder / 'submitted', metadata)
        atomic_json(folder / 'receipt.json', {'stage': 'submitted, not proof of completed generation',
                    'has_editable_workflow': 'workflow' in metadata,
                    'saved_at_utc': stamp, 'input_assets_embedded': False})
        print('[H3 WORKFLOW SAVED] '+str(folder), flush=True)
    except (OSError, ValueError) as exc:
        # Do not turn a metadata failure into a mystery generation failure.
        LOG.warning('Could not journal H3 workflow: %s', exc)
    return data


def register_journal():
    from server import PromptServer
    instance = PromptServer.instance
    if not getattr(instance, '_h3_workflow_journal_installed', False):
        instance.add_on_prompt_handler(journal_request)
        instance._h3_workflow_journal_installed = True

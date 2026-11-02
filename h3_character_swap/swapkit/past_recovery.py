"""Recover recorded settings, never guess a workflow from video pixels."""
from __future__ import annotations
import copy
import datetime as dt
import json
from pathlib import Path
import urllib.request
import uuid
from .config import data_root, atomic_json
from .independent_media import local_media, video_info
from .recovery import MAX_METADATA_BYTES, json_text, prompt_for_metadata, export_run


def read_record(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_METADATA_BYTES:
        raise ValueError('Missing or oversized workflow record.')
    value=json.loads(path.read_text(encoding='utf-8'));json_text(value);return value


def valid_canvas(value):
    return isinstance(value,dict) and isinstance(value.get('nodes'),list) and bool(value['nodes'])


def live_history():
    # This is the private, local Comfy server. No remote URLs or authentication
    # tokens are accepted from an input widget.
    with urllib.request.urlopen('http://127.0.0.1:8189/history?max_items=300',timeout=15) as response:
        blob=response.read(64*1024**2+1)
    if len(blob)>64*1024**2: raise ValueError('Live history is too large; use saved sidecars instead.')
    return json.loads(blob)


def _output_matches(value, relative):
    if isinstance(value,dict):
        if (value.get('type')=='output' and isinstance(value.get('filename'),str)
            and isinstance(value.get('subfolder',''),str)):
            candidate=(Path(value.get('subfolder',''))/value['filename']).as_posix()
            if candidate==relative: return True
        return any(_output_matches(v,relative) for v in value.values())
    if isinstance(value,list): return any(_output_matches(v,relative) for v in value)
    return False


def _from_history(history, path, root):
    try:relative=path.relative_to(root/'output').as_posix()
    except ValueError:return None
    candidates=[]
    for item in history.values() if isinstance(history,dict) else []:
        if not isinstance(item,dict) or not _output_matches(item.get('outputs',{}),relative): continue
        record=item.get('prompt',[])
        if not isinstance(record,list) or len(record)<4: continue
        prompt=record[2];extra=record[3] if isinstance(record[3],dict) else {}
        info=extra.get('extra_pnginfo',{});wf=info.get('workflow') if isinstance(info,dict) else None
        result={}
        if valid_canvas(wf):result['workflow']=copy.deepcopy(wf)
        if isinstance(prompt,dict):result['prompt']=prompt_for_metadata(prompt)
        if result:json_text(result);candidates.append(result)
    if len({json_text(v) for v in candidates})>1:
        raise ValueError('Several different histories claim this filename. Recovery refuses to guess.')
    return candidates[0] if candidates else None



def saved_histories(root):
    """Read explicit pre-restart backups, not arbitrary files or external paths."""
    folder=Path(root)/'h3-swap/history-backups'
    if not folder.is_dir() or folder.is_symlink(): return
    paths=sorted((p for p in folder.glob('*.json') if p.is_file() and not p.is_symlink()),
                 key=lambda p:p.name,reverse=True)[:20]
    for path in paths:
        if path.stat().st_size>64*1024**2: continue
        try: yield path,json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError): continue


def recover_video(video, storage=None, history_provider=live_history):
    root=Path(storage or data_root()).resolve();path=local_media(str(video),root)
    # Existing sidecars are the most direct, persistent record.
    wf=Path(str(path)+'.workflow.json');api=Path(str(path)+'.api.json')
    if wf.exists():
        canvas=read_record(wf)
        if not valid_canvas(canvas):raise ValueError('Saved sidecar is not an editable workflow.')
        return {'source':'existing sidecar','workflow':str(wf),
                'api':str(api) if api.exists() else None,'instruction':'Drag the .workflow.json into ComfyUI.'}
    # Chain finals/chunks can recover their controller from the immutable config.
    chain=root/'output/h3_character_swap/disk_chain_v2'
    if path.is_relative_to(chain):
        relative=path.relative_to(chain)
        if len(relative.parts)>1:
            run=chain/relative.parts[0]
            if (run/'config.json').is_file():
                saved=export_run(run)
                if not saved:raise ValueError('Enable H3_SWAP_SAVE_METADATA=1 to export chain recovery records.')
                return {'source':'saved chain config','workflow':str(run/'repeat.workflow.json'),
                        'resume':str(run/'resume.workflow.json'),
                        'instruction':'Repeat starts a NEW run; Resume continues the recorded run. This is the controller, not a reconstructed native canvas.'}
    info=video_info(path,False);tags=info.get('format',{}).get('tags',{})
    tags={k.lower():v for k,v in tags.items()}
    metadata={}
    for k in ('workflow','prompt'):
        if k not in tags:continue
        text=tags[k]
        if not isinstance(text,str) or len(text.encode())>MAX_METADATA_BYTES:
            raise ValueError('Embedded workflow record is oversized.')
        value=json.loads(text)
        if k=='workflow' and not valid_canvas(value):continue
        if k=='prompt' and isinstance(value,dict):value=prompt_for_metadata(value)
        metadata[k]=value
    method='embedded video metadata'
    if 'workflow' not in metadata:
        candidates=[]
        try:found=_from_history(history_provider(),path,root)
        except (OSError,TimeoutError):found=None
        if found:candidates.append(('matching surviving ComfyUI execution history',found))
        for backup,history in saved_histories(root):
            found=_from_history(history,path,root)
            if found:candidates.append(('saved ComfyUI history backup: '+backup.name,found))
        if len({json_text(v) for _,v in candidates})>1:
            raise ValueError('Different live/backed-up histories claim this filename. Recovery refuses to guess.')
        if candidates:
            method,found=candidates[0];metadata.update(found)
    if not metadata and api.exists():metadata={'prompt':prompt_for_metadata(read_record(api))};method='saved API sidecar'
    if not metadata:
        raise ValueError('No recorded workflow found in the video, sidecars, chain config or surviving/backed-up history. '
                         'The old file was saved without those records. Its exact workflow cannot be inferred from pixels. '
                         'Use a workflow you saved manually; this update fixes NEW saves.')
    json_text(metadata)
    directory=root/'output/h3_character_swap/recovered'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    result={'source':method,'original_video':str(path)}
    for key,suffix in [('workflow','.workflow.json'),('prompt','.api.json')]:
        if key in metadata:
            target=directory/('generation'+suffix);atomic_json(target,metadata[key]);result[key]=str(target)
    result['instruction']=('Drag the .workflow.json into ComfyUI. Original media/models are still required.'
        if 'workflow' in metadata else 'Only the executable API graph survived, not the original editable layout. Import the .api.json separately or use it for reconstruction.')
    atomic_json(directory/'receipt.json',result)
    return result

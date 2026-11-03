"""Persist native Comfy history so its EXISTING right-click Open workflow works.

The pinned frontend reads /api/jobs/<id> -> workflow.extra_data.extra_pnginfo.workflow
for output assets; it does not fall back to MP4 tags when that job lacks a canvas.
This module preserves that real queue history, not a fabricated workflow. It never
requeues a job, rewrites live prompts, or changes how the sampler runs.
"""
from __future__ import annotations
import copy
import json
import logging
from pathlib import Path
import uuid
from .config import atomic_json, data_root
from .recovery import enabled, media_metadata, json_text

LOG=logging.getLogger(__name__)
MAX_RECORD=16*1024**2
MAX_BACKUP=256*1024**2
MAX_RESTORE=300
MAX_RESTORE_BYTES=128*1024**2


def job_id(value):
    try:
        canonical=str(uuid.UUID(str(value)))
        if canonical!=value: raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise ValueError('Invalid Comfy job id') from None
    return canonical


def clean_record(ident, record):
    """Export a completed queue record with ONLY workflow-related extra data."""
    job_id(ident)
    if not isinstance(record,dict):raise ValueError('Not a Comfy history record')
    p=record.get('prompt')
    if not isinstance(p,(list,tuple)) or len(p)<5 or p[1]!=ident or not isinstance(p[2],dict):
        raise ValueError('Not a matching Comfy history record')
    extra=p[3] if isinstance(p[3],dict) else {}
    metadata=media_metadata(p[2],extra.get('extra_pnginfo'))
    if 'workflow' not in metadata:
        raise ValueError('No editable workflow was submitted. Not inventing one.')
    if not any('H3' in str(n.get('class_type','')) or 'MiniMax' in str(n.get('class_type',''))
               for n in p[2].values() if isinstance(n,dict)):
        raise ValueError('Not an H3 job')
    safe_extra={'extra_pnginfo':{'workflow':metadata['workflow']}}
    if isinstance(extra.get('create_time'),(int,float)):
        safe_extra['create_time']=extra['create_time']
    status=copy.deepcopy(record.get('status') or {})
    if not isinstance(status,dict):raise ValueError('Invalid history status')
    if not status.get('completed') and status.get('status_str')!='error':
        raise ValueError('Only terminal history is recoverable; unfinished jobs are not restored')
    # Diagnostic error payloads can contain arbitrary tensors/nonfinite values.
    # Preserve timing/type, not those runtime inputs or potentially sensitive data.
    messages=[]
    for entry in status.get('messages',[]):
        if not isinstance(entry,(list,tuple)) or len(entry)!=2 or not isinstance(entry[1],dict):continue
        payload={k:v for k,v in entry[1].items() if k in ('timestamp','prompt_id','node_id','node_type','exception_type')}
        messages.append([entry[0],payload])
    status['messages']=messages
    result={'prompt':[p[0],ident,metadata['prompt'],safe_extra,copy.deepcopy(p[4])],
            'outputs':copy.deepcopy(record.get('outputs',{})),'status':status}
    for key in ('meta',):
        if key in record:result[key]=copy.deepcopy(record[key])
    json_text(result)
    return result


def persist(ident, record, root=None):
    if not enabled():return False
    root=Path(root or data_root());saved=clean_record(ident,record)
    atomic_json(root/'h3-swap/job-history'/(ident+'.json'),saved)
    return True


def restore(queue, root=None):
    """Read bounded, completed history only. In-progress queues are never restored."""
    if not enabled():return 0
    root=Path(root or data_root());items={};used=0
    files=sorted((root/'h3-swap/job-history').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in files[:MAX_RESTORE]:
        if p.is_symlink() or p.stat().st_size>MAX_RECORD:continue
        used+=p.stat().st_size
        if used>MAX_RESTORE_BYTES:break
        try:
            ident=p.stem;r=clean_record(ident,json.loads(p.read_text()))
            if r['status'].get('completed') or r['status'].get('status_str')=='error':items[ident]=r
        except (OSError,ValueError,TypeError,KeyError) as exc:LOG.warning('Skipping unusable H3 history %s: %s',p.name,exc)
    # Import the backups the user already made, when they contain real canvases.
    for p in sorted((root/'h3-swap/history-backups').glob('comfy-history-*.json'),reverse=True)[:8]:
        if len(items)>=MAX_RESTORE:break
        if p.is_symlink() or p.stat().st_size>MAX_BACKUP:continue
        try:
            records=json.loads(p.read_text())
            if not isinstance(records,dict):continue
            for ident,record in reversed(list(records.items())):
                if len(items)>=MAX_RESTORE:break
                if ident in items:continue
                try:
                    r=clean_record(ident,record)
                    size=len(json_text(r))
                    if used+size>MAX_RESTORE_BYTES:break
                    used+=size;items[ident]=r
                except (ValueError,TypeError,KeyError):continue
        except (OSError,ValueError):LOG.warning('Cannot read history backup %s',p.name)
    count=0
    # Oldest first matches native history insertion order. Never replace a live job.
    ordered=sorted(items.items(),key=lambda item:item[1]['prompt'][3].get('create_time',0))
    with queue.mutex:
        for ident,record in ordered:
            if ident not in queue.history:
                queue.history[ident]=record;count+=1
    print(f'[H3 HISTORY] Restored {count} completed jobs with editable workflows. No jobs queued.',flush=True)
    return count


def attach(queue, root=None):
    if getattr(queue,'_h3_persistent_history',False):return
    original=queue.task_done
    original_history=queue.get_history
    def history(*args, **kwargs):
        result=original_history(*args, **kwargs)
        ident=kwargs.get('prompt_id', args[0] if args else None)
        # The asset library can reference a job older than the restored recent
        # list. Load that one exact archive lazily, without injecting or queuing it.
        if not ident or result or not enabled():return result
        try:
            job_id(ident)
            path=Path(root or data_root())/'h3-swap/job-history'/(ident+'.json')
            if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_RECORD:return result
            record=clean_record(ident,json.loads(path.read_text()))
            mapper=kwargs.get('map_function', args[3] if len(args)>3 else None)
            return {ident:mapper(record) if mapper else record}
        except (OSError,ValueError,KeyError,TypeError):return result
    queue.get_history=history
    def done(item_id, history_result, status, process_item=None):
        ident=None
        with queue.mutex:
            item=queue.currently_running.get(item_id)
            if item:ident=item[1]
        result=original(item_id,history_result,status,process_item)
        if ident and enabled():
            try:
                record=queue.get_history(prompt_id=ident).get(ident)
                if record:persist(ident,record,root)
            except (ValueError,TypeError,KeyError,OSError) as exc:
                # Do not fail an expensive successful render because of history I/O.
                LOG.warning('H3 completed-job workflow persistence: %s',exc)
        return result
    queue.task_done=done;queue._h3_persistent_history=True
    restore(queue,root)


def register():
    from server import PromptServer
    server=PromptServer.instance
    if getattr(server,'_h3_history_registration',False):return
    server._h3_history_registration=True
    if getattr(server,'prompt_queue',None) is not None:
        attach(server.prompt_queue)
    else:
        async def on_startup(app):attach(server.prompt_queue)
        server.app.on_startup.append(on_startup)

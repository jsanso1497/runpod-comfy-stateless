"""Start/review/resume the controller without blocking ComfyUI's worker thread."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
from .config import ROOT, data_root, public_child_env
from .disk_common import ChainError, atomic_json, read_json
from .disk_chain import create_run, resolve_run, root_dir
from .disk_storage import verify_receipt

ACTIONS=('Start new run','Status','Resume','Approve candidate and continue','Retry candidate','Stop')


def is_running(run):
    try:
        pid=int((Path(run)/'controller.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return 'swapkit.disk_chain' in cmd and str(Path(run).resolve()) in cmd
    except (OSError,ValueError): return False


def launch(run):
    run=Path(run).resolve()
    if is_running(run): raise ChainError('This chain is already running')
    # Refuse starting another runner even while the new one is waiting for the
    # launcher prompt to finish. The controller also takes a POSIX file lock.
    for sibling in run.parent.iterdir():
        if sibling.is_dir() and sibling!=run and is_running(sibling):
            raise ChainError('Another chain is active: '+sibling.name)
    (run/'STOP').unlink(missing_ok=True)
    with (run/'controller.log').open('a',buffering=1) as log:
        p=subprocess.Popen([sys.executable,'-m','swapkit.disk_chain','--run',str(run)],
            cwd=str(ROOT),env=public_child_env(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (run/'controller.pid').write_text(str(p.pid)+'\n')
    return p.pid


def act(output_root,run_name,action):
    run=resolve_run(output_root,run_name)
    if not (run/'state.json').is_file(): raise ChainError('Run not found: '+run_name)
    state=read_json(run/'state.json')
    if action=='Status': return summarize(run)
    if action=='Stop':
        (run/'STOP').write_text('stop requested\n')
        return summarize(run)
    if is_running(run): raise ChainError('Runner is active. Stop it before changing candidate decisions.')
    if state.get('status')=='complete': raise ChainError('Run is complete. Use a new run name for a new render.')
    candidate=state.get('candidate')
    if action=='Approve candidate and continue':
        if not candidate or state.get('status')!='needs_review':
            raise ChainError('No candidate is waiting for review')
        verify_receipt(candidate['folder'],candidate['segment']['new_frames'])
        candidate['manual_approval']=True
        state['candidate']=candidate
    elif action=='Retry candidate':
        if not candidate or state.get('status')!='needs_review':
            raise ChainError('No candidate is waiting for review')
        state.update(attempt=candidate['attempt']+1,candidate=None,pending=None,status='retry_requested')
    elif action!='Resume': raise ChainError('Unknown control action')
    atomic_json(run/'state.json',state)
    launch(run)
    return summarize(run)


def _view(path,output_root):
    path=Path(path).resolve();root=Path(output_root).resolve()
    if not path.is_file() or not path.is_relative_to(root): return None
    rel=path.relative_to(root)
    return '/view?'+urllib.parse.urlencode({'filename':rel.name,'subfolder':rel.parent.as_posix(),'type':'output'})


def summarize(run):
    run=Path(run)
    state=read_json(run/'state.json');config=read_json(run/'config.json')
    output=config['output_root']
    candidate=state.get('candidate')
    folder=Path(candidate['folder']) if candidate else (Path(state['chunks'][-1]['folder']) if state.get('chunks') else None)
    links={}
    if folder:
        for name in ('preview.mp4','join_review.mp4','first_frame.png','source_mask.png'):
            link=_view(folder/name,output)
            if link: links[name]=link
    if state.get('final'):
        link=_view(state['final'],output)
        if link: links['final.mp4']=link
    return {'run':run.name,'status':state.get('status'),'controller_running':is_running(run),
            'accepted_chunks':len(state.get('chunks',[])),'total_chunks':len(state.get('plan',[])),
            'accepted_seconds':sum(x['new_frames'] for x in state.get('chunks',[]))/24,
            'active_section':state.get('active_section'),'active_attempt':state.get('active_attempt'),
            'error':state.get('error'),'candidate_quality':candidate.get('quality') if candidate else None,
            'review_mode':config.get('review_mode'),'output_folder':str(run),'links':links,
            'review_url':'/_swap/chain?run='+urllib.parse.quote(run.name),
            'seamless_motion_guaranteed':False,
            'reference_mode':config.get('reference_mode','single'),
            'reference_roles':config.get('reference_roles', {'Picture 1':'complete character'}),
            'reference_files':{'Picture 1':Path(config['reference']).name,
                **({'Picture 2':Path(config['body_reference']).name} if config.get('reference_mode')=='face_body' else {})}}


def summaries(output_root):
    dirs=[p for p in root_dir(output_root).iterdir() if p.is_dir() and (p/'state.json').exists()]
    result=[]
    for p in sorted(dirs,key=lambda x:x.stat().st_mtime,reverse=True)[:50]:
        try: result.append(summarize(p))
        except (OSError,ValueError,KeyError): continue
    return result

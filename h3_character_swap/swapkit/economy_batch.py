"""Bounded independent H3 jobs, first-chunk review, resumable on-disk assembly.

No generated history is fed forward. The optional legacy continuation workflow
remains separate. Every submitted job contains an exact editable canvas for the
native ComfyUI Open workflow action, not merely an API prompt or MP4 sidecar.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from .config import ROOT, atomic_json, data_root, file_sha, public_child_env
from . import independent_media as media
from .disk_common import ComfyClient, ChainError, Stopped, check_stop, json_digest
from .precision import profile
from .recovery import SIMPLE_PROMPT

ACTIONS=('Start new run','Status','Approve and continue','Retry candidate','Resume','Stop')
REVIEWS=('Approve first chunk','Review every chunk','Automatic')
CLEANUPS=('Stage offload','Full cleanup between chunks')


def folder(name, root=None):
    media.safe_name(name)
    if len(name)>40:raise ValueError('Use a run name of at most 40 characters.')
    return Path(root or data_root()).resolve()/'output/h3_character_swap/economy'/name


def active(run):
    try:
        pid=int((Path(run)/'controller.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return 'swapkit.economy_batch' in cmd and str(Path(run).resolve()) in cmd
    except (OSError,ValueError):return False


def _reference(root, name):
    if not isinstance(name,str):raise ValueError('Select a reference image.')
    p=Path(name)
    if not p.is_absolute():p=root/p
    p=p.resolve()
    if not p.is_relative_to(root/'input') or not p.is_file() or p.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
        raise ValueError('Reference must be a PNG/JPG/WebP inside workspace/input.')
    return p


def create(source_video,character_image,run_name,duration_seconds=10.,start_seconds=0.,
           resolution='768',window_frames='124',overlap_frames=12,target_person='person',
           selection='largest',selection_index=0,mask_expansion=64,seed=904239,
           review_policy=REVIEWS[0],cleanup_policy=CLEANUPS[0],prompt=SIMPLE_PROMPT):
    from .analysis_options import validate_prompt
    from .recovery import enabled
    if not enabled():raise ValueError('Automatic batches require H3_SWAP_SAVE_METADATA=1 for right-click workflow recovery.')
    root=data_root().resolve();source=media.local_media(source_video,root);ref=_reference(root,character_image)
    run=folder(run_name,root)
    if run.exists():raise ValueError('Run already exists. Resume it or choose a new run_name.')
    if not 0<float(duration_seconds)<=600 or not 0<=float(start_seconds)<=86400:
        raise ValueError('Use 0 < duration <= 600 seconds and a nonnegative start.')
    media.plan_chunks(1,int(window_frames),int(overlap_frames))
    if resolution not in ('480','768') or selection not in ('largest','leftmost','rightmost','index'):
        raise ValueError('Unsupported resolution or selection.')
    if not 0<=int(selection_index)<=100 or not 0<=int(mask_expansion)<=256 or not 0<=int(seed)<2**64:
        raise ValueError('Invalid mask/seed setting.')
    if review_policy not in REVIEWS or cleanup_policy not in CLEANUPS:raise ValueError('Unknown batch policy.')
    if not isinstance(target_person,str) or not 1<=len(target_person.strip())<=1000:raise ValueError('Describe the original performer.')
    validate_prompt(prompt)
    status_path=root/'h3-swap/status.json'
    if status_path.exists() and not json.loads(status_path.read_text()).get('ready'):
        raise ValueError('Wait for /_swap/status: ready=true before generation. Files remain available now.')
    if shutil.disk_usage(root).free < 8*1024**3:raise ValueError('At least 8 GiB free is required to begin; longer lossless runs need more.')
    # Freeze both the API and the original editable canvas from this image.
    base_api=json.loads((ROOT/'workflows/12_Akatz_Independent_Chunk_Preview.api.json').read_text())
    base_canvas=json.loads((ROOT/'workflows/12_Akatz_Independent_Chunk_Preview.json').read_text())
    config=dict(version=1,run_name=run_name,source=str(source),reference=str(ref),
        source_sha256=file_sha(source),reference_sha256=file_sha(ref),
        duration_seconds=float(duration_seconds),start_seconds=float(start_seconds),
        resolution=resolution,window_frames=int(window_frames),overlap_frames=int(overlap_frames),
        target_person=target_person,selection=selection,selection_index=int(selection_index),
        mask_expansion=int(mask_expansion),seed=int(seed),review_policy=review_policy,
        cleanup_policy=cleanup_policy,prompt=prompt,model_profile=profile(),
        recipe_sha256=json_digest(base_api),canvas_sha256=json_digest(base_canvas),
        history_mode='independent; no generated appearance inherited')
    run.mkdir(parents=True)
    atomic_json(run/'config.json',config);atomic_json(run/'base.api.json',base_api)
    atomic_json(run/'base.workflow.json',base_canvas)
    atomic_json(run/'state.json',dict(status='created',accepted=[],candidate=None,attempt=0,pending=None,
                                   config_digest=json_digest(config)))
    return run


def launch(run):
    run=Path(run).resolve()
    if active(run):raise ValueError('This batch is already running.')
    run.parent.mkdir(parents=True,exist_ok=True)
    for p in run.parent.iterdir():
        if p.is_dir() and p!=run and active(p):raise ValueError('Another economy batch is active: '+p.name)
    (run/'STOP').unlink(missing_ok=True)
    with (run/'controller.log').open('a',buffering=1) as log:
        p=subprocess.Popen([sys.executable,'-m','swapkit.economy_batch','--run',str(run)],
            cwd=str(ROOT),env=public_child_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    (run/'controller.pid').write_text(str(p.pid))


def action(run_name, command):
    run=folder(run_name)
    if not (run/'state.json').is_file():raise ValueError('Run not found.')
    if command=='Status':return summary(run)
    if command=='Stop':(run/'STOP').write_text('stop requested\n');return summary(run)
    if active(run):raise ValueError('Batch is still active. Stop it before changing a decision.')
    state=json.loads((run/'state.json').read_text())
    if state['status']=='complete':return summary(run)
    if command=='Approve and continue':
        if state['status']!='needs_review' or not state.get('candidate'):raise ValueError('No candidate is awaiting review.')
        state['candidate']['approved']=True
    elif command=='Retry candidate':
        if state['status']!='needs_review' or not state.get('candidate'):raise ValueError('No candidate is awaiting review.')
        state.update(candidate=None,pending=None,attempt=state['attempt']+1)
    elif command!='Resume':raise ValueError('Unknown action.')
    atomic_json(run/'state.json',state);launch(run);return summary(run)


def _set(canvas, api, ident, **values):
    ident=str(ident);node=next(n for n in canvas['nodes'] if str(n['id'])==ident)
    named=node.get('widgets_values_named',{})
    for key,value in values.items():
        api[ident]['inputs'][key]=value
        if key not in named:raise ValueError('Factory widget not found: '+key)
        idx=list(named).index(key);node['widgets_values'][idx]=value;named[key]=value


def render_job(canvas, api, config, clip, reference, index, attempt):
    """Only edit user widgets. Preserve native mask, reference and sampler wiring."""
    canvas,api=copy.deepcopy(canvas),copy.deepcopy(api)
    _set(canvas,api,'source',source_clip=clip)
    _set(canvas,api,'reference',image=reference)
    _set(canvas,api,'prepare',resolution=config['resolution'],duration_seconds=0.)
    _set(canvas,api,'subject',text=config['target_person'])
    _set(canvas,api,'select',selection=config['selection'],index=config['selection_index'])
    _set(canvas,api,'expand_mask',expansion_px=config['mask_expansion'])
    seed=(config['seed']+(index*7919+attempt*1000003 if attempt else 0))%(2**64)
    _set(canvas,api,'plan',window_frames=str(config['window_frames']),history_frames='0',seed=seed)
    _set(canvas,api,'reference_conditioning',prompt=config['prompt'])
    _set(canvas,api,'save',take_name=f"econ_{config['run_name']}_p{index:04d}_a{attempt:02d}")
    for ident,node in api.items():
        if node['class_type']=='H3SwapWindowPreview':_set(canvas,api,ident,hold_seconds=0)
    canvas.setdefault('extra',{})['h3_batch']={'run':config['run_name'],'section':index,'attempt':attempt,
        'mode':'independent','reference_sha256':config['reference_sha256'],'model_profile':config['model_profile']}
    # Per-chunk stable UUID prevents frontend job/workflow associations collapsing
    # every section onto a single unrelated factory tab.
    canvas['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,f"h3-economy/{config['run_name']}/{index}/{attempt}"))
    return canvas,api


class Client(ComfyClient):
    def submit(self,graph,workflow):
        result=self.request('/prompt',{'prompt':graph,'client_id':self.client_id,
            'extra_data':{'extra_pnginfo':{'workflow':workflow}}})
        if result.get('node_errors') or not result.get('prompt_id'):raise ChainError('Comfy rejected this job: '+str(result)[-3000:])
        self.current=result['prompt_id'];return self.current
    def cancel_owned(self):
        if self.current:
            try:self.request('/api/jobs/'+self.current+'/cancel',{})
            except ChainError:pass


def _completed_master(result):
    output=result.get('outputs',{}).get('save',{})
    for item in output.get('text',[]):
        line=item.splitlines()[0]
        if line.startswith('output/') and line.endswith('/master.mkv'):
            path=media.local_media(line)
            rec=json.loads((path.parent/'receipt.json').read_text())
            if rec.get('format')!='h3-independent-render-v1' or file_sha(path)!=rec['master_sha256']:
                raise ValueError('Independent master is not a verified completed render.')
            return {'master':str(path),'sha256':rec['master_sha256'],'preview':str(path.parent/'preview.mp4'),
                    'frames':rec['frames'],'receipt':str(path.parent/'receipt.json')}
    raise ValueError('Completed job did not publish an independent master filename.')


def execute(run, client=None):
    run=Path(run).resolve();root=data_root().resolve()
    config=json.loads((run/'config.json').read_text());state=json.loads((run/'state.json').read_text())
    if folder(config['run_name'])!=run or config['model_profile']!=profile():raise ValueError('Wrong run location or precision profile.')
    if json_digest(config)!=state['config_digest']:raise ValueError('Run settings changed. Start a new run.')
    if state['status']=='complete':return state
    client=client or Client();client.stop_file=run/'STOP'
    locks=root/'output/h3_character_swap/disk_chain_v2';locks.mkdir(parents=True,exist_ok=True)
    lock=(locks/'.controller.lock').open('a+')
    def save(**kw):state.update(kw);atomic_json(run/'state.json',state)
    def stop():check_stop(run/'STOP')
    def log(text):print('[H3 ECONOMY] '+text,flush=True)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop()
        if file_sha(Path(config['source']))!=config['source_sha256'] or file_sha(Path(config['reference']))!=config['reference_sha256']:
            raise ValueError('Original source/reference changed. Start a new run.')
        api=json.loads((run/'base.api.json').read_text());canvas=json.loads((run/'base.workflow.json').read_text())
        if json_digest(api)!=config['recipe_sha256'] or json_digest(canvas)!=config['canvas_sha256']:raise ValueError('Frozen recipe changed.')
        # Wait for the short launcher node to return; never clear someone else's queue.
        client.wait_idle()
        if not state.get('manifest'):
            save(status='preparing_source',error=None)
            manifest,mp=media.split_source(config['source'],'econ_'+config['run_name'],
                config['duration_seconds'],config['start_seconds'],config['window_frames'],config['overlap_frames'],cancel=stop)
            save(manifest=str(mp),total_chunks=len(manifest['clips']))
        manifest=json.loads(Path(state['manifest']).read_text())
        stage=Path(state['manifest']).parent
        ref=stage/('character'+Path(config['reference']).suffix.lower())
        if not ref.exists():shutil.copy2(config['reference'],ref)
        if file_sha(ref)!=config['reference_sha256']:raise ValueError('Staged reference was changed.')
        for accepted in state['accepted']:
            if file_sha(Path(accepted['master']))!=accepted['sha256']:raise ValueError('An accepted master changed.')
        while len(state['accepted'])<len(manifest['clips']):
            stop();index=len(state['accepted'])+1;candidate=state.get('candidate')
            if candidate:
                review=config['review_policy']=='Review every chunk' or (config['review_policy']=='Approve first chunk' and index==1)
                if review and not candidate.get('approved'):
                    save(status='needs_review');log('Review candidate at /_swap/economy. No GPU job is running.');return state
                if file_sha(Path(candidate['master']))!=candidate['sha256']:raise ValueError('Candidate changed after review.')
                state['accepted'].append(candidate);save(candidate=None,attempt=0,pending=None,status='accepted');continue
            row=manifest['clips'][index-1];clip=media.local_media(row['file'])
            if file_sha(clip)!=row['sha256']:raise ValueError('Prepared source clip changed.')
            c,g=render_job(canvas,api,config,row['file'],ref.relative_to(root/'input').as_posix(),index,state['attempt'])
            jobdir=run/f"part_{index:04d}"/f"attempt_{state['attempt']:02d}";jobdir.mkdir(parents=True,exist_ok=True)
            atomic_json(jobdir/'workflow.json',c);atomic_json(jobdir/'job.api.json',g)
            started=time.monotonic()
            if state.get('pending'):
                ident=state['pending'];result=client.request('/history/'+ident).get(ident)
                if not result:
                    raise ValueError('Interrupted job has no completed history. Inspect its output before retrying; no accepted clip was replaced.')
                if result.get('status',{}).get('status_str')!='success':
                    save(pending=None,attempt=state['attempt']+1)
                    raise ValueError('Previous job failed. Settings retained; Resume retries with a recorded new seed.')
            else:
                client.wait_idle();stop()
                if config['cleanup_policy']==CLEANUPS[1]:client.clear_memory()
                ident=client.submit(g,c);save(status='rendering',pending=ident,active_chunk=index,error=None)
                result=client.wait(ident)
            atomic_json(jobdir/'history.json',result)
            record=_completed_master(result)
            record.update(job_id=ident,section=index,attempt=state['attempt'],elapsed_seconds=time.monotonic()-started)
            save(candidate=record,pending=None,status='candidate_ready')
            # Native Relay unloads the active model around encoding/sampling.
            # Do not clear every execution cache a second time in Stage mode.
        stop();save(status='joining')
        if state.get('join_pending'):
            result=client.request('/history/'+state['join_pending']).get(state['join_pending'])
            if not result or result.get('status',{}).get('status_str')!='success':
                save(join_pending=None)
                raise ValueError('Previous join did not finish. Resume creates a new join output; source masters are intact.')
        else:
            ca=json.loads((ROOT/'workflows/13_H3_Join_Rendered_Chunks.json').read_text())
            ga=json.loads((ROOT/'workflows/13_H3_Join_Rendered_Chunks.api.json').read_text())
            names='\n'.join(Path(x['master']).relative_to(root).as_posix() for x in state['accepted'])
            _set(ca,ga,'1',ordered_videos=names,join_name='econ_'+config['run_name']+'_'+uuid.uuid4().hex[:6],
                overlap_frames=config['overlap_frames'],cut_placement='Middle of overlap',
                audio_mode='Original source' if media.video_info(Path(config['source']),False)['audio'] else 'No audio',
                original_source=Path(config['source']).relative_to(root).as_posix(),source_start_seconds=config['start_seconds'])
            atomic_json(run/'join.workflow.json',ca);atomic_json(run/'join.api.json',ga)
            client.wait_idle();ident=client.submit(ga,ca);save(join_pending=ident)
            result=client.wait(ident)
        text=result.get('outputs',{}).get('1',{}).get('text',[])
        report=json.loads(text[0]) if text else {}
        if not report.get('video'):raise ValueError('Join completed without final video report.')
        save(status='complete',final=report['video'],join_pending=None,error=None)
        log('Complete. Right-click the chunk or final output in ComfyUI History and Open workflow.')
        return state
    except BlockingIOError:
        save(status='error',error='Another H3 controller is active. Stop or finish it before resuming this batch.');raise
    except BaseException as exc:
        if isinstance(exc,(Stopped,KeyboardInterrupt)):client.cancel_owned()
        save(status='stopped' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'error',error=str(exc))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def summary(run):
    run=Path(run);state=json.loads((run/'state.json').read_text())
    from .disk_control import _view
    candidate=state.get('candidate') or (state['accepted'][-1] if state.get('accepted') else {})
    return {**{k:state.get(k) for k in ('status','error','active_chunk','total_chunks','final')},
            'run_name':run.name,'accepted':len(state.get('accepted',[])),'running':active(run),
            'preview':_view(candidate.get('preview',''),data_root()/'output') if candidate else None,
            'final_url':_view(state['final'],data_root()/'output') if state.get('final') else None,
            'review_url':'/_swap/economy?run='+run.name}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    options=parser.parse_args()
    try:execute(Path(options.run))
    except Exception as exc:print('H3 ECONOMY ERROR: '+str(exc),file=sys.stderr);raise SystemExit(1)

"""Sequential, disk-backed H3 Akatz controller. Runs outside the Comfy worker.

One short prompt at a time. Save/checkpoint -> cleanup -> validate/review -> next.
The original 18-frame Relay history contract is retained. No claim of pixel
locking or guaranteed seamless motion is made by this implementation.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from .disk_common import (ChainError, Stopped, FPS, HISTORY, VALID_WINDOWS, atomic_json,
    read_json, sha256, json_digest, inside, normalize_source, plan_segments, extract_segment,
    validate_api_graph, ComfyClient, schema_check, run_media, media_info, verify_video, log_event,
    check_stop)
from .disk_storage import verify_receipt, load_arrays
from .disk_quality import assess_join

VERSION='2026-10-26.2'
RECIPE=Path(__file__).parent/'data/akatz_disk_base.api.json'
MODES=('Auto check + retry', 'Review every chunk')
VIDEO_EXT={'.mp4','.mov','.mkv','.webm','.m4v'}
IMAGE_EXT={'.png','.jpg','.jpeg','.webp'}


def root_dir(output):
    root=Path(output).resolve()/'h3_character_swap/disk_chain_v2'
    root.mkdir(parents=True,exist_ok=True)
    return root


def resolve_run(output, name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ChainError('Run name: 1-64 letters, numbers, underscores or hyphens; no path.')
    return inside(root_dir(output)/name,root_dir(output))


def runtime_signature(input_root, base):
    """Detect changed build/model files on resume without hashing 60 GB each job.

    Image startup already verifies weight SHA256. Here size/mtime receipts detect
    ordinary replacements; this is not an adversarial filesystem integrity proof.
    """
    here=Path(__file__).resolve().parents[1]
    build={name:(sha256(here/name) if (here/name).exists() else None)
           for name in ('pins.json','assets.lock.json','requirements.txt')}
    fields={'UNETLoader':('diffusion_models','unet_name'),
            'CLIPLoader':('text_encoders','clip_name'),'VAELoader':('vae','vae_name'),
            'LoraLoaderModelOnly':('loras','lora_name'),'CheckpointLoaderSimple':('checkpoints','ckpt_name')}
    model_root=Path(input_root).parent/'models';models={}
    for node in base.values():
        if node['class_type'] not in fields: continue
        category,field=fields[node['class_type']];name=node['inputs'][field]
        path=model_root/category/name
        if path.exists():
            stat=path.stat();models[category+'/'+name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    return {'build':build,'models':models,'integration_version':VERSION}


def create_run(*, source, reference, input_root, output_root, run_name, duration=19.,
               start=0., resolution='768', window=124, target='person', selection='largest',
               selection_index=0, expansion=64, seed=904239, review=MODES[0], retries=2,
               prompt=None):
    input_root,output_root=Path(input_root).resolve(),Path(output_root).resolve()
    if not input_root.is_dir() or not output_root.is_dir():
        raise ChainError('Comfy input/output directories must exist')
    source=inside(Path(source),input_root)
    reference=Path(reference).resolve()
    if not any(reference.is_relative_to(r) and reference!=r for r in (input_root,output_root)):
        raise ChainError('Character reference must be in Comfy input or output storage')
    if source.suffix.lower() not in VIDEO_EXT or reference.suffix.lower() not in IMAGE_EXT:
        raise ChainError('Choose a supported local video and PNG/JPG/WebP reference')
    if not source.is_file() or not reference.is_file():
        raise ChainError('Source/reference not found')
    if not math.isfinite(float(duration)) or not 1<=float(duration)<=3600:
        raise ChainError('Duration must be between 1 and 3600 seconds')
    if not math.isfinite(float(start)) or not 0<=float(start)<=86400:
        raise ChainError('Start must be a finite nonnegative timestamp')
    if str(resolution) not in ('480','768') or int(window) not in VALID_WINDOWS:
        raise ChainError('Unsupported resolution/window')
    if not 0<=int(expansion)<=256 or int(selection_index)<0 or not 0<=int(seed)<2**64:
        raise ChainError('Invalid mask/seed settings')
    if selection not in ('largest','leftmost','rightmost','index') or review not in MODES or not 0<=int(retries)<=3:
        raise ChainError('Invalid selection/review/retry setting')
    if not str(target).strip() or len(str(target))>1000:
        raise ChainError('Describe the source performer in 1-1000 characters')
    run=resolve_run(output_root,run_name)
    if run.exists():
        raise ChainError('Run name already exists. Use Resume/Status, or a new name for different settings.')
    if shutil.disk_usage(output_root).free<4*1024**3:
        raise ChainError('At least 4 GiB free is needed to start; long lossless runs need more.')
    base=read_json(RECIPE)
    validate_api_graph(base)
    prompt=str(prompt or base['reference_conditioning']['inputs']['prompt'])
    if len(prompt)>20000:
        raise ChainError('Prompt is too long')
    run.mkdir()
    try:
        config={'version':VERSION,'run_name':run_name,'source':str(source),'reference':str(reference),
                'input_root':str(input_root),'output_root':str(output_root),
                'source_sha256':sha256(source),'reference_sha256':sha256(reference),
                'workflow_digest':json_digest(base),'duration_seconds':float(duration),'start_seconds':float(start),
                'resolution':str(resolution),'window_frames':int(window),'target_person':str(target),
                'selection':selection,'selection_index':int(selection_index),'mask_expansion':int(expansion),
                'seed':int(seed),'review_mode':review,'max_retries':int(retries),'prompt':prompt,
                'url':'http://127.0.0.1:8189','history_contract':'17 prior + 1 boundary, native Relay sliding history',
                'continuity_guarantee':False,'runtime_signature':runtime_signature(input_root,base)}
        atomic_json(run/'base.api.json',base)
        atomic_json(run/'config.json',config)
        atomic_json(run/'state.json',{'version':VERSION,'status':'created','chunks':[],
                                     'pending':None,'candidate':None,'attempt':0})
    except BaseException:
        shutil.rmtree(run)
        raise
    return run


def retry_seed(seed,index,attempt):
    return int(seed) if attempt==0 else (int(seed)+index*7919+attempt*1000003)%(2**64)


def make_job(base, config, segment, source_file, reference_file, prefix, previous=None, attempt=0):
    graph=copy.deepcopy(base)
    graph['source']['inputs']['file']=source_file
    graph['reference']['inputs']['image']=reference_file
    graph['prepare']['inputs'].update(duration_seconds=0.,resolution=config['resolution'])
    graph['subject']['inputs']['text']=config['target_person']
    graph['select']['inputs'].update(selection=config['selection'],index=config['selection_index'])
    graph['expand_mask']['inputs']['expansion_px']=config['mask_expansion']
    graph['reference_conditioning']['inputs']['prompt']=config['prompt']
    graph['plan']['inputs'].update(schedule='segment',window_frames=str(config['window_frames']),
        history_frames=str(HISTORY),seed=retry_seed(config['seed'],segment['index'],attempt),
        seed_mode='fixed',seed_overrides='{}')
    for name in ('start','end','video','save'):
        graph.pop(name,None)
    graph['window']['inputs'].update(iteration_index=0)
    graph['window']['inputs'].pop('previous',None)
    graph['append']['inputs'].pop('previous',None)
    graph['finish']['inputs']['state']=['append',0]
    if segment['history_frames']:
        if previous is None:
            raise ChainError('Continuation is missing its accepted checkpoint')
        graph['checkpoint']={'class_type':'H3SwapDiskHistory','inputs':{
            'checkpoint':previous['checkpoint'],'sha256':previous['checkpoint_sha256'],
            'selection':config['selection'],'selection_index':config['selection_index']}}
        for name in ('plan','window','append'):
            graph[name]['inputs']['initial_history']=['checkpoint',0]
        # A raw mask at the SAME source timestamp seeds tracking, preventing a
        # fresh independent person choice at the start of every continuation.
        graph['track']['inputs']['initial_mask']=['checkpoint',1]
        graph['recover']['inputs']['selection']=['checkpoint',2]
        for name in ('first_frame','detect','select'):
            graph.pop(name,None)
    graph['disk_save']={'class_type':'H3SwapDiskSave','inputs':{
        'images':['finish',0],'source':['prepare',0],'raw_masks':['recover',0],
        'folder':prefix,'new_frames':segment['new_frames'],
        'history_frames':segment['history_frames'],'selection':config['selection'],
        'selection_index':config['selection_index']},
        '_meta':{'title':'Save lossless master, float history, source mask and join evidence'}}
    validate_api_graph(graph)
    return graph


def preview_part(folder, logdir, stop):
    folder=Path(folder)
    target=folder/'preview.mp4'
    if target.exists(): return target
    tmp=folder/'preview.partial.mp4'
    run_media(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(folder/'master.mkv'),
        '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',
        '-movflags','+faststart',str(tmp)],logdir/(folder.name+'_preview.log'),stop)
    os.replace(tmp,target)
    return target


def join_preview(previous_folder,current_folder,target,logdir,stop):
    """A normal-speed, unblended view around the true join, repeated three times."""
    p=Path(previous_folder)/'master.mkv';n=Path(current_folder)/'master.mkv'
    plen=verify_video(p)['frames']
    filt=(f'[0:v]trim=start_frame={max(0,plen-24)},setpts=PTS-STARTPTS[a];'
          '[1:v]trim=end_frame=24,setpts=PTS-STARTPTS[b];'
          '[a][b]concat=n=2:v=1:a=0[v]')
    temp=Path(target).with_name('join_once.mp4')
    run_media(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(p),'-i',str(n),
        '-filter_complex',filt,'-map','[v]','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-threads','2',str(temp)],logdir/'join_once.log',stop)
    run_media(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-stream_loop','2',
        '-i',str(temp),'-c','copy','-movflags','+faststart',str(target)],logdir/'join_repeat.log',stop)
    temp.unlink(missing_ok=True)


def assemble(rows, normalized, target, frames, logs, stop):
    """Decode one disk master at a time, encode once. Never stack whole movie."""
    manifest=Path(target).parent/'concat_masters.txt'
    out=[]
    for row in rows:
        path=Path(row['folder'])/'master.mkv'
        if "'" in str(path) or '\n' in str(path): raise ChainError('Unsafe concat path')
        out+= ["file '"+str(path)+"'",f"duration {row['new_frames']/24:.12f}"]
    manifest.write_text('\n'.join(out)+'\n')
    temp=Path(target).with_suffix('.partial.mp4')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-f','concat','-safe','0',
         '-i',str(manifest),'-i',str(normalized),'-map','0:v:0','-c:v','libx264','-preset','medium',
         '-crf','16','-pix_fmt','yuv420p','-threads','2']
    if media_info(normalized)['audio']:
        cmd+=['-map','1:a:0','-af',f'apad,atrim=end_sample={frames*2000}',
              '-c:a','aac','-b:a','192k','-ar','48000','-ac','2']
    else: cmd+=['-an']
    cmd+=['-t',f'{frames/24:.12f}','-map_metadata','-1','-movflags','+faststart',str(temp)]
    run_media(cmd,logs/'assemble.log',stop)
    info=verify_video(temp,frames)
    os.replace(temp,target)
    return info


def _accept(state,candidate):
    segment=candidate['segment'];folder=Path(candidate['folder'])
    receipt=verify_receipt(folder,segment['new_frames'])
    row={**segment,'folder':str(folder),'receipt_sha256':sha256(folder/'receipt.json'),
         'checkpoint':str(folder/'history.npz'),'checkpoint_sha256':receipt['files']['history.npz'],
         'attempt':candidate['attempt'],'seam':candidate['quality'],
         'manual_approval':bool(candidate.get('manual_approval'))}
    state['chunks'].append(row)
    state.update(candidate=None,pending=None,attempt=0,status='accepted')


def _candidate_policy(config,state):
    candidate=state['candidate']
    if candidate.get('manual_approval'):
        return 'accept'
    if config['review_mode']=='Review every chunk': return 'review'
    q=candidate['quality']
    if q['verdict']=='pass': return 'accept'
    if 'possible_source_cut' in q['flags']: return 'review'
    return 'retry' if candidate['attempt']<config['max_retries'] else 'review'


def execute(run_dir, client=None):
    run=Path(run_dir).resolve();config=read_json(run/'config.json')
    if config.get('version')!=VERSION:
        raise ChainError('Use a new v2 run. Earlier runs use different history precision/contracts.')
    output=Path(config['output_root']);inputs=Path(config['input_root'])
    if resolve_run(output,config['run_name'])!=run: raise ChainError('Run location mismatch')
    state_path=run/'state.json';state=read_json(state_path)
    if state.get('status')=='complete': return state
    lock=(root_dir(output)/'.controller.lock').open('a+')
    client=client or ComfyClient()
    client.stop_file=run/'STOP'
    logs=run/'logs';logs.mkdir(exist_ok=True)
    stage=inside(inputs/('h3_chain_v2_'+config['run_name']),inputs);stage.mkdir(exist_ok=True)
    stop=run/'STOP'
    def persist(**values):
        state.update(values);atomic_json(state_path,state)
    def event(message): log_event(run,message)
    try:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ChainError('Another disk-chain runner is active. Only one is allowed.')
        check_stop(stop)
        fingerprint=json_digest(config)
        if state.get('config_digest',fingerprint)!=fingerprint: raise ChainError('Run settings changed; start a new run')
        persist(config_digest=fingerprint,status='checking')
        if sha256(Path(config['source']))!=config['source_sha256'] or sha256(Path(config['reference']))!=config['reference_sha256']:
            raise ChainError('Source or reference changed; do not resume this run')
        base=read_json(run/'base.api.json')
        if json_digest(base)!=config['workflow_digest']: raise ChainError('Workflow changed; do not mix settings')
        if runtime_signature(inputs,base)!=config['runtime_signature']:
            raise ChainError('Build or model files changed. Start a new run instead of mixed-model resume.')
        schema_check(client,base)
        schema=client.request('/object_info')
        missing={'H3SwapDiskSave','H3SwapDiskHistory'}-set(schema)
        if missing: raise ChainError('New disk nodes are missing. Rebuild/update the character-swap image.')
        # Reclaim models/caches, but never while a previously submitted prompt
        # might still be using them. This also lets the launcher prompt finish.
        client.wait_idle()
        client.clear_memory()
        source=stage/'source_24fps.mkv'
        if not state.get('prepared'):
            event('Preparing a disk-backed 24 fps source. No long IMAGE batch is created.')
            prepared=normalize_source(config,source,logs,stop)
            persist(prepared=prepared,source_master_sha256=sha256(source))
        else:
            if not source.is_file() or sha256(source)!=state['source_master_sha256']:
                raise ChainError('Normalized source changed/missing; start a new run')
        plan=plan_segments(state['prepared']['frames'],config['window_frames'])
        persist(plan=plan)
        reference=stage/('character'+Path(config['reference']).suffix.lower())
        if not reference.exists(): shutil.copy2(config['reference'],reference)
        if sha256(reference)!=config['reference_sha256']: raise ChainError('Staged reference changed')
        for i,row in enumerate(state['chunks']):
            if row['index']!=i or row['new_frames']!=plan[i]['new_frames']:
                raise ChainError('Accepted timeline is inconsistent')
            if sha256(Path(row['folder'])/'receipt.json')!=row['receipt_sha256']:
                raise ChainError('An accepted receipt changed')
            verify_receipt(row['folder'],row['new_frames'])
        event(f"{state['prepared']['frames']/24:.3f}s; {len(plan)} short jobs; {len(state['chunks'])} accepted.")
        while len(state['chunks'])<len(plan):
            check_stop(stop)
            index=len(state['chunks']);segment=plan[index]
            candidate=state.get('candidate')
            if candidate:
                if candidate['segment']!=segment: raise ChainError('Candidate timeline changed')
                verify_receipt(candidate['folder'],segment['new_frames'])
                policy=_candidate_policy(config,state)
                if policy=='accept':
                    _accept(state,candidate);persist()
                    event(f'Accepted section {index+1}/{len(plan)}. Saved prefix is unchanged.')
                    continue
                if policy=='review':
                    persist(status='needs_review')
                    event('PAUSED FOR REVIEW: '+candidate['folder']+' (preview.mp4 / join_review.mp4 / quality.json).')
                    return state
                event('Join flagged: '+', '.join(candidate['quality']['flags'])+'. Retrying this section only.')
                persist(candidate=None,attempt=candidate['attempt']+1,status='retrying')
                continue
            attempt=int(state.get('attempt',0))
            part=run/f'part_{index+1:04d}'/f'attempt_{attempt:02d}'
            part.mkdir(parents=True,exist_ok=True)
            prefix=part.relative_to(output).as_posix()
            bounded_source=stage/f'source_{index+1:04d}.mkv'
            previous=state['chunks'][-1] if state['chunks'] else None
            # A completed on-disk receipt survives controller or Comfy restarts.
            # It is never replaced by a new sample during resume.
            if not (part/'receipt.json').exists():
                pending=state.get('pending')
                if pending and pending['index']==index and pending['attempt']==attempt:
                    old=client.request('/history/'+pending['prompt_id']).get(pending['prompt_id'])
                    if old and old.get('status',{}).get('status_str')=='error':
                        persist(pending=None,status='error')
                        event('Retrying the failed, unaccepted attempt after explicit Resume.')
                    # wait_idle above means it is not still executing. No valid
                    # receipt means this incomplete attempt can be retried safely.
                    persist(pending=None)
                extract_segment(source,bounded_source,segment,logs,stop)
                graph=make_job(base,config,segment,bounded_source.relative_to(inputs).as_posix(),
                    reference.relative_to(inputs).as_posix(),prefix,previous,attempt)
                atomic_json(part/'job.api.json',graph)
                client.wait_idle();check_stop(stop)
                ident=client.queue(graph)
                persist(pending={'prompt_id':ident,'index':index,'attempt':attempt},status='rendering',
                        active_section=index+1,active_attempt=attempt)
                event(f'Rendering {index+1}/{len(plan)}, attempt {attempt+1}; {segment["input_frames"]} source frames including history.')
                result=client.wait(ident)
                atomic_json(part/'comfy_history.json',result)
            receipt=verify_receipt(part,segment['new_frames'])
            if (receipt['width'],receipt['height'])!=(state['prepared']['width'],state['prepared']['height']):
                raise ChainError('Output canvas changed')
            # Free all previous job caches BEFORE thumbnails/preview work or
            # another generation. Capture the actual reported memory stats.
            persist(pending=None,status='freeing_memory')
            before=client.request('/system_stats')
            client.clear_memory()
            after=client.request('/system_stats')
            atomic_json(part/'memory_cleanup.json',{'before':before,'after':after,
                        'method':'/free + two worker fences','not_a_peak_memory_measurement':True})
            check_stop(stop)
            qc=load_arrays(part/'qc.npz',max_bytes=16*1024**2)
            quality=assess_join(load_arrays(Path(previous['folder'])/'qc.npz',max_bytes=16*1024**2),qc) if previous else {
                'policy':'first-section-review','verdict':'pass','flags':[],
                'limitations':'No preceding join. Review identity, full-body edit mask and source adherence.'}
            atomic_json(part/'quality.json',quality)
            preview_part(part,logs,stop)
            if previous: join_preview(previous['folder'],part,part/'join_review.mp4',logs,stop)
            persist(candidate={'segment':segment,'folder':str(part),'attempt':attempt,'quality':quality},
                    status='checking_join')
            bounded_source.unlink(missing_ok=True)
        check_stop(stop)
        persist(status='joining')
        event('Joining accepted disk masters and the continuous original soundtrack.')
        final=run/'final.mp4'
        info=assemble(state['chunks'],source,final,state['prepared']['frames'],logs,stop)
        persist(status='complete',final=str(final),final_sha256=sha256(final),final_info=info)
        event('COMPLETE: '+str(final))
        return state
    except KeyboardInterrupt:
        client.cancel_owned();persist(status='stopped',error='User interrupted. Accepted sections retained.')
        raise
    except Exception as exc:
        if isinstance(exc,Stopped): client.cancel_owned()
        persist(status='stopped' if isinstance(exc,Stopped) else 'error',error=str(exc))
        event('STOPPED: '+str(exc))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Existing v2 run folder created by the control workflow')
    args=parser.parse_args()
    try: execute(Path(args.run))
    except (ChainError,OSError,ValueError) as exc:
        print('H3 DISK CHAIN ERROR: '+str(exc),file=sys.stderr);return 1
    except KeyboardInterrupt: return 130
    return 0

if __name__=='__main__': raise SystemExit(main())

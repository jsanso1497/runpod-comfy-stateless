"""Two-node ComfyUI graph. Video stays file-backed, including its final output."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

APP=Path(os.environ.get('FAITHFUL_APP_ROOT','/opt/video-fidelity'))
if str(APP) not in sys.path:
    sys.path.insert(0,str(APP))
from fv_common import atomic_json, child_env, terminate

class FaithfulVideoRestore:
    CATEGORY='Faithful Video'
    FUNCTION='restore'
    RETURN_TYPES=('VIDEO','STRING')
    RETURN_NAMES=('video','file_path')
    OUTPUT_NODE=True
    DESCRIPTION='Standard SeedVR2 FP16 + source blend. No prompts, faces, sharpening, cropping, or interpolation.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'video':('VIDEO',),
            'output_size':(['Original size','2x size'],),
            'restoration_strength':('FLOAT',{'default':0.5,'min':0.0,'max':1.0,'step':0.05,
                'tooltip':'0 bypasses AI. 0.5 blends equal source/restoration. 1 is full AI. Not a denoise setting or a guarantee against invention.'}),
            'start_seconds':('FLOAT',{'default':0.0,'min':0.0,'max':86400.0,'step':0.01}),
            'duration_seconds':('FLOAT',{'default':3.0,'min':0.0,'max':86400.0,'step':0.01,
                'tooltip':'Starts with a 3-second acceptance test. Set 0 for the remainder; no hidden clip cap.'}),
            'temporal_batch':('INT',{'default':9,'min':5,'max':33,'step':4,
                'tooltip':'5, 9, 13, 17... Larger needs more memory. No silent quality fallback.'}),
            'keep_16bit_frames':('BOOLEAN',{'default':False,
                'tooltip':'Retain RGB16 PNGs and prepared source for inspection/finishing. These are not 16-bit sensor recovery.'}),
        }}

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        return float('nan')

    async def restore(self,video,output_size,restoration_strength,start_seconds,
                      duration_seconds,temporal_batch,keep_16bit_frames):
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        from comfy.utils import ProgressBar
        from filelock import FileLock,Timeout
        get_source=getattr(video,'get_stream_source',None)
        source=get_source() if get_source else None
        if not isinstance(source,(str,os.PathLike)):
            raise ValueError('Connect Load Video directly. In-memory frame batches are not accepted.')
        if getattr(video,'_VideoFromFile__crop',None) is not None:
            raise ValueError('Upstream crops are not supported. Supply the intended full source file.')
        trim=getattr(video,'get_active_trim_window',lambda:(0,0))()
        if trim != (0,0):
            raise ValueError('Set trim here, not on an upstream video object.')
        source=Path(source).resolve()
        roots=[Path(folder_paths.get_input_directory()).resolve(),Path(folder_paths.get_output_directory()).resolve()]
        if not source.is_file() or not any(source.is_relative_to(r) for r in roots):
            raise ValueError('Source must be an uploaded video or a file in ComfyUI input/output.')
        data=Path(os.environ.get('FAITHFUL_DATA_ROOT','/workspace/faithful-video')).resolve()
        jobs=data/'jobs'
        jobs.mkdir(parents=True,exist_ok=True)
        lock=FileLock(str(jobs/'.render.lock'))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError('Another restoration is running. Use the ComfyUI queue.') from None
        name='faithful_'+uuid.uuid4().hex[:16]
        job=jobs/name
        job.mkdir()
        outroot=Path(folder_paths.get_output_directory()).resolve()
        output=outroot/name/'enhanced.mp4'
        request=dict(source=str(source),output=str(output),scale=1 if output_size=='Original size' else 2,
                     strength=float(restoration_strength),start=float(start_seconds),duration=float(duration_seconds),
                     batch=int(temporal_batch),keep_frames=bool(keep_16bit_frames),
                     model_dir=str(data/'models'),seedvr_root=os.environ.get('FAITHFUL_SEEDVR_ROOT','/opt/seedvr2'))
        atomic_json(job/'request.json',request)
        proc=None
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
            env=child_env()
            env['FAITHFUL_PARENT_PID']=str(os.getpid())
            pbar=ProgressBar(100)
            with (job/'supervisor.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-u',str(APP/'job.py'),'--job',str(job)],
                                      stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                      env=env,cwd=APP,start_new_session=True)
                while proc.poll() is None:
                    mm.throw_exception_if_processing_interrupted()
                    try:
                        state=json.loads((job/'status.json').read_text())
                        worker=json.loads((job/'worker_status.json').read_text()) if (job/'worker_status.json').exists() else {}
                        progress=95 if state.get('phase')=='encode' else min(90,5+85*worker.get('frames_written',0)/max(1,state.get('frames',1)))
                        pbar.update_absolute(int(progress),100)
                    except (OSError,ValueError,KeyError):
                        pass
                    await asyncio.sleep(0.4)
            state=json.loads((job/'status.json').read_text()) if (job/'status.json').exists() else {}
            if proc.returncode or state.get('state')!='complete' or not output.is_file():
                raise RuntimeError(state.get('error','Restoration process did not finish.')+f'\nLogs: {job}')
            pbar.update_absolute(100,100)
            rel=output.relative_to(outroot)
            return {'ui':{'images':[{'filename':output.name,'subfolder':str(rel.parent),'type':'output'}],
                          'animated':(True,)},'result':(InputImpl.VideoFromFile(str(output)),str(output))}
        finally:
            if proc is not None:
                terminate(proc,group=True)
            lock.release()

NODE_CLASS_MAPPINGS={'FaithfulVideoRestore':FaithfulVideoRestore}
NODE_DISPLAY_NAME_MAPPINGS={'FaithfulVideoRestore':'Faithful Video | Enhance + Save'}

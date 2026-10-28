"""Lightweight ComfyUI control and checkpoint nodes for the H3 disk chain.

The control node RETURNS after starting a separate controller. It never waits
for jobs queued to its own worker. Model generation remains in normal prompts.
"""
from __future__ import annotations
import json
from pathlib import Path
from .disk_common import ChainError, sha256, inside, HISTORY
from .disk_chain import create_run, resolve_run, MODES, VALID_WINDOWS
from .disk_control import ACTIONS, launch, act, summarize
from .disk_storage import load_arrays, save_chunk
from .config import data_root


def _inputs():
    import folder_paths
    root=Path(folder_paths.get_input_directory())
    return root,Path(folder_paths.get_output_directory())


def choices(ext):
    root,_=_inputs()
    return sorted(p.name for p in root.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in ext) or ['upload_a_file_first']


class H3SwapDiskChain:
    @classmethod
    def INPUT_TYPES(cls):
        from .disk_chain import VIDEO_EXT, IMAGE_EXT
        return {'required':{
            'action':(list(ACTIONS),),
            'run_name':('STRING',{'default':'shot_01'}),
            'source_video':(choices(VIDEO_EXT),),
            'character_image':(choices(IMAGE_EXT),),
            'duration_seconds':('FLOAT',{'default':19.,'min':1.,'max':3600.,'step':1.}),
            'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':86400.,'step':.001}),
            'resolution':(['768','480'],),
            'window_frames':([str(x) for x in reversed(VALID_WINDOWS)],),
            'target_person':('STRING',{'default':'person'}),
            'selection':(['largest','leftmost','rightmost','index'],),
            'selection_index':('INT',{'default':0,'min':0,'max':100}),
            'mask_expansion':('INT',{'default':64,'min':0,'max':256}),
            'seed':('INT',{'default':904239,'min':0,'max':0xffffffffffffffff}),
            'review_mode':(list(MODES),),
            'max_retries':('INT',{'default':2,'min':0,'max':3}),
            'prompt':('STRING',{'default':DIRECT_PROMPT,'multiline':True})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('status_and_review_link',)
    FUNCTION='control'
    OUTPUT_NODE=True
    CATEGORY='H3 Swap/Low memory'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    @classmethod
    def VALIDATE_INPUTS(cls,action,source_video,character_image,**kwargs):
        # Source dropdown values are irrelevant for status/resume decisions.
        if action!='Start new run': return True
        root,_=_inputs()
        return all((root/n).is_file() for n in (source_video,character_image)) or 'Upload source video and character image first'
    def control(self,action,run_name,source_video,character_image,duration_seconds,start_seconds,
                resolution,window_frames,target_person,selection,selection_index,mask_expansion,
                seed,review_mode,max_retries,prompt):
        inputs,outputs=_inputs()
        if action=='Start new run':
            prep=data_root()/'h3-swap/status.json'
            if prep.exists() and not json.loads(prep.read_text()).get('ready'):
                raise ChainError('Wait for /_swap/status to show ready before starting a chain')
            run=create_run(source=inputs/source_video,reference=inputs/character_image,
                input_root=inputs,output_root=outputs,run_name=run_name,duration=duration_seconds,
                start=start_seconds,resolution=resolution,window=int(window_frames),target=target_person,
                selection=selection,selection_index=selection_index,expansion=mask_expansion,
                seed=seed,review=review_mode,retries=max_retries,prompt=prompt)
            launch(run);result=summarize(run)
        else: result=act(outputs,run_name,action)
        text=json.dumps(result,indent=2)
        return {'ui':{'text':[text]},'result':(text,)}


class H3SwapDiskChainFaceBody(H3SwapDiskChain):
    """Separate file-backed references, used in every short generation job."""
    @classmethod
    def INPUT_TYPES(cls):
        old = super().INPUT_TYPES()['required']
        fields = {}
        for name, spec in old.items():
            if name == 'character_image':
                fields['face_image'] = spec
                fields['body_image'] = spec
            elif name == 'prompt':
                fields['additional_instructions'] = ('STRING', {'default': '', 'multiline': True,
                    'tooltip': 'Optional FACE+BODY chain-safe draft or additional details. Built-in role instructions are retained.'})
            else:
                fields[name] = spec
        # Review is deliberate for this experimental two-photo adaptation.
        fields['review_mode'] = (list(MODES), {'default': 'Review every chunk'})
        return {'required': fields}

    @classmethod
    def VALIDATE_INPUTS(cls, action, source_video, face_image, body_image, **kwargs):
        if action != 'Start new run':
            return True
        inputs, _ = _inputs()
        try:
            for name in (source_video, face_image, body_image):
                if not inside(inputs / name, inputs).is_file():
                    return 'Upload the source video, face and body references first'
        except (ValueError, ChainError):
            return 'All inputs must be files inside ComfyUI input storage'
        return True

    def control(self, action, run_name, source_video, face_image, body_image, duration_seconds,
                start_seconds, resolution, window_frames, target_person, selection, selection_index,
                mask_expansion, seed, review_mode, max_retries, additional_instructions=''):
        inputs, outputs = _inputs()
        if action == 'Start new run':
            prep = data_root() / 'h3-swap/status.json'
            if prep.exists() and not json.loads(prep.read_text()).get('ready'):
                raise ChainError('Wait for /_swap/status to show ready before starting a chain')
            run = create_run(source=inputs/source_video, reference=inputs/face_image,
                body_reference=inputs/body_image, input_root=inputs, output_root=outputs,
                run_name=run_name, duration=duration_seconds, start=start_seconds,
                resolution=resolution, window=int(window_frames), target=target_person,
                selection=selection, selection_index=selection_index, expansion=mask_expansion,
                seed=seed, review=review_mode, retries=max_retries, prompt=additional_instructions)
            launch(run)
            result = summarize(run)
        else:
            result = act(outputs, run_name, action)
        text = json.dumps(result, indent=2)
        return {'ui': {'text': [text]}, 'result': (text,)}


class H3SwapDiskHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'checkpoint':('STRING',{}),'sha256':('STRING',{}),
                'selection':(['largest','leftmost','rightmost','index'],),
                'selection_index':('INT',{'default':0,'min':0})}}
    RETURN_TYPES=('IMAGE','MASK','H3_SUBJECT_SELECTION')
    RETURN_NAMES=('generated_history','initial_raw_mask','selection')
    FUNCTION='load'
    CATEGORY='H3 Swap/Low memory/internal'
    @classmethod
    def IS_CHANGED(cls,checkpoint,**kwargs):
        _, outputs = _inputs()
        path = inside(Path(checkpoint), outputs / "h3_character_swap/disk_chain_v2")
        return sha256(path)
    def load(self,checkpoint,sha256,selection,selection_index):
        import torch
        import numpy as np
        from .disk_common import sha256 as digest
        _,outputs=_inputs()
        path=inside(Path(checkpoint),outputs/'h3_character_swap/disk_chain_v2')
        if path.is_symlink() or digest(path)!=sha256: raise ChainError('Generated history checksum mismatch')
        arrays=load_arrays(path)
        if set(arrays)!={'frames','masks'}: raise ChainError('Invalid history file')
        f,m=arrays['frames'],arrays['masks']
        if f.ndim!=4 or len(f)!=HISTORY or f.shape[-1]!=3 or m.shape!=f.shape[:3]:
            raise ChainError('Expected exactly 18 matching generated frames and raw masks')
        if f.dtype!=np.float32 or m.dtype!=np.float32 or max(f.shape[1:3])>4096 or f.shape[1]*f.shape[2]>1_100_000:
            raise ChainError('Unexpected history precision/size')
        return (torch.from_numpy(f),torch.from_numpy(m[:1].copy()),
                {'selection':selection,'index':selection_index})


class H3SwapDiskSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'source':('IMAGE',),'raw_masks':('MASK',),
            'folder':('STRING',{}),'new_frames':('INT',{'default':124,'min':1,'max':124}),
            'history_frames':('INT',{'default':0,'min':0,'max':18}),
            'selection':(['largest','leftmost','rightmost','index'],),
            'selection_index':('INT',{'default':0,'min':0})}}
    RETURN_TYPES=('STRING',)
    FUNCTION='save'
    OUTPUT_NODE=True
    CATEGORY='H3 Swap/Low memory/internal'
    def save(self,images,source,raw_masks,folder,new_frames,history_frames,selection,selection_index):
        import comfy.model_management
        _,outputs=_inputs()
        path=inside(outputs/folder,outputs/'h3_character_swap/disk_chain_v2')
        if history_frames not in (0,18): raise ChainError('Unsupported history contract')
        result=save_chunk(images,source,raw_masks,path,new_frames,history_frames,selection,selection_index,
            comfy.model_management.throw_exception_if_processing_interrupted)
        text=json.dumps({'receipt':str(path/'receipt.json'),**result},indent=2)
        return {'ui':{'text':[text]},'result':(text,)}


class H3SwapClipPrep:
    @classmethod
    def INPUT_TYPES(cls):
        from .clip_prep import MODES,TAIL_POLICIES,AUDIO_MODES
        from .disk_chain import VIDEO_EXT
        return {'required':{'source_video':(choices(VIDEO_EXT),),
            'mode':(list(MODES),),'start_seconds':('FLOAT',{'default':0.,'min':0.,'step':.001}),
            'first_chunk':('INT',{'default':1,'min':1}),
            'chunk_count':('INT',{'default':0,'min':0,'max':1000}),
            'final_chunk':(list(TAIL_POLICIES),),'audio':(list(AUDIO_MODES),)}}
    RETURN_TYPES=('STRING',)
    FUNCTION='split'
    OUTPUT_NODE=True
    CATEGORY='H3 Swap/Low memory'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    def split(self,source_video,mode,start_seconds,first_chunk,chunk_count,final_chunk,audio):
        from .clip_prep import export_clips
        import comfy.model_management
        inputs,outputs=_inputs()
        result=export_clips(inside(inputs/source_video,inputs),outputs,inputs,
            mode=mode,start_seconds=start_seconds,first_chunk=first_chunk,chunk_count=chunk_count,
            final_chunk=final_chunk,audio=audio,make_loadable=True,make_zip=False,crf=16,
            check_cancel=comfy.model_management.throw_exception_if_processing_interrupted)
        text=json.dumps(result,indent=2)
        return {'ui':{'text':[text]},'result':(text,)}


DIRECT_PROMPT=("Replace the selected person in <Video 1> with the same character shown in <Picture 1>. "
    "Keep the reference character's face, hair, body proportions and clothing. "
    "Follow the source performance's pose, gaze, expression, gestures, object interactions and movement at the source timing. "
    "Preserve the source scene, lighting, camera and other people. The selected performer is replaced, not duplicated.")

NODE_CLASS_MAPPINGS={c.__name__:c for c in (H3SwapDiskChain,H3SwapDiskChainFaceBody,H3SwapDiskHistory,H3SwapDiskSave,H3SwapClipPrep)}
NODE_DISPLAY_NAME_MAPPINGS={
    'H3SwapDiskChainFaceBody':'H3 Low-Memory Chain: FACE + BODY (experimental)',
    'H3SwapDiskChain':'H3 Low-Memory Chain: Start / Review / Resume',
    'H3SwapDiskHistory':'H3 Load Accepted Frame History',
    'H3SwapDiskSave':'H3 Save Lossless Chunk + Checkpoint',
    'H3SwapClipPrep':'H3 Split Long Video Into 5.17s Clips'}

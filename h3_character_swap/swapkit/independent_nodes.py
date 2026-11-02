"""Manual independent native renders and an on-disk joiner for both model profiles."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid
from .config import data_root, atomic_json, file_sha
from . import independent_media as media
from .recovery import media_metadata, sidecars


def _cancel():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


def _preview(path):
    import folder_paths
    from comfy_api.latest import ui, io
    rel=Path(path).resolve().relative_to(Path(folder_paths.get_output_directory()).resolve())
    return ui.PreviewVideo([ui.SavedResult(rel.name,rel.parent.as_posix(),io.FolderType.output)]).as_dict()


class H3SwapPrepareIndependentClips:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'source_video':(media.choices(include_output=False),),
            'set_name':('STRING',{'default':'shot48_independent'}),
            'duration_seconds':('FLOAT',{'default':48.,'min':0.,'max':3600.,'step':.01}),
            'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':86400.,'step':.001}),
            'window_frames':([str(n) for n in media.WINDOWS[::-1]],),
            'overlap_frames':('INT',{'default':12,'min':0,'max':24})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('clip_list_and_instructions',)
    FUNCTION='prepare';OUTPUT_NODE=True;CATEGORY='H3 Swap/Independent clips'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    def prepare(self,source_video,set_name,duration_seconds,start_seconds,window_frames,overlap_frames):
        manifest,path=media.split_source(source_video,set_name,float(duration_seconds),float(start_seconds),
            int(window_frames),int(overlap_frames),cancel=_cancel)
        text=(f"Created {len(manifest['clips'])} independent source sections. Refresh file lists.\n"
              f"Render EACH clip with workflow 12 and the SAME original reference. Do not feed generated results forward.\n"
              f"Join with overlap_frames={overlap_frames}; source_start_seconds={start_seconds}.\n"
              f"Manifest: {path}\n\n"+'\n'.join(r['file'] for r in manifest['clips']))
        return {'ui':{'text':[text]},'result':(text,)}


class H3SwapIndependentSource:
    @classmethod
    def INPUT_TYPES(cls):
        values=[v for v in media.choices(include_output=False) if not v.endswith('/_source_24fps.mkv')]
        return {'required':{'source_clip':(values or ['input/upload_a_clip_first.mkv'],)}}
    RETURN_TYPES=('VIDEO',)
    RETURN_NAMES=('independent_source_video',)
    FUNCTION='load';CATEGORY='H3 Swap/Independent clips'
    @classmethod
    def IS_CHANGED(cls,source_clip):
        p=media.local_media(source_clip);s=p.stat();return (s.st_size,s.st_mtime_ns)
    def load(self,source_clip):
        from comfy_api.latest import InputImpl
        p=media.local_media(source_clip);info=media.video_info(p);media.require_24(info)
        if info['frames']>124:
            raise ValueError('Independent render accepts at most 124 frames. Prepare source clips in workflow 11 first.')
        receipt=p.with_suffix('.source.json')
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r.get('format')!='h3-independent-clip-v1' or r['frames']!=info['frames'] or r['sha256']!=file_sha(p):
                raise ValueError('Prepared source clip changed after splitting. Make a new source set.')
        return (InputImpl.VideoFromFile(str(p)),)


class H3SwapSaveIndependentChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'images':('IMAGE',),'source_video':('VIDEO',),
            'take_name':('STRING',{'default':'shot48_take1'}),
            'crf':('INT',{'default':16,'min':0,'max':30})},
            'hidden':{'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
    RETURN_TYPES=('VIDEO','STRING')
    RETURN_NAMES=('preview_video','join_filename')
    OUTPUT_NODE=True;FUNCTION='save';CATEGORY='H3 Swap/Independent clips'
    def save(self,images,source_video,take_name,crf,prompt=None,extra_pnginfo=None):
        import torch
        from comfy_api.latest import InputImpl
        from .media import source_file
        from .precision import profile
        from .disk_storage import write_master
        from comfy.cli_args import args
        media.safe_name(take_name)
        p,start,duration=source_file(source_video)
        if start or duration or getattr(source_video,'_VideoFromFile__crop',None):
            raise ValueError('Independent saving needs the unwrapped source clip from workflow 12.')
        p=media.local_media(str(p));info=media.video_info(p);media.require_24(info)
        n=info['frames']
        if n>124 or images.ndim!=4 or len(images)!=n or images.shape[-1]!=3:
            raise ValueError('Independent output must match the complete short source interval, including its final tail.')
        if not torch.isfinite(images).all(): raise ValueError('Generated frames contain nonfinite pixels. This is not a metadata cache-marker error.')
        metadata={} if args.disable_metadata else media_metadata(prompt,extra_pnginfo)
        root=data_root().resolve();parent=root/'output/h3_character_swap/independent'/take_name
        parent.mkdir(parents=True,exist_ok=True)
        stem=''.join(c if c.isalnum() or c in '_-' else '_' for c in p.stem)[:64]
        folder=parent/(stem+'_'+uuid.uuid4().hex[:10]);folder.mkdir()
        raw=folder/'raw.partial.mkv';master=folder/'master.mkv'
        write_master(images,raw,_cancel)
        cmd=media._ff()+['-i',str(raw),'-i',str(p)]
        mi,mo=media._metadata_args(metadata,folder,2);cmd+=mi
        cmd+=['-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','pcm_s16le',
              '-t',f'{n/24:.9f}']+mo+[str(master)]
        media.run_ffmpeg(cmd,folder/'master.log',_cancel)
        if media.video_info(master)['frames']!=n: raise ValueError('Independent master frame-count check failed.')
        raw.unlink(missing_ok=True);sidecars(master,metadata)
        mp4=folder/'preview.mp4';media.make_mp4(master,mp4,int(crf),_cancel);sidecars(mp4,metadata)
        origin=p.with_suffix('.source.json')
        references={}
        for node in (prompt or {}).values():
            if isinstance(node,dict) and node.get('class_type')=='LoadImage':
                name=node.get('inputs',{}).get('image')
                if isinstance(name,str):
                    ref=(root/'input'/name).resolve()
                    if ref.is_relative_to(root/'input') and ref.is_file():
                        references[name]=file_sha(ref)
        receipt={'format':'h3-independent-render-v1','source':str(p),'source_sha256':file_sha(p),
                 'frames':n,'fps':24,'width':int(images.shape[2]),'height':int(images.shape[1]),
                 'model_profile':profile(),'master_sha256':file_sha(master),
                 'preview_sha256':file_sha(mp4), 'source_clip':json.loads(origin.read_text()) if origin.exists() else None,
                 'history':'None: this section was generated independently.',
                 'metadata_saved':bool(metadata),'master':str(master),'reference_sha256':references}
        atomic_json(folder/'receipt.json',receipt)
        name=master.relative_to(root).as_posix()
        text=name+'\n\nCopy the first line into workflow 13 in timeline order. Use master.mkv, not the compressed preview.'
        return {'ui':{**_preview(mp4),'text':[text]},'result':(InputImpl.VideoFromFile(str(mp4)),name)}


class H3SwapJoinIndependentClips:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'ordered_videos':('STRING',{'default':'','multiline':True,
                'tooltip':'One input/... or output/... filename per line, in chronological order. Prefer master.mkv.'}),
            'join_name':('STRING',{'default':'shot48_join01'}),
            'overlap_frames':('INT',{'default':12,'min':0,'max':24}),
            'cut_placement':(['Middle of overlap','Keep earlier take','Keep later take'],),
            'audio_mode':(['Original source','No audio'],),
            'original_source':(media.choices(include_output=False),),
            'source_start_seconds':('FLOAT',{'default':0.,'min':0.,'max':86400.,'step':.001}),
            'crf':('INT',{'default':16,'min':0,'max':30})},
            'hidden':{'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
    RETURN_TYPES=('VIDEO','STRING');RETURN_NAMES=('joined_video','report')
    OUTPUT_NODE=True;FUNCTION='join';CATEGORY='H3 Swap/Independent clips'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    @classmethod
    def VALIDATE_INPUTS(cls, original_source, audio_mode, **kwargs):
        if audio_mode == 'No audio': return True
        try: media.local_media(original_source)
        except (ValueError, OSError) as exc: return str(exc)
        return True
    def join(self,ordered_videos,join_name,overlap_frames,cut_placement,audio_mode,original_source,
             source_start_seconds,crf,prompt=None,extra_pnginfo=None):
        from comfy_api.latest import InputImpl
        from comfy.cli_args import args
        metadata={} if args.disable_metadata else media_metadata(prompt,extra_pnginfo)
        report=media.join_videos(ordered_videos,join_name,int(overlap_frames),cut_placement,audio_mode,
            original_source,float(source_start_seconds),int(crf),metadata,cancel=_cancel)
        text=json.dumps(report,indent=2)
        return {'ui':{**_preview(report['video']),'text':[text]},
                'result':(InputImpl.VideoFromFile(report['video']),text)}


class H3SwapRecoverPastWorkflow:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'generation_video':(media.choices(),)}}
    RETURN_TYPES=('STRING',);RETURN_NAMES=('recovered_paths',)
    OUTPUT_NODE=True;FUNCTION='recover';CATEGORY='H3 Swap/Preview and recovery'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    def recover(self,generation_video):
        from .past_recovery import recover_video
        result=recover_video(media.local_media(generation_video))
        text=json.dumps(result,indent=2)
        return {'ui':{'text':[text]},'result':(text,)}


_CLASSES=(H3SwapPrepareIndependentClips,H3SwapIndependentSource,H3SwapSaveIndependentChunk,
          H3SwapJoinIndependentClips,H3SwapRecoverPastWorkflow)
NODE_CLASS_MAPPINGS={x.__name__:x for x in _CLASSES}
NODE_DISPLAY_NAME_MAPPINGS={
 'H3SwapPrepareIndependentClips':'H3 Prepare Independent Source Clips (editorial overlap)',
 'H3SwapIndependentSource':'H3 Load One Independent Clip (124-frame maximum)',
 'H3SwapSaveIndependentChunk':'H3 Save Independent Master + Preview + Workflow',
 'H3SwapJoinIndependentClips':'H3 Join Selected Videos + Original Audio',
 'H3SwapRecoverPastWorkflow':'H3 Recover Workflow from a Past Video'}

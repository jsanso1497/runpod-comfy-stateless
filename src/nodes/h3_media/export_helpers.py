"""Export H.264 MP4, then decode its final displayed frame for chaining.

No reverse filter, full-video tensor, guessed timestamp, or filename glob is used.
The exact completed export travels through an explicit graph dependency.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import uuid


def safe_output(relative, root):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('Invalid exported-video path.')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Export must stay inside ComfyUI/output.')
    return path


def decode_last_frame(path, interrupt=lambda: None, timeout=180):
    """Stream RGB frames, retain one complete frame, and count actual decoded frames."""
    path = Path(path)
    info = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate','-of','json',str(path)],
        check=True, capture_output=True, text=True, timeout=20)
    rows = json.loads(info.stdout).get('streams', [])
    if not rows: raise ValueError('The exported file has no video stream.')
    w,h = int(rows[0]['width']), int(rows[0]['height'])
    if not 1 <= w*h <= 16_777_216: raise ValueError('Unsupported exported frame size.')
    from fractions import Fraction
    rate = Fraction(rows[0].get('avg_frame_rate','0/1'))
    size = w*h*3
    buffer = bytearray(); last = None; count = 0
    cmd = ['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-an','-sn',
           '-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    with tempfile.TemporaryFile() as error_log:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=error_log)
        selector = selectors.DefaultSelector(); selector.register(child.stdout,selectors.EVENT_READ)
        deadline = time.monotonic()+timeout
        try:
            ended = False
            while not ended:
                interrupt()
                if time.monotonic()>deadline: raise TimeoutError('Final-frame extraction exceeded its time budget.')
                for key,_ in selector.select(.2):
                    chunk=os.read(key.fileobj.fileno(),262144)
                    if not chunk:
                        ended=True;break
                    buffer.extend(chunk)
                    while len(buffer)>=size:
                        last=bytes(buffer[:size]); del buffer[:size]; count+=1
                        if count>10000: raise ValueError('Export is too long for the portrait final-frame helper.')
            if child.wait(timeout=10)!=0: raise RuntimeError('ffmpeg could not decode the completed export.')
            if buffer or last is None: raise ValueError('Export ended without a complete video frame.')
        finally:
            selector.close()
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            child.stdout.close()
    import numpy as np
    return np.frombuffer(last,dtype=np.uint8).reshape(h,w,3).copy(), {
        'decoded_frames': count, 'last_frame_index': count-1, 'width':w,'height':h,
        'average_fps':float(rate), 'source':'last decoded frame of completed MP4'}


class H3PortraitExportVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'video':('VIDEO',),
            'filename_prefix':('STRING',{'default':'H3_Portrait/video'}),
            'crf':('INT',{'default':18,'min':0,'max':30})},
            'hidden':{'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
    RETURN_TYPES=('H3_EXPORTED_VIDEO',)
    RETURN_NAMES=('completed_export',)
    FUNCTION='save'
    CATEGORY='H3 Portrait/export'
    OUTPUT_NODE=True
    DESCRIPTION='Save an H.264 MP4 with audio and provide its exact completed path to the last-frame node.'
    def save(self,video,filename_prefix,crf=18,prompt=None,extra_pnginfo=None):
        import folder_paths
        from comfy_api.latest import Types
        from comfy.cli_args import args
        root=Path(folder_paths.get_output_directory()).resolve()
        w,h=video.get_dimensions()
        folder,name,counter,subfolder,_=folder_paths.get_save_image_path(filename_prefix,str(root),w,h)
        filename=f'{name}_{counter:05}_{uuid.uuid4().hex[:6]}.mp4'
        relative=(Path(subfolder)/filename).as_posix()
        path=safe_output(relative,root);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.stem+'.partial.mp4')
        metadata=None
        if not args.disable_metadata:
            metadata=dict(extra_pnginfo or {})
            if prompt is not None: metadata['prompt']=prompt
        try:
            video.save_to(str(temp),format=Types.VideoContainer('mp4'),codec=Types.VideoCodec('h264'),
                          metadata=metadata,crf=float(crf))
            if not temp.is_file() or temp.stat().st_size==0: raise RuntimeError('No MP4 was written.')
            os.replace(temp,path)
        finally:temp.unlink(missing_ok=True)
        manifest={'relative_path':relative,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        return {'ui':{'images':[{'filename':filename,'subfolder':subfolder,'type':'output'}],'animated':(True,)},
                'result':(manifest,)}


class H3PortraitSaveLastFrame:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'completed_export':('H3_EXPORTED_VIDEO',)}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('last_frame','saved_png')
    FUNCTION='save'
    CATEGORY='H3 Portrait/export'
    OUTPUT_NODE=True
    DESCRIPTION='Decode the final displayed frame of the completed MP4 and save a matching _last.png for a later clip.'
    def save(self,completed_export):
        import folder_paths
        import comfy.model_management as mm
        import torch
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        root=Path(folder_paths.get_output_directory()).resolve()
        path=safe_output(completed_export['relative_path'],root)
        if not path.is_file(): raise ValueError('Completed export is missing. Rerun the export node.')
        stat=path.stat()
        if stat.st_size!=completed_export['bytes'] or stat.st_mtime_ns!=completed_export['mtime_ns']:
            raise ValueError('The exported MP4 changed after saving. Rerun the export node.')
        array,details=decode_last_frame(path,mm.throw_exception_if_processing_interrupted)
        out=path.with_name(path.stem+'_last.png'); temp=out.with_suffix('.partial.png')
        details['video']=path.relative_to(root).as_posix()
        metadata=PngInfo();metadata.add_text('h3_last_frame',json.dumps(details))
        try:
            Image.fromarray(array).save(temp,format='PNG',pnginfo=metadata)
            os.replace(temp,out)
        finally: temp.unlink(missing_ok=True)
        out.with_suffix('.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
        print(f'H3 LAST FRAME SAVED: {out.name} | decoded frame {details["last_frame_index"]}',flush=True)
        image=torch.from_numpy(array.copy()).float().div(255)[None]
        relative=out.relative_to(root)
        return {'ui':{'images':[{'filename':out.name,'subfolder':str(relative.parent),'type':'output'}]},
                'result':(image,relative.as_posix())}

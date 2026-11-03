"""Simple control node for automatic independent native renders."""
from . import economy_batch as batch
from .config import data_root
from .independent_media import choices
from .recovery import SIMPLE_PROMPT
import json


def references():
    root=data_root()/'input'
    return sorted('input/'+p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')) or ['input/character.png']


class H3SwapEconomyBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'action':(list(batch.ACTIONS),), 'run_name':('STRING',{'default':'economy_test10'}),
            'source_video':(choices(include_output=False),),'character_image':(references(),),
            'duration_seconds':('FLOAT',{'default':10.,'min':0.1,'max':600.,'step':.01}),
            'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':86400.,'step':.001}),
            'resolution':(['768','480'],), 'window_frames':(['124','107','90','73','56'],),
            'overlap_frames':('INT',{'default':12,'min':0,'max':24}),
            'target_person':('STRING',{'default':'person'}),
            'selection':(['largest','leftmost','rightmost','index'],),
            'selection_index':('INT',{'default':0,'min':0,'max':100}),
            'mask_expansion':('INT',{'default':64,'min':0,'max':256}),
            'seed':('INT',{'default':904239,'min':0,'max':0xffffffffffffffff}),
            'review_policy':(list(batch.REVIEWS),), 'cleanup_policy':(list(batch.CLEANUPS),),
            'prompt':('STRING',{'default':SIMPLE_PROMPT,'multiline':True})}}
    RETURN_TYPES=('STRING',);RETURN_NAMES=('status_and_review_link',)
    FUNCTION='control';OUTPUT_NODE=True;CATEGORY='H3 Swap/Economy'
    @classmethod
    def IS_CHANGED(cls,**kwargs):return float('nan')
    @classmethod
    def VALIDATE_INPUTS(cls,action,source_video,character_image,**kwargs):
        if action!='Start new run':return True
        try:
            from .independent_media import local_media
            local_media(source_video);batch._reference(data_root().resolve(),character_image)
        except (OSError,ValueError) as exc:return str(exc)
        return True
    def control(self,action,run_name,**values):
        if action=='Start new run':
            run=batch.create(run_name=run_name,**values);batch.launch(run);result=batch.summary(run)
        else:result=batch.action(run_name,action)
        text=json.dumps(result,indent=2)
        return {'ui':{'text':[text]},'result':(text,)}

NODE_CLASS_MAPPINGS={'H3SwapEconomyBatch':H3SwapEconomyBatch}
NODE_DISPLAY_NAME_MAPPINGS={'H3SwapEconomyBatch':'H3 Economy: automatic independent chunks'}

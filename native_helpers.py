"""Small helpers for the standard native Ref2VA graph; no Ollama requests."""
from pathlib import Path


class H3PortraitOptionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required':{'model':('MODEL',),'enabled':('BOOLEAN',{'default':False}),
            'lora_name':(['(none)']+folder_paths.get_filename_list('loras'),{'default':'(none)'}),
            'strength':('FLOAT',{'default':1.0,'min':-2.,'max':2.,'step':.05})}}
    RETURN_TYPES=('MODEL',)
    FUNCTION='apply'
    CATEGORY='H3 Portrait/native'
    def apply(self,model,enabled,lora_name,strength):
        if not enabled or lora_name=='(none)' or strength==0:return (model,)
        import folder_paths
        from . import logic, load_loras
        entries=logic.active_loras(True,[(lora_name,strength)],Path(folder_paths.models_dir)/'loras')
        return (load_loras(model,entries),)


class H3PortraitCropToAspect:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'images':('IMAGE',),'aspect':(['9:16','2:3'],{'default':'9:16'})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='crop'
    CATEGORY='H3 Portrait/native'
    def crop(self,images,aspect):
        if images.ndim==5 and images.shape[0]==1:images=images[0]
        if images.ndim!=4:raise ValueError('Expected [frames,height,width,channels].')
        h,w=images.shape[1:3];a,b={'9:16':(9,16),'2:3':(2,3)}[aspect]
        # Smallest even multiple preserves exact ratio and H.264 4:2:0 dimensions.
        scale=min(w//a,h//b);scale-=scale%2
        if scale<2:raise ValueError('Image is too small for the selected output ratio.')
        ow,oh=a*scale,b*scale;x=(w-ow)//2;y=(h-oh)//2
        return (images[:,y:y+oh,x:x+ow,:],)

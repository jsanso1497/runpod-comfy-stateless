"""Protected native loaders. No inference, model precision or tensor math changes."""
from pathlib import Path
from .policy import SPECS, check_selection


class _SafeLoader:
    CATEGORY = 'Model Safety'
    FUNCTION = 'load'
    DESCRIPTION = 'Restricted to this workflow family. A wrong model is rejected before any weights are loaded.'

    @classmethod
    def INPUT_TYPES(cls):
        native, field, _, allowed, encoder_type = SPECS[cls.__name__]
        required = {field: (list(allowed), {'default': allowed[0]})}
        if native == 'UNETLoader':
            required['weight_dtype'] = (['default'], {'default': 'default'})
        elif native == 'CLIPLoader':
            required['type'] = ([encoder_type], {'default': encoder_type})
        schema = {'required': required}
        if native == 'CLIPLoader':
            schema['optional'] = {'device': (['default', 'cpu'], {'default': 'default'})}
        return schema

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        try:
            category, filename = check_selection(cls.__name__, kwargs)
            import folder_paths
            path = folder_paths.get_full_path(category, filename)
            if not path or not Path(path).is_file():
                return f'{cls.__name__}: missing models/{category}/{filename}. Finish asset startup; do not select a different model as a substitute.'
        except (ValueError, OSError) as e:
            return str(e)
        return True

    def load(self, **kwargs):
        error = self.VALIDATE_INPUTS(**kwargs)
        if error is not True:
            raise ValueError(error)
        import nodes
        native = SPECS[self.__class__.__name__][0]
        print(f'MODEL SAFETY: {self.__class__.__name__} verified.', flush=True)
        if native == 'UNETLoader':
            return nodes.UNETLoader().load_unet(kwargs['unet_name'], 'default')
        if native == 'CLIPLoader':
            return nodes.CLIPLoader().load_clip(kwargs['clip_name'], kwargs['type'], kwargs.get('device', 'default'))
        return nodes.VAELoader().load_vae(kwargs['vae_name'])


class SafeKreaUNETLoader(_SafeLoader): RETURN_TYPES = ('MODEL',)
class SafeKreaCLIPLoader(_SafeLoader): RETURN_TYPES = ('CLIP',)
class SafeKreaVAELoader(_SafeLoader): RETURN_TYPES = ('VAE',)
class SafeH3UNETLoader(_SafeLoader): RETURN_TYPES = ('MODEL',)
class SafeH3CLIPLoader(_SafeLoader): RETURN_TYPES = ('CLIP',)
class SafeH3VideoVAELoader(_SafeLoader): RETURN_TYPES = ('VAE',)
class SafeH3AudioVAELoader(_SafeLoader): RETURN_TYPES = ('VAE',)

NODE_CLASS_MAPPINGS = {name: globals()[name] for name in SPECS}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SafeKreaUNETLoader': 'KREA 2 ONLY: diffusion model',
    'SafeKreaCLIPLoader': 'KREA 2 ONLY: 4B image text encoder',
    'SafeKreaVAELoader': 'KREA 2 ONLY: image VAE',
    'SafeH3UNETLoader': 'H3 ONLY: Ref2VA diffusion model',
    'SafeH3CLIPLoader': 'H3 ONLY: 32B H3 text encoder',
    'SafeH3VideoVAELoader': 'H3 ONLY: VIDEO VAE',
    'SafeH3AudioVAELoader': 'H3 ONLY: AUDIO VAE',
}

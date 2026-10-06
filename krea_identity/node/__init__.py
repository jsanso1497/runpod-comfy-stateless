"""Small local image helpers. Diffusion/editing/rebalancing stay in upstream nodes.

No remote inference, face recognition, identity training, or reference beautification.
All tensors use ComfyUI IMAGE layout [1, height, width, channels].
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

VERSION = '1.0.0'
REFERENCE_MODES = ['Face + body sheet', 'Face only', 'Body only']
CLOTHING = ['Keep scene clothing', 'Use body reference clothing']


def image_check(image, name='image'):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f'{name} must be a ComfyUI IMAGE tensor.')
    if image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f'{name}: use one RGB/RGBA photograph, not a batch or video.')
    if min(image.shape[1:3]) < 16 or not image.is_floating_point():
        raise ValueError(f'{name} must have floating pixels and be at least 16 x 16.')
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f'{name} contains invalid pixels.')
    return image[..., :3]


def canvas_size(width, height, megapixels):
    if not math.isfinite(megapixels) or not 0.25 <= megapixels <= 2.0:
        raise ValueError('Krea edit canvas must be between 0.25 and 2.0 megapixels.')
    if min(width, height) < 16 or max(width / height, height / width) > 5:
        raise ValueError('Use a photograph with an aspect ratio between 1:5 and 5:1.')
    scale = math.sqrt(megapixels * 1_000_000 / (width * height))
    w = max(16, int(width * scale) // 16 * 16)
    h = max(16, int(height * scale) // 16 * 16)
    return w, h


def resize(image, width, height):
    return F.interpolate(image.movedim(-1, 1).float(), size=(height, width),
                         mode='bicubic', align_corners=False, antialias=True
                         ).movedim(1, -1).clamp(0, 1).to(image.dtype)


def fit_inside(image, width, height, background=0.5):
    """Letterbox rather than stretch/crop identity references."""
    image = image_check(image)
    ih, iw = image.shape[1:3]
    scale = min(width / iw, height / ih)
    w, h = max(1, round(iw * scale)), max(1, round(ih * scale))
    out = image.new_full((1, height, width, 3), background)
    x, y = (width-w)//2, (height-h)//2
    out[:, y:y+h, x:x+w] = resize(image, w, h)
    return out


def scene_canvas(image, megapixels):
    """Minimal center crop to the /16 grid, never anisotropic stretching."""
    image = image_check(image, 'scene')
    ih, iw = image.shape[1:3]
    w, h = canvas_size(iw, ih, megapixels)
    scale = max(w/iw, h/ih)
    cw, ch = min(iw, round(w/scale)), min(ih, round(h/scale))
    x, y = (iw-cw)//2, (ih-ch)//2
    return resize(image[:, y:y+ch, x:x+cw], w, h), w, h


def reference_sheet(face, body, mode):
    if mode not in REFERENCE_MODES:
        raise ValueError('Unknown reference mode.')
    face, body = image_check(face, 'face'), image_check(body, 'body')
    if mode == 'Face only':
        return face
    if mode == 'Body only':
        return body
    # Two ORIGINAL photographs, never a generated stand-in. A preview exposes
    # the exact sheet. Labels are kept in the prompt, not burned into the pixels.
    face = face.to(device=body.device, dtype=body.dtype)
    out = body.new_full((1, 1280, 1280, 3), 0.5)
    out[:, :, :640] = fit_inside(face, 640, 1280)
    out[:, :, 640:] = fit_inside(body, 640, 1280)
    return out


def instruction(mode, clothing, extra):
    if mode not in REFERENCE_MODES or clothing not in CLOTHING:
        raise ValueError('Unknown reference/clothing mode.')
    if mode == 'Face only' and clothing == CLOTHING[1]:
        raise ValueError('Face only does not send a body reference to Krea. Keep scene clothing or select Face + body sheet.')
    subject = {
        'Face + body sheet': ('Image 2 is a reference sheet of ONE person: the LEFT panel '
                             'defines facial identity; the RIGHT panel defines physique '
                             'and body proportions. The panels are references, not the output layout.'),
        'Face only': 'Image 2 defines the replacement person\'s facial identity. Retain the scene body pose.',
        'Body only': 'Image 2 defines the replacement person and physique.'
    }[mode]
    wardrobe = ('Keep the clothing in image 1, adjusting its fit naturally.'
                if clothing == CLOTHING[0] else 'Use the clothing shown in the body reference.')
    return ('Replace the person in image 1 with the person in image 2. ' + subject + ' '
            'Preserve image 1\'s pose, camera angle, framing, head orientation, expression, '
            'lighting and background. Match the reference identity without beautifying '
            'or averaging facial proportions. ' + wardrobe + ' '
            'Output one realistic photograph of one person, not a collage or split image. '
            + str(extra).strip()).strip()


class KreaIdentityPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'scene': ('IMAGE',), 'face': ('IMAGE',), 'body': ('IMAGE',),
            'reference_mode': (REFERENCE_MODES,), 'clothing': (CLOTHING,),
            'megapixels': ('FLOAT', {'default': 1.5, 'min': 0.25, 'max': 2.0, 'step': 0.25}),
            'extra_instruction': ('STRING', {'default': '', 'multiline': True})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('scene', 'subject_reference', 'prompt', 'width', 'height')
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity'
    def prepare(self, scene, face, body, reference_mode, clothing, megapixels, extra_instruction):
        ready, w, h = scene_canvas(scene, megapixels)
        return ready, reference_sheet(face, body, reference_mode), instruction(reference_mode, clothing, extra_instruction), w, h


class KreaIdentityPair:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'scene': ('IMAGE',), 'subject': ('IMAGE',),
            'megapixels': ('FLOAT', {'default': 1.5, 'min': 0.25, 'max': 2.0, 'step': 0.25}),
            'instruction': ('STRING', {'default': 'Replace the person in image 1 with the person in image 2. Preserve the scene pose, camera, clothing, lighting and background. Keep the reference identity. Output one photograph of one person.', 'multiline': True})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('scene', 'subject_reference', 'prompt', 'width', 'height')
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity'
    def prepare(self, scene, subject, megapixels, instruction):
        ready, w, h = scene_canvas(scene, megapixels)
        if not str(instruction).strip():
            raise ValueError('An edit instruction is required.')
        return ready, image_check(subject, 'subject'), str(instruction).strip(), w, h


class KreaIdentityRebalanceControl:
    """Select native conditioning or the actual upstream rebalancer output.

    No guessed layer roles, no alternative rebalancer implementation, and no
    double application. Both branches use one switch for a clean fixed-seed A/B.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'positive': ('CONDITIONING',), 'negative': ('CONDITIONING',),
            'rebalanced_positive': ('CONDITIONING',), 'rebalanced_negative': ('CONDITIONING',),
            'enabled': ('BOOLEAN', {'default': False})}}
    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING')
    RETURN_NAMES = ('positive', 'negative')
    FUNCTION = 'select'
    CATEGORY = 'Krea Identity'
    def select(self, positive, negative, rebalanced_positive, rebalanced_negative, enabled):
        if type(enabled) is not bool:
            raise ValueError('Rebalance enabled must be a boolean.')
        return (rebalanced_positive, rebalanced_negative) if enabled else (positive, negative)


class KreaIdentityCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'left': ('FLOAT', {'default': 0.25, 'min': 0.0, 'max': 0.95, 'step': 0.01}),
            'top': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.95, 'step': 0.01}),
            'width': ('FLOAT', {'default': 0.5, 'min': 0.05, 'max': 1.0, 'step': 0.01}),
            'height': ('FLOAT', {'default': 0.5, 'min': 0.05, 'max': 1.0, 'step': 0.01})}}
    RETURN_TYPES = ('IMAGE', 'KREA_IDENTITY_CROP')
    RETURN_NAMES = ('head_crop', 'crop_geometry')
    FUNCTION = 'crop'
    CATEGORY = 'Krea Identity'
    def crop(self, image, left, top, width, height):
        image = image_check(image)
        if not all(math.isfinite(v) for v in (left, top, width, height)):
            raise ValueError('Crop coordinates must be finite.')
        if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError('Crop coordinates are fractions of the image, from 0 to 1.')
        ih, iw = image.shape[1:3]
        x, y = round(left*iw), round(top*ih)
        w, h = min(round(width*iw), iw-x), min(round(height*ih), ih-y)
        if min(w, h) < 16:
            raise ValueError('Crop is too small or outside the image.')
        return image[:, y:y+h, x:x+w], {'x': x, 'y': y, 'width': w, 'height': h, 'image_width': iw, 'image_height': ih}


class KreaIdentityStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'original': ('IMAGE',), 'refined_crop': ('IMAGE',), 'crop_geometry': ('KREA_IDENTITY_CROP',),
            'feather_pixels': ('INT', {'default': 32, 'min': 0, 'max': 256, 'step': 1})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'stitch'
    CATEGORY = 'Krea Identity'
    def stitch(self, original, refined_crop, crop_geometry, feather_pixels):
        original, refined_crop = image_check(original), image_check(refined_crop)
        g = crop_geometry
        if tuple(original.shape[1:3]) != (g['image_height'], g['image_width']):
            raise ValueError('Crop geometry belongs to a different original image size.')
        x,y,w,h = (g[k] for k in ('x','y','width','height'))
        if min(x,y) < 0 or min(w,h) < 16 or x+w > original.shape[2] or y+h > original.shape[1]:
            raise ValueError('Invalid crop geometry.')
        if type(feather_pixels) is not int or not 0 <= feather_pixels <= 256:
            raise ValueError('Invalid feather width.')
        crop = resize(refined_crop.to(original), w, h)
        feather = min(feather_pixels, max(1, min(w,h)//4))
        mask = original.new_ones((h,w))
        if feather:
            xx, yy = torch.arange(w,device=original.device), torch.arange(h,device=original.device)
            dx, dy = torch.minimum(xx, w-1-xx), torch.minimum(yy, h-1-yy)
            mask = (torch.minimum(dy[:,None], dx[None,:]).to(original.dtype)/feather).clamp(0,1)
            mask = mask*mask*(3-2*mask)
        out = original.clone()
        out[:,y:y+h,x:x+w] = crop*mask[None,:,:,None]+original[:,y:y+h,x:x+w]*(1-mask[None,:,:,None])
        return (out,)


class KreaIdentityUpscaleSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'scale': ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.25}),
            'long_edge_cap': ('INT', {'default': 4096, 'min': 1024, 'max': 8192, 'step': 64})}}
    RETURN_TYPES = ('INT', 'INT')
    RETURN_NAMES = ('short_side', 'max_resolution')
    FUNCTION = 'size'
    CATEGORY = 'Krea Identity'
    def size(self, image, scale, long_edge_cap):
        image = image_check(image)
        if not math.isfinite(scale) or not 1 <= scale <= 4 or not 1024 <= long_edge_cap <= 8192:
            raise ValueError('Invalid upscale size.')
        h,w = image.shape[1:3]
        s = min(scale, long_edge_cap/max(h,w))
        target = round(min(h,w)*s)
        if target < 64:
            raise ValueError('Image aspect ratio is too extreme for this upscale cap.')
        return target, int(long_edge_cap)


NODE_CLASS_MAPPINGS = {c.__name__:c for c in (KreaIdentityPrepare,KreaIdentityPair,KreaIdentityRebalanceControl,KreaIdentityCrop,KreaIdentityStitch,KreaIdentityUpscaleSize)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'KreaIdentityPrepare': 'Krea Identity - Face / Body / Scene',
    'KreaIdentityPair': 'Krea Identity - Scene + One Reference',
    'KreaIdentityRebalanceControl': 'Krea Identity - Enable Rebalancer (A/B)',
    'KreaIdentityCrop': 'Krea Identity - Head Crop (fractions)',
    'KreaIdentityStitch': 'Krea Identity - Stitch Head Back',
    'KreaIdentityUpscaleSize': 'Krea Identity - Single Image Upscale Size',
}

"""Small local image helpers. Diffusion/editing/rebalancing stay in upstream nodes.

No remote inference, face recognition, identity training, or reference beautification.
All tensors use ComfyUI IMAGE layout [1, height, width, channels].
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

VERSION = '1.2.0'
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


# Version 1.1 additions. The six original nodes and their input schemas above
# remain unchanged, so already-saved user workflows continue to deserialize.
MAX_REFERENCES = 6
WARDROBE_MODES = ['Keep scene clothing', 'Use labeled wardrobe references', 'Follow extra instructions']


def clean_text(value, name, limit=2000, required=True):
    if not isinstance(value, str):
        raise ValueError(name + ' must be text.')
    result = value.strip()
    if required and not result:
        raise ValueError('Fill in ' + name + '.')
    if len(result) > limit:
        raise ValueError(f'{name} is too long (maximum {limit} characters).')
    return result


def checked_references(references, allow_empty=False):
    if not isinstance(references, (tuple, list)):
        raise ValueError('Connect a labeled reference list, not an image batch.')
    if len(references) > MAX_REFERENCES or (not references and not allow_empty):
        raise ValueError(f'Use 1 to {MAX_REFERENCES} active references PER person. Fewer clear references often work better.')
    result = []
    labels = set()
    for entry in references:
        if not isinstance(entry, dict) or set(entry) != {'image', 'label', 'use_for'}:
            raise ValueError('Invalid reference entry; reconnect the Labeled Reference nodes.')
        label = clean_text(entry['label'], 'reference label', 100)
        if label.casefold() in labels:
            raise ValueError('Reference labels must be unique within each person: ' + label)
        labels.add(label.casefold())
        result.append({'image': image_check(entry['image'], label), 'label': label,
                       'use_for': clean_text(entry['use_for'], 'reference purpose', 600)})
    return tuple(result)


class KreaIdentityReference:
    """Chain these nodes to add original photographs without generating a proxy.

    The first ACTIVE reference occupies the largest tile when there are 3+.
    Purpose labels become an explicit panel-to-instruction map, not a trained
    attribute router or an invented per-reference attention weight.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'label': ('STRING', {'default': 'Primary face'}),
            'use_for': ('STRING', {'default': 'Facial identity, facial proportions and natural skin detail only.', 'multiline': True}),
            'include': ('BOOLEAN', {'default': True}),
        }, 'optional': {'references': ('KREA_IDENTITY_REFS',)}}
    RETURN_TYPES = ('KREA_IDENTITY_REFS',)
    RETURN_NAMES = ('references',)
    FUNCTION = 'add'
    CATEGORY = 'Krea Identity / Directed'
    def add(self, image, label, use_for, include, references=None):
        if type(include) is not bool:
            raise ValueError('Include must be a boolean.')
        previous = checked_references(() if references is None else references, allow_empty=True)
        if not include:
            return (previous,)
        entry = {'image': image_check(image), 'label': label, 'use_for': use_for}
        return (checked_references(previous + (entry,)),)


def caption_strip(width, height, text, prototype):
    # Only ASCII panel IDs are burned into separate margins. User labels stay in
    # the text prompt, avoiding font/Unicode issues and overlays on facial pixels.
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    image = Image.new('RGB', (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=max(12, min(28, int(height * 0.6))))
    except TypeError:  # Older Pillow releases still used by some Comfy images.
        font = ImageFont.load_default()
    draw.text((10, max(1, int(height * 0.08))), text, font=font, fill=(245, 245, 245))
    data = torch.from_numpy(np.asarray(image).copy()).to(device=prototype.device, dtype=prototype.dtype) / 255
    return data[None]


def labeled_sheet(references, prefix='A', width=1536, height=1536):
    refs = checked_references(references)
    if prefix not in ('A', 'B') or min(width, height) < 128:
        raise ValueError('Invalid reference-sheet geometry.')
    proto = refs[0]['image']
    out = proto.new_full((1, height, width, 3), 0.5)
    n = len(refs)
    if n == 1:
        rects = [(0, 0, width, height)]
    elif n == 2:
        half = width // 2
        rects = [(0, 0, half, height), (half, 0, width-half, height)]
    else:
        half = width // 2
        rects = [(0, 0, half, height)]
        columns = 1 if n <= 4 else 2
        rows = math.ceil((n-1) / columns)
        for k in range(n-1):
            col, row = k % columns, k // columns
            x1 = half + (width-half)*col//columns
            x2 = half + (width-half)*(col+1)//columns
            y1, y2 = height*row//rows, height*(row+1)//rows
            rects.append((x1, y1, x2-x1, y2-y1))
    descriptions = []
    for number, (entry, rect) in enumerate(zip(refs, rects), start=1):
        x, y, w, h = rect
        strip = min(44, max(22, h//10))
        ref = entry['image'].to(device=proto.device, dtype=proto.dtype)
        out[:, y:y+strip, x:x+w] = caption_strip(w, strip, f'{prefix}{number}', proto)
        out[:, y+strip:y+h, x:x+w] = fit_inside(ref, w, h-strip)
        descriptions.append(f'{prefix}{number}: {entry["label"]}. Use ONLY for: {entry["use_for"]}')
    return out, '\n'.join(descriptions)


def framed_canvas(image, megapixels):
    """Contain, never center-crop, the source in a /16 diffusion canvas.

    At most a narrow rounding margin is added. Save/composite nodes remove it
    using this exact transform, so the original scene's edges are not discarded.
    """
    image = image_check(image, 'scene')
    ih, iw = image.shape[1:3]
    w, h = canvas_size(iw, ih, megapixels)
    scale = min(w/iw, h/ih)
    rw, rh = max(1, round(iw*scale)), max(1, round(ih*scale))
    x, y = (w-rw)//2, (h-rh)//2
    canvas = fit_inside(image, w, h)
    geometry = {'target_width': w, 'target_height': h, 'x': x, 'y': y,
                'width': rw, 'height': rh, 'original_width': iw, 'original_height': ih}
    return canvas, w, h, geometry


def unframe(image, geometry, original_size=False):
    image = image_check(image, 'generated image')
    keys = {'target_width', 'target_height', 'x', 'y', 'width', 'height', 'original_width', 'original_height'}
    if not isinstance(geometry, dict) or set(geometry) != keys or any(type(v) is not int for v in geometry.values()):
        raise ValueError('Invalid canvas transform; use the geometry from the matching edit.')
    g = geometry
    if image.shape[1:3] != (g['target_height'], g['target_width']):
        raise ValueError('Generated size does not match the prepared canvas; do not override sampler dimensions.')
    if min(g['width'], g['height'], g['original_width'], g['original_height']) < 1 or min(g['x'], g['y']) < 0:
        raise ValueError('Invalid canvas bounds.')
    if g['x']+g['width'] > g['target_width'] or g['y']+g['height'] > g['target_height']:
        raise ValueError('Canvas crop is out of bounds.')
    result = image[:, g['y']:g['y']+g['height'], g['x']:g['x']+g['width']]
    return resize(result, g['original_width'], g['original_height']) if original_size else result


class KreaIdentityUnframe:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'canvas_geometry': ('KREA_IDENTITY_CANVAS',)}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('native_image',)
    FUNCTION = 'restore'
    CATEGORY = 'Krea Identity / Directed'
    def restore(self, image, canvas_geometry):
        return (unframe(image, canvas_geometry),)


def wardrobe_instruction(mode):
    if mode not in WARDROBE_MODES:
        raise ValueError('Choose a valid clothing mode.')
    return {
        WARDROBE_MODES[0]: 'Keep this target\'s original scene clothing; adjust only its fit to the replacement physique.',
        WARDROBE_MODES[1]: 'Use clothing only from panels explicitly labeled for wardrobe or clothing; use identity-only panels for identity, not clothing.',
        WARDROBE_MODES[2]: 'Follow the extra instruction for clothing. Otherwise keep the scene clothing.'
    }[mode]


def subject_instruction(prefix, target, replacement, wardrobe, manifest):
    target = clean_text(target, f'target {prefix} description', 800)
    replacement = clean_text(replacement, f'replacement {prefix} description', 800)
    return (f'REPLACEMENT {prefix}: In image 1, replace ONLY [{target}] with [{replacement}], '
            f'using identity group {prefix} from image 2.\n'
            f'{wardrobe_instruction(wardrobe)}\n{manifest}')


PRESERVE_INSTRUCTION = (
    'Image 1 is the scene and composition authority. Image 2 contains labeled reference photographs, '
    'not a collage to reproduce. Preserve the scene camera, framing, pose, head orientation, expression, '
    'lighting, environment, interactions and person count unless explicitly instructed otherwise. '
    'Do not add people or change unselected people. Do not copy reference poses, backgrounds or '
    'lighting unless a panel is explicitly assigned that role. Match each selected identity\'s '
    'distinctive facial geometry and natural skin detail without beautification, face averaging or '
    'identity mixing. Adapt the requested physique to the existing pose, with realistic anatomy, '
    'clothing fit, contact shadows and occlusions. Output one realistic photograph, never panels, labels or text.'
)


class KreaIdentityDirectedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'scene': ('IMAGE',), 'references': ('KREA_IDENTITY_REFS',),
            'reference_group': (['A', 'B'],),
            'target_description': ('STRING', {'default': 'the woman standing on the right', 'multiline': True}),
            'replacement_description': ('STRING', {'default': 'the woman shown in reference group A', 'multiline': True}),
            'clothing': (WARDROBE_MODES,),
            'extra_instruction': ('STRING', {'default': '', 'multiline': True}),
            'megapixels': ('FLOAT', {'default': 1.5, 'min': 0.25, 'max': 2.0, 'step': 0.25}),
        }}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING', 'INT', 'INT', 'KREA_IDENTITY_CANVAS')
    RETURN_NAMES = ('scene', 'subject_reference', 'prompt', 'width', 'height', 'canvas_geometry')
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity / Directed'
    def prepare(self, scene, references, target_description, replacement_description, clothing, extra_instruction, megapixels, reference_group='A'):
        sheet, manifest = labeled_sheet(references, reference_group)
        prompt = (PRESERVE_INSTRUCTION + '\n\n' + subject_instruction(reference_group, target_description, replacement_description, clothing, manifest)
                  + '\n\nEXTRA INSTRUCTION: ' + clean_text(extra_instruction, 'extra instruction', 2000, required=False)).strip()
        ready, w, h, geometry = framed_canvas(scene, megapixels)
        return ready, sheet, prompt, w, h, geometry


class KreaIdentityCoupleEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'scene': ('IMAGE',), 'references_a': ('KREA_IDENTITY_REFS',), 'references_b': ('KREA_IDENTITY_REFS',),
            'target_a': ('STRING', {'default': 'the man on the left', 'multiline': True}),
            'replacement_a': ('STRING', {'default': 'the man shown in reference group A', 'multiline': True}),
            'clothing_a': (WARDROBE_MODES,),
            'target_b': ('STRING', {'default': 'the woman on the right', 'multiline': True}),
            'replacement_b': ('STRING', {'default': 'the woman shown in reference group B', 'multiline': True}),
            'clothing_b': (WARDROBE_MODES,),
            'extra_instruction': ('STRING', {'default': 'Keep all other people unchanged.', 'multiline': True}),
            'megapixels': ('FLOAT', {'default': 2.0, 'min': 0.25, 'max': 2.0, 'step': 0.25}),
        }}
    RETURN_TYPES = KreaIdentityDirectedEdit.RETURN_TYPES
    RETURN_NAMES = KreaIdentityDirectedEdit.RETURN_NAMES
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity / Directed'
    def prepare(self, scene, references_a, references_b, target_a, replacement_a, clothing_a,
                target_b, replacement_b, clothing_b, extra_instruction, megapixels):
        # Each list is a separate identity namespace, even when labels repeat.
        a, am = labeled_sheet(references_a, 'A', 1536, 1536)
        b, bm = labeled_sheet(references_b, 'B', 1536, 1536)
        if clean_text(target_a, 'target A').casefold() == clean_text(target_b, 'target B').casefold():
            raise ValueError('Targets A and B must identify DIFFERENT people in the scene.')
        # Keep the master reference board at 2048 x 1024, not an unbounded strip.
        # The upstream encoder and FIT latent path still resize it independently.
        sheet = torch.cat((resize(a, 1024, 1024), resize(b.to(a), 1024, 1024)), dim=2)
        prompt = (PRESERVE_INSTRUCTION + '\n\nPerform BOTH replacements in this single photograph. '
                  'Image 2 LEFT half is group A; RIGHT half is group B. Keep the two identities distinct. '
                  'Never use A panels for target B or B panels for target A.\n\n'
                  + subject_instruction('A', target_a, replacement_a, clothing_a, am) + '\n\n'
                  + subject_instruction('B', target_b, replacement_b, clothing_b, bm)
                  + '\n\nEXTRA INSTRUCTION: ' + clean_text(extra_instruction, 'extra instruction', 2000, required=False)).strip()
        ready, w, h, geometry = framed_canvas(scene, megapixels)
        return ready, sheet, prompt, w, h, geometry


def checked_mask(mask, image, name):
    if not isinstance(mask, torch.Tensor):
        raise ValueError(name + ' must be a ComfyUI MASK.')
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or mask.shape != image.shape[:3]:
        raise ValueError(name + ' must match the original scene dimensions exactly. Load/paint the same scene; do not use an empty LoadImage mask.')
    if not mask.is_floating_point() or not bool(torch.isfinite(mask).all()):
        raise ValueError(name + ' must contain finite floating values.')
    if bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError(name + ' values must be between 0 and 1 (white edits, black protects).')
    return mask.to(device=image.device, dtype=torch.float32)


class KreaIdentityRegion:
    """Choose an edit region without guessing person identity or segmentation.

    A painted mask overrides the rectangle. The entire crop is a contextual Krea
    edit; the mask is a final compositing constraint, NOT diffusion inpainting.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'left': ('FLOAT', {'default': 0.02, 'min': 0.0, 'max': 0.99, 'step': 0.01}),
            'top': ('FLOAT', {'default': 0.02, 'min': 0.0, 'max': 0.99, 'step': 0.01}),
            'width': ('FLOAT', {'default': 0.46, 'min': 0.01, 'max': 1.0, 'step': 0.01}),
            'height': ('FLOAT', {'default': 0.96, 'min': 0.01, 'max': 1.0, 'step': 0.01}),
            'context_padding': ('FLOAT', {'default': 0.12, 'min': 0.0, 'max': 0.5, 'step': 0.01}),
            'feather_pixels': ('INT', {'default': 16, 'min': 0, 'max': 128, 'step': 1}),
        }, 'optional': {'edit_mask': ('MASK',), 'protect_mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE', 'KREA_IDENTITY_REGION', 'MASK', 'IMAGE')
    RETURN_NAMES = ('context_crop', 'region_geometry', 'effective_mask', 'region_preview')
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity / Directed'
    def prepare(self, image, left, top, width, height, context_padding, feather_pixels, edit_mask=None, protect_mask=None):
        image = image_check(image)
        ih, iw = image.shape[1:3]
        vals = (left, top, width, height, context_padding)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError('Region coordinates must be finite numbers.')
        if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1 and 0 <= context_padding <= .5):
            raise ValueError('Invalid region fractions or context padding.')
        if type(feather_pixels) is not int or not 0 <= feather_pixels <= 128:
            raise ValueError('Feather must be an integer from 0 to 128 original-scene pixels.')
        if edit_mask is None:
            mask = torch.zeros((1, ih, iw), device=image.device, dtype=torch.float32)
            x, y = round(left*iw), round(top*ih)
            w, h = min(round(width*iw), iw-x), min(round(height*ih), ih-y)
            if min(w, h) < 16:
                raise ValueError('The selected rectangle is too small. Include the whole person plus room for the replacement silhouette.')
            mask[:, y:y+h, x:x+w] = 1
        else:
            mask = checked_mask(edit_mask, image, 'Edit mask').clone()
        if protect_mask is not None:
            # Hard exclusion, not alpha subtraction: no pixel from the first edit,
            # even a feather-edge pixel, may be changed by this second composite.
            protected = checked_mask(protect_mask, image, 'Protect mask') > 0
            mask = torch.where(protected, torch.zeros_like(mask), mask)
        active = mask[0] > 0
        if not bool(active.any()):
            raise ValueError('The edit mask is empty or entirely protected. Adjust the region or painted mask.')
        ys, xs = torch.where(active)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
        if min(x1-x0, y1-y0) < 16:
            raise ValueError('The active edit region must be at least 16 x 16 pixels.')
        if feather_pixels:
            import numpy as np
            from scipy.ndimage import distance_transform_edt
            # Inward feathering leaves exactly-zero exterior pixels unchanged.
            # Padding prevents an all-white mask from having undefined distances.
            hard = active.detach().cpu().numpy()
            distance = distance_transform_edt(np.pad(hard, 1))[1:-1, 1:-1]
            ramp = torch.from_numpy(distance).to(device=image.device, dtype=torch.float32)
            ramp = (ramp / feather_pixels).clamp(0, 1)
            ramp = ramp*ramp*(3-2*ramp)
            mask = mask * ramp[None]
        px, py = round((x1-x0)*context_padding), round((y1-y0)*context_padding)
        rx, ry = max(0, x0-px), max(0, y0-py)
        rw, rh = min(iw, x1+px)-rx, min(ih, y1+py)-ry
        geometry = {'x': rx, 'y': ry, 'width': rw, 'height': rh,
                    'image_width': iw, 'image_height': ih, 'mask': mask}
        # Bright edit region, dim protected context. This does NOT alter the crop
        # given to Krea, only the separate preview output.
        preview = image * (0.25 + 0.75*mask[..., None].to(image.dtype))
        return image[:, ry:ry+rh, rx:rx+rw], geometry, mask, preview


class KreaIdentityRegionComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'original': ('IMAGE',), 'generated': ('IMAGE',),
                             'region_geometry': ('KREA_IDENTITY_REGION',),
                             'canvas_geometry': ('KREA_IDENTITY_CANVAS',)}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('full_scene',)
    FUNCTION = 'composite'
    CATEGORY = 'Krea Identity / Directed'
    def composite(self, original, generated, region_geometry, canvas_geometry):
        original = image_check(original)
        g = region_geometry
        required = {'x', 'y', 'width', 'height', 'image_width', 'image_height', 'mask'}
        if not isinstance(g, dict) or set(g) != required:
            raise ValueError('Invalid region geometry.')
        if any(type(g[k]) is not int for k in required - {'mask'}):
            raise ValueError('Region bounds must be integer pixels.')
        if original.shape[1:3] != (g['image_height'], g['image_width']):
            raise ValueError('Composite original differs from the region source dimensions.')
        if min(g['x'],g['y']) < 0 or min(g['width'],g['height']) < 1 or g['x']+g['width'] > original.shape[2] or g['y']+g['height'] > original.shape[1]:
            raise ValueError('Region is outside the original scene.')
        if (canvas_geometry.get('original_width'), canvas_geometry.get('original_height')) != (g['width'], g['height']):
            raise ValueError('Canvas transform belongs to a different region.')
        crop = unframe(generated, canvas_geometry, original_size=True).to(original)
        mask = checked_mask(g['mask'], original, 'Composite mask').to(original.dtype)
        x, y, w, h = g['x'], g['y'], g['width'], g['height']
        alpha = mask[:, y:y+h, x:x+w, None]
        old = original[:, y:y+h, x:x+w]
        # Explicit where preserves protected pixels bit-for-bit, including low
        # precision original tensors, rather than relying on float arithmetic.
        blended = torch.where(alpha > 0, crop*alpha + old*(1-alpha), old)
        out = original.clone()
        out[:, y:y+h, x:x+w] = blended
        return (out,)


NODE_CLASS_MAPPINGS.update({c.__name__: c for c in (
    KreaIdentityReference, KreaIdentityDirectedEdit, KreaIdentityCoupleEdit,
    KreaIdentityUnframe, KreaIdentityRegion, KreaIdentityRegionComposite)})
NODE_DISPLAY_NAME_MAPPINGS.update({
    'KreaIdentityReference': 'Krea Identity - Labeled Reference (chain to add)',
    'KreaIdentityDirectedEdit': 'Krea Identity - Who Replaces Whom',
    'KreaIdentityCoupleEdit': 'Krea Identity - Two People / Separate Reference Groups',
    'KreaIdentityUnframe': 'Krea Identity - Remove Canvas Margins',
    'KreaIdentityRegion': 'Krea Identity - Protected Region / Optional Painted Mask',
    'KreaIdentityRegionComposite': 'Krea Identity - Composite Only Selected Region',
})


# Version 1.2 addition: two full, separate references and NO source scene.
TEXT_SCENE_MODES = ['One person - complementary references', 'Two people - separate identities']
TEXT_SCENE_ASPECTS = {
    '3:2 landscape': (3, 2), '16:9 landscape': (16, 9),
    '4:3 landscape': (4, 3), '1:1 square': (1, 1),
    '4:5 portrait': (4, 5), '2:3 portrait': (2, 3), '9:16 portrait': (9, 16),
}


def described_canvas(aspect, megapixels):
    if aspect not in TEXT_SCENE_ASPECTS:
        raise ValueError('Choose a supported landscape, portrait or square aspect ratio.')
    if not isinstance(megapixels, (float, int)) or isinstance(megapixels, bool) or not math.isfinite(megapixels) or not .25 <= megapixels <= 2.0:
        raise ValueError('Krea working resolution must be 0.25 through 2.0 megapixels.')
    a, b = TEXT_SCENE_ASPECTS[aspect]
    # Exact aspect AND /16 grid, without inventing an image to act as a scene.
    block = max(1, math.floor(math.sqrt(megapixels * 1_000_000 / (a*b)) / 16))
    return a * block * 16, b * block * 16


class KreaIdentityTextSceneTwoRefs:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'reference_a': ('IMAGE',), 'reference_b': ('IMAGE',),
            'reference_relationship': (TEXT_SCENE_MODES, {'default': TEXT_SCENE_MODES[0]}),
            'reference_a_use': ('STRING', {'default': 'Primary facial identity, facial proportions and natural skin detail.', 'multiline': True}),
            'reference_b_use': ('STRING', {'default': 'Physique, body proportions, hair and visible identity details appropriate to the selected reference relationship.', 'multiline': True}),
            'scene_description': ('STRING', {'default': 'A realistic full-body photograph of this person standing on a quiet city sidewalk in soft morning light. Eye-level camera, relaxed natural posture, neutral expression. Plain gray T-shirt and dark jeans. The person is in sharp focus against a softly detailed background.', 'multiline': True}),
            'aspect': (list(TEXT_SCENE_ASPECTS), {'default': '3:2 landscape'}),
            'megapixels': ('FLOAT', {'default': 1.5, 'min': .25, 'max': 2.0, 'step': .25}),
            'extra_instruction': ('STRING', {'default': '', 'multiline': True}),
        }}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('reference_a', 'reference_b', 'prompt', 'width', 'height')
    FUNCTION = 'prepare'
    CATEGORY = 'Krea Identity'
    DESCRIPTION = 'Two original references, one text-described new scene. No scene photograph, blank-image stand-in, or reference collage.'

    def prepare(self, reference_a, reference_b, reference_relationship, reference_a_use,
                reference_b_use, scene_description, aspect, megapixels, extra_instruction=''):
        a, b = image_check(reference_a, 'reference A'), image_check(reference_b, 'reference B')
        if reference_relationship not in TEXT_SCENE_MODES:
            raise ValueError('Choose whether both images describe one person or two different people.')
        for name, value in [('scene_description', scene_description), ('reference_a_use', reference_a_use), ('reference_b_use', reference_b_use)]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(name + ' must contain a description.')
        if not isinstance(extra_instruction, str):
            raise ValueError('Extra instruction must be text.')
        if reference_relationship == TEXT_SCENE_MODES[0]:
            relationship = ('Both images show ONE and the same person. Combine their complementary identity details into that single person. '
                            'Do not create a second person just because there are two references.')
        else:
            relationship = ('Image 1 defines person A; image 2 defines person B. They are TWO distinct people. '
                            'Keep their facial identities separate, with no face averaging or feature mixing. '
                            'Place them as described in the scene. Neither image is a background template.')
        prompt = ('Create one new photorealistic photograph in the scene described below. '
                  'The two supplied images are references only; neither is the target scene. '
                  'Do not reproduce their backgrounds, camera positions or poses unless explicitly requested. '
                  + relationship + '\nImage 1 use: ' + reference_a_use.strip()
                  + '\nImage 2 use: ' + reference_b_use.strip()
                  + '\nNew scene, camera, pose, clothing and lighting: ' + scene_description.strip()
                  + '\nRetain the requested identity geometry, proportions and distinctive details without beautification. '
                    'Use physically plausible anatomy, perspective, lighting and shadows. Output one coherent photograph, '
                    'not panels, a reference sheet, side-by-side views, labels or a collage.')
        if extra_instruction.strip():
            prompt += '\nAdditional direction: ' + extra_instruction.strip()
        w, h = described_canvas(aspect, megapixels)
        return a, b, prompt, w, h


NODE_CLASS_MAPPINGS['KreaIdentityTextSceneTwoRefs'] = KreaIdentityTextSceneTwoRefs
NODE_DISPLAY_NAME_MAPPINGS['KreaIdentityTextSceneTwoRefs'] = 'Krea Identity - Two References + Described Scene'

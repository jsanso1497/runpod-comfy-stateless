"""SAM people selection and protected Qwen repair overlays.

Inference stays in ComfyUI: native SAM 3.1 detects instances, and the stock
Qwen 2.1 subgraph repairs a bounded crop. All geometry/compositing is on CPU.
This module never downloads models or modifies an existing workflow.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import uuid

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import maximum_filter
import torch
import torch.nn.functional as F

VERSION = 'qwen-keep-one-audit-20261018'
SAM_FILE = 'sam3.1_multiplex_fp16.safetensors'
MAX_PEOPLE = 5
# Native SAM defaults to one instance for plain "person". Request a sixth
# only as an overflow sentinel; never silently discard an extra person.
SAM_TEXT = 'person:6'
CATEGORY = 'Qwen 2.1 / Keep one person'
DEFAULT_INSTRUCTIONS = 'Refine the rough background-filled patches in this photograph into realistic, continuous surrounding scenery. Preserve the remaining visible person, their face, pose, clothing and location. Match the existing light, perspective, texture, grain and focus. Complete any newly exposed background and local edges so this looks like a normal photograph taken with only that person present. Produce one fully opaque photographic frame.'


def _image(value, label='Source'):
    if not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3:
        raise ValueError(label + ': load one complete RGB photograph, not a batch or RGBA overlay.')
    if min(value.shape[1:3]) < 1 or not torch.isfinite(value).all().item():
        raise ValueError(label + ': image is empty or contains non-finite pixels.')
    if value.min().item() < 0 or value.max().item() > 1:
        raise ValueError(label + ': pixels must be in [0,1].')
    return value.detach().to(device='cpu', dtype=torch.float32)


def _manual(value, h, w, label):
    if value is None:
        return np.zeros((h, w), dtype=bool)
    value = torch.as_tensor(value).detach().float().cpu()
    if not value.numel():
        raise ValueError(label + ': mask is empty.')
    if not torch.isfinite(value).all().item() or value.min().item() < 0 or value.max().item() > 1:
        raise ValueError(label + ': mask must be finite and in [0,1].')
    # LoadImage supplies a blank 64x64 placeholder when there is no painted mask.
    if not torch.any(value > 0).item():
        return np.zeros((h, w), dtype=bool)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if tuple(value.shape) != (1, h, w):
        raise ValueError(label + ': non-empty masks must match the ORIGINAL image exactly. Repaint/save without resizing.')
    # Every painted pixel, including a soft edge, is a hard protection when
    # this helper is used for manual_protection. It is never silently ignored.
    return value[0].numpy() > 0


def _grow(mask, pixels):
    if type(pixels) is not int or not 0 <= pixels <= 256:
        raise ValueError('Mask growth must be an integer from 0 to 256 original-image pixels.')
    if pixels == 0:
        return mask.copy()
    return maximum_filter(mask, size=2 * pixels + 1, mode='constant', cval=0).astype(bool)


def _bounds(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('An empty mask cannot define a crop.')
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _resize(image, width, height):
    if image.shape[1:3] == (height, width):
        return image
    arr = image[0].numpy()
    channels = [np.asarray(Image.fromarray(arr[..., c]).resize(
        (width, height), Image.Resampling.LANCZOS), dtype=np.float32) for c in range(3)]
    return torch.from_numpy(np.stack(channels, -1).copy()).unsqueeze(0).clamp(0, 1)


def _budget(width, height, resolution, max_side=3072):
    if type(resolution) is not int or resolution % 32 or not 256 <= resolution <= 2048:
        raise ValueError('Working resolution must be 256 to 2048, in 32-pixel steps.')
    # Match the source aspect ratio; only right/bottom padding provides alignment.
    scale = min(1., resolution / math.sqrt(width * height), max_side / max(width, height))
    def dims(s):
        sw, sh = max(1, round(width * s)), max(1, round(height * s))
        return sw, sh, math.ceil(sw / 32) * 32, math.ceil(sh / 32) * 32
    def fits(d):
        return d[2] * d[3] <= resolution ** 2 and max(d[2:]) <= max_side
    if fits(dims(scale)):
        return dims(scale)
    low, high = 0., scale
    for _ in range(60):
        middle = (low + high) / 2
        if fits(dims(middle)):
            low = middle
        else:
            high = middle
    return dims(low)


def _work_image(image, resolution, max_side=3072):
    sw, sh, ww, wh = _budget(image.shape[2], image.shape[1], resolution, max_side)
    resized = _resize(image, sw, sh)
    result = F.pad(resized.movedim(-1, 1), (0, ww-sw, 0, wh-sh), mode='replicate').movedim(1, -1)
    return result, (sw, sh, ww, wh)


def people_from_masks(source, masks, expected_people=0):
    """Build reproducible left-to-right IDs from original-coordinate masks."""
    source = _image(source)
    h, w = source.shape[1:3]
    if not torch.is_tensor(masks) or masks.ndim != 3 or tuple(masks.shape[1:]) != (h, w):
        raise ValueError('SAM instance masks do not match the original image; refusing to resize selections.')
    if not torch.isfinite(masks).all().item() or (masks.numel() and (masks.min().item() < 0 or masks.max().item() > 1)):
        raise ValueError('SAM returned invalid mask values.')
    count = masks.shape[0]
    if count == 0:
        raise ValueError('SAM found no people. Adjust threshold and verify the original photograph.')
    if count > MAX_PEOPLE:
        raise ValueError('SAM found more than five people. This workflow supports 1-5; no people were silently discarded.')
    if type(expected_people) is not int or not 0 <= expected_people <= MAX_PEOPLE:
        raise ValueError('Expected people must be 0 (automatic) or 1-5.')
    if expected_people and count != expected_people:
        raise ValueError(f'Expected {expected_people} people but SAM detected {count}. Review detection before editing.')
    binary = masks.detach().cpu().numpy() > 0.5
    rows = []
    for i, mask in enumerate(binary):
        if not mask.any():
            raise ValueError('SAM returned an empty person instance. Adjust the threshold rather than ignoring it.')
        box = _bounds(mask)
        # Digest breaks geometric ties without depending on SAM's confidence order.
        key = ((box[0]+box[2])/2, (box[1]+box[3])/2, box,
               hashlib.sha256(np.packbits(mask).tobytes()).hexdigest())
        rows.append((key, i, box))
    rows.sort(key=lambda row: row[0])
    ordered = binary[[r[1] for r in rows]]
    return {'source': source, 'masks': ordered, 'boxes': [r[2] for r in rows],
            'count': count, 'numbering': 'left to right by bounding-box center; then top to bottom',
            'version': VERSION}


def prepare_people(detections, keep_person=1, resolution=2048, context_pixels=128,
                   repair_grow_pixels=24, keep_grow_pixels=2, protection_grow_pixels=0,
                   instructions=DEFAULT_INSTRUCTIONS, manual_protection=None, extra_repair_mask=None):
    source = _image(detections['source'])
    h, w = source.shape[1:3]
    count = detections['count']
    if type(keep_person) is not int or not 1 <= keep_person <= count:
        raise ValueError(f'Choose a detected person from 1 to {count}, using the numbered preview.')
    if type(context_pixels) is not int or not 0 <= context_pixels <= 2048:
        raise ValueError('Context must be 0-2048 original-image pixels.')
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError('Enter the removal/repair instruction.')
    masks = detections['masks']
    keeper = masks[keep_person-1]
    raw_remove = np.zeros((h, w), dtype=bool)
    for i, mask in enumerate(masks):
        if i != keep_person-1:
            raw_remove |= mask
    manual = _manual(manual_protection, h, w, 'Manual PROTECTION')
    extra = _manual(extra_repair_mask, h, w, 'Extra REPAIR')
    keep_protection = _grow(keeper, keep_grow_pixels)
    manual = _grow(manual, protection_grow_pixels)
    protection = keep_protection | manual
    expanded = _grow(raw_remove, repair_grow_pixels) | extra
    allowed = expanded & ~protection
    if not allowed.any() and raw_remove.any():
        raise ValueError('Protection covers all removal pixels. Review the painted protection and person selection.')
    # Include all detections so image1 always shows the survivor, even when the
    # removed people are distant. Context does NOT expand the final paste mask.
    framing = masks.any(axis=0) | allowed
    x, y, x2, y2 = _bounds(framing)
    x, y = max(0, x-context_pixels), max(0, y-context_pixels)
    x2, y2 = min(w, x2+context_pixels), min(h, y2+context_pixels)
    crop = source[:, y:y2, x:x2, :]
    work, (sw, sh, ww, wh) = _work_image(crop, resolution)
    from .qkeep_repair import prefill
    # Exclude ALL detected people and protected areas from background donors.
    donors = ~(masks.any(axis=0) | allowed | protection)
    work, prefill_info = prefill(work, allowed[y:y2,x:x2], donors[y:y2,x:x2], sw, sh)
    plan = SimpleNamespace(source_width=w, source_height=h, x=x, y=y,
                           crop_width=x2-x, crop_height=y2-y,
                           scaled_width=sw, scaled_height=sh, work_width=ww, work_height=wh)
    # Old saved graphs still have an image2 port. Return a complete scene,
    # never the isolated gray-backdrop person that encouraged cutout outputs.
    ref = _work_image(work, 768, 1024)[0]
    instruction = instructions.strip()
    if ('Remove all people from <image1> except' in instruction or
            instruction.startswith('Replace every bright green region') or
            instruction.startswith('Retouch the already-filled background patches')):
        instruction = DEFAULT_INSTRUCTIONS
    prompt = instruction + (
        f' The rough background patches replace {count-1} unwanted person(s). '
        'Refine the patches into believable photographic background. '
        'Keep the one visible person in place. Return the complete scene.')
    report = {'version': VERSION, 'people_detected': count, 'keep_person': keep_person,
              'remove_people': [i for i in range(1,count+1) if i != keep_person],
              'numbering': detections['numbering'], 'person_boxes_xyxy': [list(b) for b in detections['boxes']],
              'manual_protected_pixels': int(manual.sum()),
              'kept_person_protected_pixels': int(keep_protection.sum()),
              'protected_overlap_with_removed_people_pixels': int((raw_remove & protection).sum()),
              'manual_overlap_with_removed_people_pixels': int((raw_remove & manual).sum()),
              'editable_pixels': int(allowed.sum()), 'repair_grow_pixels': repair_grow_pixels,
              'keep_grow_pixels': keep_grow_pixels, 'protection_grow_pixels': protection_grow_pixels,
              'source_size': [w,h], 'processing_size': [ww,wh],
              'no_op': not bool(allowed.any()), 'gpu_inference_validated': False}
    report.update(prefill_info)
    package = {'source': source, 'plan': plan, 'work_image': work,
               'paste_mask': torch.from_numpy(allowed[y:y2,x:x2].astype(np.float32))[None],
               'overlay_protection': torch.from_numpy(protection.astype(np.float32))[None],
               'manual_protection': manual, 'keep_protection': keep_protection,
               'effective_mask': allowed, 'detections': detections, 'report': report}
    return package, work, ref, prompt


def preview_people(package):
    source = package['source'][0].numpy()
    h,w = source.shape[:2]
    ratio = min(1., 1400/max(h,w))
    size = max(1,round(w*ratio)), max(1,round(h*ratio))
    arr = np.asarray(Image.fromarray(np.rint(source*255).astype(np.uint8)).resize(size,Image.Resampling.LANCZOS)).copy()
    def scaled(mask):
        return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(size,Image.Resampling.NEAREST)) > 0
    # Selection visualization only. These colors never enter the Qwen input.
    for mask,color,strength in [(package['effective_mask'],(255,70,60),0.45),
                                 (package['keep_protection'],(45,210,90),0.35),
                                 (package['manual_protection'],(25,175,255),0.65)]:
        hit = scaled(mask)
        arr[hit] = np.rint(arr[hit]*(1-strength)+np.array(color)*strength).astype(np.uint8)
    image=Image.fromarray(arr); draw=ImageDraw.Draw(image)
    font_size=max(16, round(min(size)*0.028))
    try: font=ImageFont.truetype('DejaVuSans.ttf',font_size)
    except OSError: font=ImageFont.load_default()
    for index,box in enumerate(package['detections']['boxes'],1):
        xy=tuple(round(v*ratio) for v in box)
        is_kept = index == package['report']['keep_person']
        color=(20,230,70) if is_kept else (255,70,55)
        draw.rectangle(xy,outline=color,width=max(2,font_size//9))
        text=f'{index}: KEEP' if is_kept else f'{index}: REMOVE'
        tb=draw.textbbox((0,0),text,font=font)
        tw,th=tb[2]-tb[0]+10,tb[3]-tb[1]+8
        tx=max(0,min(xy[0],size[0]-tw)); ty=max(0,xy[1]-th)
        draw.rectangle((tx,ty,tx+tw,ty+th),fill=(0,0,0))
        draw.text((tx+5,ty+3-tb[1]),text,font=font,fill=color)
    return image


def compose_people(package, generated_crop, feather_pixels=12, transparency_policy='review draft'):
    from .qkeep_repair import compose
    return compose(package, generated_crop, feather_pixels, transparency_policy)


def _write_png(path, array, info, profile):
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.qkeep-',suffix='.png')
    os.close(fd)
    try:
        kwargs={'icc_profile':profile} if array.ndim==3 else {}
        Image.fromarray(array).save(tmp,format='PNG',pnginfo=info,compress_level=4,**kwargs)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


class WBPeopleDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',),
            'threshold': ('FLOAT',{'default':0.5,'min':0.05,'max':0.95,'step':0.05}),
            'refine_iterations': ('INT',{'default':2,'min':0,'max':5}),
            'expected_people': ('INT',{'default':0,'min':0,'max':5})}}
    RETURN_TYPES=('WB_PEOPLE', 'INT')
    RETURN_NAMES=('detected_people','person_count')
    FUNCTION='detect'
    CATEGORY=CATEGORY
    DESCRIPTION='Native SAM 3.1. Numbers 1-5 left to right. Expected 0 = automatic; more than 5 stops the run.'

    @torch.inference_mode()
    def detect(self,source,threshold=0.5,refine_iterations=2,expected_people=0):
        source=_image(source)
        if not math.isfinite(threshold) or not 0.05<=threshold<=0.95:
            raise ValueError('SAM threshold must be 0.05-0.95.')
        if type(refine_iterations) is not int or not 0<=refine_iterations<=5:
            raise ValueError('SAM refinement must be 0-5.')
        import folder_paths
        import nodes
        import comfy.model_management as mm
        from comfy_extras.nodes_sam3 import SAM3_Detect
        if not folder_paths.get_full_path('checkpoints',SAM_FILE):
            raise FileNotFoundError('Missing models/checkpoints/'+SAM_FILE+'. Prepare SAM in Workbench.')
        try:
            model,clip,_=nodes.CheckpointLoaderSimple().load_checkpoint(SAM_FILE)
            conditioning=nodes.CLIPTextEncode().encode(clip,SAM_TEXT)[0]
            result=SAM3_Detect.execute(model=model,image=source,conditioning=conditioning,
                threshold=threshold,refine_iterations=refine_iterations,individual_masks=True)
            result=result.result if hasattr(result,'result') else result
            if isinstance(result,dict):result=result['result']
            data=people_from_masks(source,result[0],expected_people)
            return data,data['count']
        finally:
            mm.unload_all_models()
            mm.soft_empty_cache()


class WBPeoplePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'detections':('WB_PEOPLE',),
            'keep_person':('INT',{'default':1,'min':1,'max':5}),
            'resolution':('INT',{'default':2048,'min':256,'max':2048,'step':32}),
            'context_pixels':('INT',{'default':128,'min':0,'max':2048,'step':16}),
            'repair_grow_pixels':('INT',{'default':24,'min':0,'max':256}),
            'keep_grow_pixels':('INT',{'default':2,'min':0,'max':256}),
            'protection_grow_pixels':('INT',{'default':0,'min':0,'max':256}),
            'instructions':('STRING',{'default':DEFAULT_INSTRUCTIONS,'multiline':True})},
            'optional':{'manual_protection':('MASK',),'extra_repair_mask':('MASK',)}}
    RETURN_TYPES=('WB_PEOPLE_EDIT','IMAGE','IMAGE','STRING')
    RETURN_NAMES=('protected_edit_package','image1_working_scene','image2_person_to_keep','edit_prompt')
    FUNCTION='prepare'
    CATEGORY=CATEGORY
    DESCRIPTION='Paint on ORIGINAL LoadImage to PROTECT those pixels. Extra repair mask, when connected, is a separate editable-area input.'
    def prepare(self,**kwargs):return prepare_people(**kwargs)


class WBPeopleExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'package':('WB_PEOPLE_EDIT',),
            'run_edit':('BOOLEAN',{'default':False}),
            'filename_prefix':('STRING',{'default':'Qwen21_Keep_One/Result'}),
            'feather_pixels':('INT',{'default':12,'min':0,'max':256}),
            'show_overlay_preview':('BOOLEAN',{'default':False}),
            'transparency_policy':(['review draft','stop on transparency'],{'default':'review draft'})},
            'optional':{'generated_crop':('IMAGE',{'lazy':True})},
            'hidden':{'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION='save'
    CATEGORY=CATEGORY
    DESCRIPTION='OFF numbers/ previews people without running Qwen. ON saves the full photo AND a separate protected RGBA overlay. Composite is shown first.'
    def check_lazy_status(self,package,run_edit=False,generated_crop=None,**kwargs):
        return ['generated_crop'] if run_edit and not package['report']['no_op'] and generated_crop is None else []

    def save(self,package,run_edit=False,filename_prefix='Qwen21_Keep_One/Result',
             feather_pixels=12,show_overlay_preview=False,generated_crop=None,prompt=None,extra_pnginfo=None,
             transparency_policy='review draft'):
        import folder_paths
        if not run_edit:
            name='qkeep_people_'+uuid.uuid4().hex+'.png'
            temp=Path(folder_paths.get_temp_directory());temp.mkdir(parents=True,exist_ok=True)
            preview_people(package).save(temp/name)
            message=(f"Detected {package['report']['people_detected']} people. Keep person {package['report']['keep_person']}. "
                     'Green=kept person; red=removal/repair; blue=painted protection. '
                     'Change keep_person as needed, then turn run_edit ON. Preview only: Qwen was not requested.')
            if package['report']['manual_overlap_with_removed_people_pixels']:
                message+=' WARNING: your painted protection overlaps a removed person. Protected fragments will remain unchanged.'
            return {'ui':{'images':[{'filename':name,'subfolder':'','type':'temp'}],'text':[message]}}
        composite,rgba,alpha,report=compose_people(package,generated_crop,feather_pixels,transparency_policy)
        out_root=Path(folder_paths.get_output_directory()).resolve()
        directory,stem,counter,subfolder,_=folder_paths.get_save_image_path(
            filename_prefix,str(out_root),composite.shape[1],composite.shape[0])
        directory=Path(directory)
        if not directory.resolve().is_relative_to(out_root):
            raise ValueError('Output prefix escapes the ComfyUI output directory.')
        directory.mkdir(parents=True,exist_ok=True)
        info=PngInfo();info.add_text('qwen_keep_one_audit',json.dumps(report))
        if os.environ.get('WB_SAVE_METADATA','0')=='1':
            if prompt is not None:info.add_text('prompt',json.dumps(prompt))
            for key,value in (extra_pnginfo or {}).items():info.add_text(key,json.dumps(value))
        profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        if report['fallback_review_required']:
            stem += '_DRAFT_FALLBACK'
        token=f'{stem}_{counter:05d}_{uuid.uuid4().hex[:8]}'
        records=[]
        for suffix,array in [('composite',composite),('overlay',rgba),('edit_mask',alpha)]:
            name=f'{token}_{suffix}.png'
            _write_png(directory/name,array,info,profile)
            records.append({'filename':name,'subfolder':subfolder,'type':'output'})
        if report['fallback_review_required'] and generated_crop is not None:
            for label, image in [('prefill', package['work_image']), ('model_rgba', generated_crop)]:
                pixels = np.rint(image[0].detach().float().cpu().clamp(0,1).numpy()*255).astype(np.uint8)
                _write_png(directory/(token+'_diagnostic_'+label+'.png'), pixels, info, profile)
        (directory/(token+'_audit.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        text='Saved full photo: '+records[0]['filename']+'\nSeparate transparent overlay: '+records[1]['filename']
        if report['fallback_review_required']:
            text += ('\nDRAFT, NOT VERIFIED REPAIR: '
                     f"{100*report['generated_low_alpha_edit_fraction']:.1f}% of the edit area "
                     'needed coarse background fallback. Review diagnostic PNGs before use.')
        if report['manual_overlap_with_removed_people_pixels']:
            text+='\nWARNING: painted protection overlaps a removed person; those original fragments remain.'
        if report['no_op']:text+='\nNo removal needed; original photo copied and overlay is fully transparent.'
        return {'ui':{'images':records[:2] if show_overlay_preview else records[:1],
                      'text':[text,json.dumps(report,indent=2)]}}


NODE_CLASS_MAPPINGS={c.__name__:c for c in (WBPeopleDetect,WBPeoplePrepare,WBPeopleExport)}
NODE_DISPLAY_NAME_MAPPINGS={
    'WBPeopleDetect':'Qwen People | SAM: find 1-5 people',
    'WBPeoplePrepare':'Qwen People | Choose keeper + painted protection',
    'WBPeopleExport':'Qwen People | Preview / Full photo + protected overlay',
}

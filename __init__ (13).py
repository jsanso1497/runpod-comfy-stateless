"""User-directed Ollama -> native MiniMax H3 workflow. No RefMod dependency."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import time
import uuid

from . import director as d

REF = 'DIRECTED_H3_REFERENCE'
REFS = 'DIRECTED_H3_REFERENCES'
JOB = 'DIRECTED_H3_JOB'
NONE = '(none)'
MODEL_FILES = ('minimax_h3_ref2va_bf16.safetensors',
               'qwen3vl_32b_minimax_h3_bf16.safetensors',
               'minimax_h3_video_vae_fp16.safetensors',
               'minimax_h3_audio_vae_fp32.safetensors')


def registered(name):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f'Required node {name} is missing. Use the rebuilt H3 image.')
    return cls


def ollama_modules():
    # Resolve the installed package by its registered class, not a guessed path.
    package = registered('EverydayOllamaH3Prompt').__module__
    return importlib.import_module(package), importlib.import_module(package + '.client')


class DirectedH3LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        native = registered('LoadImage').INPUT_TYPES()['required']['image']
        choices = [NONE] + [p for p in native[0] if p != NONE]
        return {'required': {'image':(choices, {'image_upload':True,'default':NONE})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'load'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'Optional original photograph. Leave trailing unused slots at (none).'

    @classmethod
    def VALIDATE_INPUTS(cls,image):
        return True if image in (NONE,'',None) else registered('LoadImage').VALIDATE_INPUTS(image)

    @classmethod
    def IS_CHANGED(cls,image):
        return 'empty' if image in (NONE,'',None) else registered('LoadImage').IS_CHANGED(image)

    def load(self,image):
        if image in (NONE,'',None):
            return (None,)
        cls = registered('LoadImage')
        result = getattr(cls(),cls.FUNCTION)(image=image)
        values = result['result'] if isinstance(result,dict) else result
        return (d.validate_image(values[0]),)


class DirectedH3Reference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image':('IMAGE',),
            'belongs_to': ('STRING', {'default':'', 'tooltip':'Your label, e.g. Person A, Person B, or Room. Repeat the same label for multiple views of the SAME subject.'}),
            'what_image_contains': ('STRING', {'default':'', 'multiline':True,
                'tooltip':'Describe the reference yourself. For group photos, say which person/item is the intended target.'}),
            'use_this_reference_for': ('STRING', {'default':'', 'multiline':True,
                'tooltip':'For example: primary facial identity only; body proportions and outfit; scene background only. Explicitly name priority or exclusions.'})}}
    RETURN_TYPES = (REF,)
    RETURN_NAMES = ('defined_reference',)
    FUNCTION = 'define'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'Define ownership, source contents and role yourself. No automatic face grouping or head/body assumptions.'

    def define(self,image,belongs_to,what_image_contains,use_this_reference_for):
        if image is None:
            return (None,)
        return (d.make_reference(image,belongs_to,what_image_contains,use_this_reference_for),)


class DirectedH3ReferenceMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference_1':(REF,)},
                'optional': {f'reference_{i}':(REF,) for i in range(2,10)}}
    RETURN_TYPES = (REFS,'STRING')
    RETURN_NAMES = ('references','human_reference_map')
    FUNCTION = 'assemble'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'Groups identical human belongs_to labels; does not compare faces or assign head/body priorities.'

    def assemble(self, reference_1, **references):
        refs = d.collect({'reference_1':reference_1, **references})
        return refs, d.ledger(refs)


class DirectedH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'references':(REFS,),
            'mode':(list(d.MODES), {'default':d.MODES[0]}),
            'actions':('STRING', {'default':'', 'multiline':True,
                'tooltip':'YOU direct each subject. Example: Person A walks toward Person B; they hug, then hold still. Repeated references do not create extra people.'}),
            'scene_camera_audio':('STRING', {'default':'One continuous photorealistic shot. Stationary camera. No dialogue, no music, no on-screen text.', 'multiline':True}),
            'preservation_rules':('STRING', {'default':'Follow each reference\'s stated role and priority. Preserve the referenced identities, natural proportions and requested clothing. Do not beautify faces or blend distinct people. Do not copy incidental people or source poses unless requested.', 'multiline':True}),
            'length':('INT', {'default':124,'min':124,'max':362,'step':17}),
            'width':('INT', {'default':1344,'min':256,'max':4096,'step':32}),
            'height':('INT', {'default':768,'min':256,'max':4096,'step':32}),
            'variation_seed':('INT', {'default':42,'min':0,'max':2147483647}),
            'temperature':('FLOAT', {'default':0.2,'min':0.0,'max':1.0,'step':0.05}),
            'let_ollama_look_at_images':('BOOLEAN', {'default':False,
                'tooltip':'OFF: prompt writer uses your descriptions only. ON: supplemental visual inspection, but human assignments still prevail. H3 always receives the actual images.'}),
            'save_prompt':('BOOLEAN', {'default':True}),
            'system_prompt':('STRING', {'default':d.SYSTEM,'multiline':True}),
            'reviewed_prompt':('STRING', {'default':'','multiline':True,
                'tooltip':'Paste the COMPLETE draft preview including its first H3_DIRECT_INPUTS line. Edit narrative only; keep the fixed map and HUMAN DIRECTION block.'})}}
    RETURN_TYPES = (JOB,'STRING')
    RETURN_NAMES = ('directed_job','draft_to_review_or_copy')
    FUNCTION = 'write'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'Default: draft text only, blocking H3 loaders. Render reviewed prompt reuses exactly your reviewed wording without another Ollama call.'

    def write(self, references, mode, actions, scene_camera_audio, preservation_rules,
              length, width, height, variation_seed, temperature, let_ollama_look_at_images,
              save_prompt, system_prompt, reviewed_prompt):
        if mode not in d.MODES:
            raise ValueError('Unknown H3 directing mode.')
        facts, signature, exact = d.check_request(references, actions, scene_camera_audio,
                                                  preservation_rules, length, width, height)
        if mode == 'Render reviewed prompt':
            prompt = d.read_review(reviewed_prompt, references, signature, exact)
            metadata = {'mode':mode, 'human_inputs':facts, 'ollama_called':False}
        else:
            import comfy.model_management as mm
            package, client = ollama_modules()
            cfg = client.settings()
            with package.LOCK:
                mm.throw_exception_if_processing_interrupted()
                mm.unload_all_models()
                mm.soft_empty_cache()
                prompt, metadata = d.generate(client,cfg,references,facts,exact,system_prompt,
                    variation_seed,temperature,let_ollama_look_at_images,mm.throw_exception_if_processing_interrupted)
            print('[DirectedH3] Draft ready; Ollama unloaded. ' +
                  ('H3 remains blocked for review.' if mode == 'Draft prompt only' else 'Starting native H3.'), flush=True)
        review = d.review_copy(prompt,signature)
        metadata.update({'mode':mode,'input_signature':signature,'no_refmod':True})
        if save_prompt:
            import folder_paths
            folder = Path(folder_paths.get_output_directory()) / 'prompt_assistant'
            folder.mkdir(parents=True,exist_ok=True)
            stem = 'user_directed_' + time.strftime('%Y%m%d-%H%M%S') + '_' + uuid.uuid4().hex[:8]
            (folder/(stem+'.txt')).write_text(prompt+'\n',encoding='utf-8')
            (folder/(stem+'.review.txt')).write_text(review+'\n',encoding='utf-8')
            (folder/(stem+'.json')).write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        job = {'references':references,'prompt':prompt,'length':length,'width':width,'height':height,
               'render':mode != 'Draft prompt only','input_signature':signature}
        return job, review


class DirectedH3RenderGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'directed_job':(JOB,)}}
    RETURN_TYPES = ('COMBO','COMBO','COMBO','COMBO',JOB)
    RETURN_NAMES = ('diffusion_model','text_encoder','video_vae','audio_vae','approved_job')
    FUNCTION = 'release'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'All native model loaders depend on this gate. Draft-only mode blocks them before model loading, VAE encoding or sampling.'

    def release(self, directed_job):
        if not directed_job.get('prompt','').strip():
            raise ValueError('A valid prompt is required before H3 can load.')
        if directed_job.get('render') is not True:
            from comfy_execution.graph_utils import ExecutionBlocker
            return tuple(ExecutionBlocker(None) for _ in range(5))
        return (*MODEL_FILES,directed_job)


class DirectedH3OriginalImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'approved_job':(JOB,)}}
    RETURN_TYPES = ('IMAGE',)*9 + ('STRING','INT','INT','INT')
    RETURN_NAMES = tuple(f'picture_{i}' for i in range(1,10)) + ('prompt','width','height','length')
    FUNCTION = 'unpack'
    CATEGORY = 'H3/User Directed'
    DESCRIPTION = 'Original images to native MiniMaxH3ReferenceToVideo, in their human-defined order. No RefMod, collage, common-canvas crop or face inference.'

    def unpack(self, approved_job):
        if approved_job.get('render') is not True:
            raise ValueError('The job has not been released for rendering.')
        images = [r['image'] for r in approved_job['references']['references']]
        if not 1 <= len(images) <= 9:
            raise ValueError('Use one to nine still-image references.')
        images += [None]*(9-len(images))
        return (*images,approved_job['prompt'],approved_job['width'],approved_job['height'],approved_job['length'])


NODE_CLASS_MAPPINGS = {c.__name__:c for c in (DirectedH3LoadImage,DirectedH3Reference,DirectedH3ReferenceMap,
    DirectedH3Prompt,DirectedH3RenderGate,DirectedH3OriginalImages)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'DirectedH3LoadImage':'H3 Reference Image (Optional)',
    'DirectedH3Reference':'H3 User-Defined Reference',
    'DirectedH3ReferenceMap':'H3 Human Reference Map',
    'DirectedH3Prompt':'H3 Director - Your Definitions + Ollama',
    'DirectedH3RenderGate':'H3 Review Gate - Draft Does Not Render',
    'DirectedH3OriginalImages':'H3 Original Images - No RefMod'}

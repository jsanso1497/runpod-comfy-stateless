"""ComfyUI long-form Akatz analyzer. Legacy analyzer remains import-compatible."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import numpy as np
from . import akatz_analysis as analysis, media, prompts
from .analysis_options import AnalysisOptions, DETAIL_LEVELS, LAYOUTS, DUAL_LAYOUT, VERSION
from .config import atomic_json, data_root


@contextmanager
def analysis_lock():
    # Same lock as the disk runner. Refuse rather than wait inside Comfy's one
    # worker: waiting could deadlock a chain that is waiting for its queued job.
    folder = data_root() / 'output/h3_character_swap/disk_chain_v2'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.controller.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('A low-memory chain is running. Stop it or wait before analyzing another clip.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class H3SwapAnalyzeAkatz:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('VIDEO',), 'character': ('IMAGE',),
            'target_person': ('STRING', {'default': 'the main performer', 'multiline': True}),
            'start_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 2592000., 'step': .001}),
            'frames': ('INT', {'default': 124, 'min': 22, 'max': 362, 'step': 17}),
            'wardrobe': (['Reference clothing', 'Source clothing (experimental)'],),
            'reuse_cached_draft': ('BOOLEAN', {'default': True}),
            'detail_level': (list(DETAIL_LEVELS), {'default': 'Exhaustive'}),
            'max_output_tokens': ('INT', {'default': 16384, 'min': 1024, 'max': 32768, 'step': 1024,
                'tooltip': 'Output ceiling, not a mandatory length. Bigger output can make H3 rendering heavier.'}),
            'context_tokens': (['16384', '32768', '65536'], {'default': '32768'}),
            'sampled_frames': ('INT', {'default': 16, 'min': 8, 'max': 32, 'step': 1}),
            'reference_layout': (list(LAYOUTS), {'default': LAYOUTS[0]}),
            'timeout_seconds': ('INT', {'default': 1800, 'min': 120, 'max': 3600, 'step': 60})}}
    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'IMAGE')
    RETURN_NAMES = ('akatz_selected_clip_prompt', 'akatz_chain_safe_prompt', 'analysis_report', 'source_contact_sheet')
    FUNCTION = 'analyze'
    CATEGORY = 'H3 Character Replacement'
    DESCRIPTION = ('Akatz-only long analysis. Clip prompt includes the selected timeline; chain prompt '
                   'does not repeat that timeline. This node does not run H3 or analyze audio.')

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always enter the node; its own disk cache has the model digest/settings.
        # Otherwise reuse_cached_draft=False can itself be cached by ComfyUI.
        return float('nan')

    def analyze(self, video, character, target_person, start_seconds, frames, wardrobe,
                reuse_cached_draft=True, detail_level='Exhaustive', max_output_tokens=16384,
                context_tokens='32768', sampled_frames=16, reference_layout=LAYOUTS[0], timeout_seconds=1800, body=None):
        import torch
        from PIL import Image
        import comfy.model_management as mm
        options = AnalysisOptions(max_output_tokens, int(context_tokens), sampled_frames,
                                  detail_level, DUAL_LAYOUT if body is not None else reference_layout,
                                  timeout_seconds, 2 if body is not None else 1).validate()
        target = analysis.validate_target(target_person)
        if body is not None and wardrobe != 'Reference clothing':
            raise ValueError('Face/body mode takes body and outfit from Picture 2, not source clothing')
        media.frame_count(frames)
        if wardrobe not in ('Reference clothing', 'Source clothing (experimental)'):
            raise ValueError('Invalid wardrobe mode')
        # A raw stream read would ignore the crop wrapper. Refuse it explicitly.
        if getattr(video, '_VideoFromFile__crop', None):
            raise ValueError('Use Load Video directly or export the cropped clip first; analysis must match your render.')
        path, base_start, base_duration = media.source_file(video)
        def reference_pixels(image, label):
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
                raise ValueError('Use one image for ' + label)
            if not torch.isfinite(image).all():
                raise ValueError(label + ' contains invalid pixels')
            # Match the encoder's RGB treatment; do not invent transparency masks.
            return (image[0, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        pixels = reference_pixels(character, 'face' if body is not None else 'character')
        reference = Image.fromarray(pixels)
        body_pixels = reference_pixels(body, 'body/outfit') if body is not None else None
        references = [reference] + ([Image.fromarray(body_pixels)] if body_pixels is not None else [])
        model = analysis.model_name()
        with analysis_lock():
            mm.throw_exception_if_processing_interrupted()
            digest = analysis.current_model_digest(model)
            key = analysis.cache_key(path, pixels.tobytes(), pixels.shape, base_start + start_seconds,
                                     frames, base_duration, target, wardrobe, model, digest, options,
                                     body_bytes=body_pixels.tobytes() if body_pixels is not None else None,
                                     body_shape=body_pixels.shape if body_pixels is not None else None)
            folder = data_root() / 'output/h3_character_swap' / ('prompt_drafts_face_body' if body is not None else 'prompt_drafts_long')
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / (key + '.json')
            images, times, report = media.sample_frames(path, start_seconds, frames, base_start,
                base_duration, sample_count=sampled_frames,
                check_interrupt=mm.throw_exception_if_processing_interrupted)
            evidence, stats, cache_error = None, None, None
            if reuse_cached_draft and result_path.is_file():
                try:
                    if result_path.stat().st_size > 2000000:
                        raise ValueError('Oversized cache record')
                    cached = json.loads(result_path.read_text())
                    if cached.get('version') != VERSION or cached.get('cache_key') != key:
                        raise ValueError('Cache version/key mismatch')
                    evidence = analysis.parse_analysis(json.dumps(cached['analysis']))
                    stats = cached['generation']
                    if not isinstance(stats, dict) or stats.get('requested') != options.record():
                        raise ValueError('Cache settings mismatch')
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    evidence, stats = None, None
                    cache_error = str(exc)
            report['cached_draft'] = evidence is not None
            if evidence is None:
                mm.throw_exception_if_processing_interrupted()
                mm.unload_all_models()
                mm.soft_empty_cache()
                evidence, stats = analysis.query(references + images, times, target, options,
                    mm.throw_exception_if_processing_interrupted, folder / key)
            if body is not None:
                from .face_body import build_prompts
                clip_prompt, chain_prompt = build_prompts(evidence, target)
            else:
                clip_prompt, chain_prompt = analysis.build_prompts(evidence, target, wardrobe, reference_layout)
            files = {'clip_prompt': str(folder / (key + '_akatz_clip.txt')),
                     'chain_prompt': str(folder / (key + '_akatz_chain.txt')),
                     'report': str(result_path), 'contact_sheet': str(folder / (key + '_source_samples.jpg'))}
            warnings = ['Review both drafts. More words do not guarantee stronger identity or motion control.',
                        'The clip prompt applies only to the analyzed interval. Use the chain-safe output for a long chain.',
                        'Only timestamped samples were inspected; audio and unseen intermediate motion were not analyzed.',
                        'Longer H3 input prompts can increase its memory use and rendering time.']
            if reference_layout == LAYOUTS[1]:
                warnings.append('Face/body panel roles remain guidance, not separately weighted identity controls.')
            if body is not None:
                warnings.append('Picture 1 is FACE/HEAD; Picture 2 is BODY/OUTFIT. Roles are prompt guidance, not regional locks.')
                report['reference_roles'] = {'Picture 1': 'face/head/hair', 'Picture 2': 'physique/proportions/outfit'}
            if cache_error:
                warnings.append('Ignored invalid cached draft: ' + cache_error)
            report.update(version=VERSION, model=model, model_digest=digest, generation=stats,
                          sampled_timestamps=times, review_required=True, warnings=warnings,
                          clip_prompt_words=len(clip_prompt.split()), clip_prompt_characters=len(clip_prompt),
                          chain_prompt_words=len(chain_prompt.split()), files=files, analysis=evidence)
            record = {'version': VERSION, 'cache_key': key, 'analysis': evidence,
                      'generation': stats, 'report': report, 'akatz_clip_prompt': clip_prompt,
                      'akatz_chain_prompt': chain_prompt}
            analysis.atomic_text(Path(files['clip_prompt']), clip_prompt + '\n')
            analysis.atomic_text(Path(files['chain_prompt']), chain_prompt + '\n')
            sheet = prompts.contact_sheet(images, times)
            sheet.save(files['contact_sheet'], quality=95)
            atomic_json(result_path, record)
            return (clip_prompt, chain_prompt, json.dumps(report, indent=2),
                    torch.from_numpy(np.asarray(sheet).copy().astype(np.float32) / 255.)[None])


class H3SwapAnalyzeFaceBody(H3SwapAnalyzeAkatz):
    """Two separate photos, not a flattened reference sheet."""
    @classmethod
    def INPUT_TYPES(cls):
        old = super().INPUT_TYPES()['required']
        fields = {'video': old['video'], 'face': ('IMAGE',), 'body': ('IMAGE',)}
        fields.update({k: v for k, v in old.items()
                       if k not in ('video', 'character', 'wardrobe', 'reference_layout')})
        return {'required': fields}

    DESCRIPTION = ('Experimental face/body analysis: Picture 1 supplies face/head/hair, '
                   'Picture 2 supplies physique and wardrobe. Neither supplies the source performance.')

    def analyze(self, video, face, body, target_person, start_seconds, frames,
                reuse_cached_draft=True, detail_level='Exhaustive', max_output_tokens=16384,
                context_tokens='32768', sampled_frames=16, timeout_seconds=1800):
        return super().analyze(video, face, target_person, start_seconds, frames,
            'Reference clothing', reuse_cached_draft, detail_level, max_output_tokens,
            context_tokens, sampled_frames, LAYOUTS[0], timeout_seconds, body=body)


NODE_CLASS_MAPPINGS = {'H3SwapAnalyzeAkatz': H3SwapAnalyzeAkatz,
                       'H3SwapAnalyzeFaceBody': H3SwapAnalyzeFaceBody}
NODE_DISPLAY_NAME_MAPPINGS = {'H3SwapAnalyzeAkatz': 'H3 Akatz - Long evidence-based prompt analysis',
    'H3SwapAnalyzeFaceBody': 'H3 Akatz - FACE + BODY long analysis (experimental)'}

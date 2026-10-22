"""Motion I/O and validation around native SDPose and native MiniMax H3 ControlNet."""
from __future__ import annotations
import json


class WBH3MotionVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('VIDEO',),
            'start_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 2592000.0, 'step': .01}),
            'window_seconds': ('FLOAT', {'default': 5.25, 'min': 5.167, 'max': 15.1, 'step': .01}),
            'megapixels': ('FLOAT', {'default': 1.03, 'min': .5, 'max': 2.0, 'step': .05}),
        }}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('source_frames', 'timing_report')
    FUNCTION = 'load'
    CATEGORY = 'Workbench/H3 Motion'
    DESCRIPTION = 'Decode once at 24 fps; auto source aspect; shared motion/output duration; no temporal stretching or last-frame padding.'

    def load(self, video, start_seconds, window_seconds, megapixels):
        import folder_paths
        from workbench.h3_motion_tools import resolve_video, decode_source
        path, begin, duration = resolve_video(video, folder_paths.get_input_directory())
        frames, report = decode_source(path, start_seconds, window_seconds, megapixels, begin, duration)
        print('[H3 MOTION] ' + json.dumps(report), flush=True)
        return frames, json.dumps(report, indent=2)


class WBH3MotionTrackPerson:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'bboxes': ('BOUNDING_BOX',), 'source_frames': ('IMAGE',),
            'first_person': ('INT', {'default': -1, 'min': -1, 'max': 9,
                'tooltip': '-1: largest person in first frame. 0,1,...: first-frame people ordered left to right. Geometry tracking only; inspect preview.'}),
            'max_detection_gap': ('INT', {'default': 6, 'min': 0, 'max': 12}),
        }}
    RETURN_TYPES = ('BOUNDING_BOX', 'STRING')
    RETURN_NAMES = ('selected_person_boxes', 'tracking_report')
    FUNCTION = 'track'
    CATEGORY = 'Workbench/H3 Motion'

    def track(self, bboxes, source_frames, first_person, max_detection_gap):
        from workbench.h3_motion_tools import select_subject
        n, h, w = source_frames.shape[:3]
        boxes, report = select_subject(bboxes, int(w), int(h), int(n), first_person, max_detection_gap)
        return boxes, json.dumps(report, indent=2)


class WBH3MotionReady:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source_frames': ('IMAGE',), 'pose_frames': ('IMAGE',),
            'keypoints': ('POSE_KEYPOINT',),
            'mode': (['Render video', 'Preview motion only'], {'default': 'Render video'}),
        }}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'INT', 'INT', 'INT', 'COMBO', 'COMBO', 'COMBO', 'COMBO')
    RETURN_NAMES = ('source_frames', 'control_frames', 'width', 'height', 'length',
                    'h3_model', 'h3_encoder', 'h3_vae', 'controlnet_model')
    FUNCTION = 'ready'
    CATEGORY = 'Workbench/H3 Motion'
    DESCRIPTION = 'Validate motion before loading H3. Preview-only blocks all four large-model loaders. Frees Comfy-managed preprocessing VRAM.'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def ready(self, source_frames, pose_frames, keypoints, mode):
        from workbench.h3_motion_tools import check_alignment, pose_health
        w, h, n = check_alignment(source_frames.shape, pose_frames.shape)
        report = pose_health(keypoints, n)
        print('[H3 MOTION] Pose check: ' + json.dumps(report), flush=True)
        if mode == 'Preview motion only':
            from comfy_execution.graph_utils import ExecutionBlocker
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
        if mode != 'Render video':
            raise ValueError('Choose Render video or Preview motion only.')
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        mm.unload_all_models()
        mm.soft_empty_cache()
        return (source_frames, pose_frames, w, h, n,
                'minimax_h3_ref2va_bf16.safetensors',
                'qwen3vl_32b_minimax_h3_bf16.safetensors',
                'minimax_h3_video_vae_fp16.safetensors',
                'minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors')


class WBH3MotionCheckPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model_patch': ('MODEL_PATCH',)}}
    RETURN_TYPES = ('MODEL_PATCH',)
    FUNCTION = 'check'
    CATEGORY = 'Workbench/H3 Motion'

    def check(self, model_patch):
        from workbench.h3_motion_tools import check_control_patch
        return (check_control_patch(model_patch),)


class WBH3MotionCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',), 'other': ('IMAGE',),
                             'label': (['POSE CONTROL', 'GENERATED'], {'default': 'POSE CONTROL'})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'compare'
    CATEGORY = 'Workbench/H3 Motion'
    DESCRIPTION = 'Small side-by-side timing check only; does not alter generation or master pixels.'

    def compare(self, source, other, label):
        from workbench.h3_motion_tools import paired_preview
        return (paired_preview(source, other, label),)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (
    WBH3MotionVideo, WBH3MotionTrackPerson, WBH3MotionReady,
    WBH3MotionCheckPatch, WBH3MotionCompare)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'WBH3MotionVideo': 'H3 Motion - Source Timeline',
    'WBH3MotionTrackPerson': 'H3 Motion - Select Performer',
    'WBH3MotionReady': 'H3 Motion - Preview or Render',
    'WBH3MotionCheckPatch': 'H3 Motion - Verify ControlNet 2.0',
    'WBH3MotionCompare': 'H3 Motion - Synchronized Comparison',
}

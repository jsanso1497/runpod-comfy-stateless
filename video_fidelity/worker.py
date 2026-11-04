"""One short-lived GPU process. No Qwen, prompt model, face model, or hidden fallback."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from fv_common import atomic_json, parent_death_signal, MODEL
from streaming import windows, source_mix, OverlapWriter


def inference_args(request: dict, plan: dict) -> argparse.Namespace:
    w, h = plan['output_size']
    return argparse.Namespace(
        dit_model=MODEL, model_dir=request['model_dir'],
        resolution=min(w,h), max_resolution=max(w,h), batch_size=request['batch'],
        uniform_batch_size=True, temporal_overlap=3, prepend_frames=4, seed=42,
        color_correction='lab', input_noise_scale=0.0, latent_noise_scale=0.0,
        dit_offload_device='cpu', vae_offload_device='cpu', tensor_offload_device='cpu',
        blocks_to_swap=36, swap_io_components=True, attention_mode='sdpa',
        vae_encode_tiled=True, vae_encode_tile_size=512, vae_encode_tile_overlap=128,
        vae_decode_tiled=True, vae_decode_tile_size=512, vae_decode_tile_overlap=128,
        tile_debug='false', compile_dit=False, compile_vae=False,
        cache_dit=True, cache_vae=True,
        compile_backend="inductor", compile_mode="default", compile_fullgraph=False,
        compile_dynamic=False, compile_dynamo_cache_size_limit=64, compile_dynamo_recompile_limit=64,
    )


def load_predictor(request: dict, plan: dict):
    # Environment is configured by the supervisor BEFORE importing torch.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for AI strength above zero.')
    torch.set_num_threads(4)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    root = Path(request['seedvr_root'])
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('faithful_seedvr_cli', root/'inference_cli.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = inference_args(request, plan)
    # Caches exist ONLY within this worker, reuse weights across windows, and
    # cannot survive the job. The outer ComfyUI process never owns the weights.
    cache = {}
    debug = module.Debug(enabled=True)
    def predict(raw: np.ndarray) -> np.ndarray:
        frames = torch.from_numpy(raw).to(torch.float16).div_(255.0)
        with torch.inference_mode():
            result = module._process_frames_core(frames, args, '0', debug, cache)
        # Upstream core prepends warmup frames but leaves their removal to caller.
        n = len(raw)
        if len(result) != n + args.prepend_frames:
            raise RuntimeError('Pinned SeedVR2 warmup/frame-count contract changed.')
        result = result[args.prepend_frames:args.prepend_frames+n]
        expected = (plan['output_size'][1],plan['output_size'][0],3)
        if tuple(result.shape[1:]) != expected:
            raise RuntimeError(f'Native output geometry {tuple(result.shape[1:])} != {expected}; no corrective stretch applied.')
        array = result.float().cpu().numpy().copy()
        # Release per-window intermediates, retaining only the runner/model cache.
        ctx = cache.get('ctx', {})
        keep = {'dit_device','vae_device','dit_offload_device','vae_offload_device',
                'tensor_offload_device','compute_dtype'}
        for key in list(ctx):
            if key not in keep:
                del ctx[key]
        del result, frames
        return array
    return predict


def render(job: Path, predict_override=None) -> None:
    import cv2
    cv2.setNumThreads(2)
    request = json.loads((job/'request.json').read_text())
    plan = json.loads((job/'plan.json').read_text())
    state = {'state':'running','phase':'model_loading','frames_written':0}
    atomic_json(job/'worker_status.json', state)
    strength = float(request['strength'])
    predict = predict_override or (load_predictor(request, plan) if strength else None)
    w, h = plan['output_size']
    frames_dir = job/'frames'
    frames_dir.mkdir()
    cap = cv2.VideoCapture(str(job/'prepared.mkv'))
    if not cap.isOpened():
        raise RuntimeError('Cannot open prepared source.')
    def save_frame(index, frame):
        if not np.isfinite(frame).all():
            raise RuntimeError('Non-finite frame; refusing corrupt output.')
        image = np.rint(np.clip(frame,0,1) * 65535.0).astype(np.uint16)
        name = frames_dir / f'frame_{index:08d}.png'
        if not cv2.imwrite(str(name), cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1]):
            raise OSError('PNG writing failed: ' + str(name))
    writer = OverlapWriter(save_frame)
    # Smaller disk windows at 4K, but the temporal batch is always explicit.
    size = max(33 if w*h <= 2560*1440 else 17, int(request['batch']) + 5)
    overlap = 5
    read_position = 0
    tail = None
    try:
        for window in windows(plan['count'],plan.get('cuts',[]),size,overlap):
            raw_frames = [] if not window.overlap_before else [x for x in tail]
            if window.start != read_position - len(raw_frames):
                raise RuntimeError('Source window indexing mismatch.')
            while read_position < window.stop:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'Premature source EOF at frame {read_position}.')
                if [bgr.shape[1],bgr.shape[0]] != plan['source_size']:
                    raise RuntimeError('Decoder changed the source dimensions.')
                raw_frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                read_position += 1
            raw = np.stack(raw_frames)
            tail = raw[-overlap:].copy() if window.hold_after else None
            state.update(phase='restoration',window=[window.start,window.stop],shot=window.shot)
            atomic_json(job/'worker_status.json',state)
            if predict is not None:
                predicted = predict(raw)
            else:
                predicted = np.empty((len(raw),h,w,3),dtype=np.float32)
            for i, source in enumerate(raw):
                base = source if source.shape[:2] == (h,w) else cv2.resize(source,(w,h),interpolation=cv2.INTER_LANCZOS4)
                base = base.astype(np.float32) / 255.0
                predicted[i] = source_mix(base,predicted[i],strength) if predict is not None else base
            writer.accept(window,predicted)
            state.update(frames_written=writer.next_index)
            atomic_json(job/'worker_status.json',state)
            del predicted, raw, raw_frames
        writer.finish(plan['count'])
        state.update(state='complete',phase='complete')
        atomic_json(job/'worker_status.json',state)
    finally:
        cap.release()


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    parent_death_signal(int(os.environ.get('FAITHFUL_PARENT_PID',os.getppid())))
    try:
        render(args.job)
        return 0
    except BaseException as error:
        atomic_json(args.job/'worker_status.json',dict(state='failed',error=str(error),
                    error_type=type(error).__name__,traceback=traceback.format_exc()))
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())

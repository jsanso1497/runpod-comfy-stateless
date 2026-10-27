"""Bounded checkpoint I/O. No model imports at module load.

Generated history remains float32 NumPy data, not an H.264 or 8-bit round trip.
Video masters are 16-bit RGB FFV1; delivery MP4s are separately encoded previews.
This preserves saved samples, NOT a hard constraint on H3's next generation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import shutil
import zipfile
import numpy as np
from PIL import Image
from .disk_common import ChainError, HISTORY, FPS, sha256, atomic_json, verify_video

MAX_CHECKPOINT_BYTES = 512 * 1024**2
QC_FRAMES = 6


def load_arrays(path, *, max_bytes=MAX_CHECKPOINT_BYTES):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ChainError('Checkpoint missing or too large')
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > max_bytes or len(z.infolist()) > 16:
            raise ChainError('Checkpoint decompressed size is unsafe')
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    if any(a.dtype.hasobject or not np.isfinite(a).all() for a in result.values()):
        raise ChainError('Invalid checkpoint data')
    return result


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as out:
        np.savez(out, **arrays)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def thumb(frame, *, mask=False):
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    a = np.asarray(frame, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ChainError('Frame contains non-finite values')
    h, w = a.shape[:2]
    scale = min(1.0, 160.0 / max(h, w))
    tw = max(1, round(w * scale))
    th = max(1, round(h * scale))
    if mask:
        img = Image.fromarray(np.uint8(a > .5)*255)
        return np.asarray(img.resize((tw, th), Image.Resampling.NEAREST), dtype=np.float32)/255
    img = Image.fromarray(np.uint8(np.rint(np.clip(a[..., :3], 0, 1)*255)))
    return np.asarray(img.resize((tw, th), Image.Resampling.BILINEAR), dtype=np.float32)/255


def _array(tensor):
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def write_master(images, target, stop_check=lambda: None):
    """FFmpeg receives one frame at a time; never allocate another full batch."""
    target = Path(target)
    h, w = int(images.shape[1]), int(images.shape[2])
    temp = target.with_name(target.stem+'.partial.mkv')
    command = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
               '-f','rawvideo','-pixel_format','rgb48le','-video_size',f'{w}x{h}',
               '-framerate','24','-i','pipe:0','-an','-c:v','ffv1','-level','3',
               '-g','1','-pix_fmt','gbrp16le','-threads','2',str(temp)]
    with target.with_suffix('.log').open('w') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index, frame in enumerate(images):
                stop_check()
                if index % 16 == 0 and shutil.disk_usage(target.parent).free < 128 * 1024**2:
                    raise ChainError("Insufficient disk space while saving chunk")
                a = _array(frame)[..., :3]
                if not np.isfinite(a).all():
                    raise ChainError('Non-finite generated pixels')
                payload = np.rint(np.clip(a, 0, 1)*65535).astype('<u2').tobytes()
                proc.stdin.write(payload)
            proc.stdin.close()
            if proc.wait(timeout=300):
                raise ChainError('FFV1 encoding failed; inspect '+str(target.with_suffix('.log')))
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            temp.unlink(missing_ok=True)
            raise
    verify_video(temp, len(images), (w,h))
    os.replace(temp,target)


def save_chunk(images, source, raw_masks, folder, new_frames, history_frames,
               selection, selection_index, stop_check=lambda: None):
    folder=Path(folder)
    if (folder/'receipt.json').exists():
        raise ChainError('An attempt receipt already exists; refuse overwrite')
    folder.mkdir(parents=True, exist_ok=True)
    if len(images) != new_frames or len(source) < new_frames+history_frames:
        raise ChainError('Generated/source timeline mismatch before checkpoint')
    if images.shape[1:] != source.shape[1:]:
        raise ChainError('Generated/source canvas mismatch')
    if raw_masks.shape[1:] != images.shape[1:3] or len(raw_masks) < new_frames+history_frames:
        raise ChainError('Mask does not cover the source section')
    real_source=source[history_frames:history_frames+new_frames]
    real_masks=raw_masks[history_frames:history_frames+new_frames]
    tail_count=min(HISTORY,new_frames)
    history = _array(images[-tail_count:]).copy()
    # Carry raw masks, never expanded masks, to avoid repeated growth across jobs.
    mask_tail = _array(real_masks[-tail_count:]).copy()
    save_arrays(folder/'history.npz', frames=history, masks=mask_tail)
    q=min(QC_FRAMES,new_frames)
    qc={
        'head':np.stack([thumb(f) for f in images[:q]]),
        'tail':np.stack([thumb(f) for f in images[-q:]]),
        'source_head':np.stack([thumb(f) for f in real_source[:q]]),
        'source_tail':np.stack([thumb(f) for f in real_source[-q:]]),
        'mask_head':np.stack([thumb(f,mask=True) for f in real_masks[:q]]),
        'mask_tail':np.stack([thumb(f,mask=True) for f in real_masks[-q:]])}
    save_arrays(folder/'qc.npz',**qc)
    # A viewable full-size first output and its source mask help manual review.
    Image.fromarray(np.uint8(np.rint(np.clip(_array(images[0]),0,1)*255))).save(folder/'first_frame.png')
    im=_array(real_source[0]); m=_array(real_masks[0])>.5
    overlay=np.where(m[...,None],im*.65+np.array([.1,.9,.4],dtype=np.float32)*.35,im)
    Image.fromarray(np.uint8(np.rint(np.clip(overlay,0,1)*255))).save(folder/'source_mask.png')
    write_master(images,folder/'master.mkv',stop_check)
    receipt={'format':'h3-disk-chunk-v2','frames':new_frames,'fps':FPS,
             'width':int(images.shape[2]),'height':int(images.shape[1]),
             'history_frames':tail_count,'history_precision':'float32, no image-codec quantization',
             'selection':selection,'selection_index':selection_index,
             'files':{n:sha256(folder/n) for n in ['master.mkv','history.npz','qc.npz']}}
    atomic_json(folder/'receipt.json',receipt)
    return receipt


def verify_receipt(folder, expected_frames=None):
    folder=Path(folder)
    data=json.loads((folder/'receipt.json').read_text())
    if data.get('format')!='h3-disk-chunk-v2' or data.get('fps')!=24:
        raise ChainError('Unknown disk chunk contract')
    if expected_frames is not None and data['frames']!=expected_frames:
        raise ChainError('Saved section frame count is wrong')
    if set(data.get('files',{}))!={'master.mkv','history.npz','qc.npz'}:
        raise ChainError('Incomplete chunk receipt')
    for name, digest in data['files'].items():
        p=folder/name
        if p.is_symlink() or sha256(p)!=digest:
            raise ChainError('Accepted chunk changed: '+str(p))
    return data

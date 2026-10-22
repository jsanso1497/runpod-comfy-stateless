"""Bounded source-time video decoding and motion checks. No model inference here."""
from __future__ import annotations
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile

FPS = 24
MIN_FRAMES = 124
MAX_FRAMES = 362


def number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a number.')
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}.')
    return value


def aligned_length(available_frames):
    if type(available_frames) is not int or available_frames < MIN_FRAMES:
        raise ValueError('Select at least 124 source frames at 24 fps (5.167 seconds). No frames are invented.')
    n = min(available_frames, MAX_FRAMES)
    return 5 + ((n - 5) // 17) * 17


def canvas(width, height, megapixels=1.03):
    if min(width, height) < 16 or not 0.2 <= width / height <= 5:
        raise ValueError('Source aspect ratio must be between 1:5 and 5:1.')
    mp = number(megapixels, 'Megapixels', 0.5, 2.0)
    scale = min(math.sqrt(mp * 1_000_000 / (width * height)), 2048 / max(width, height))
    w = max(32, int(round(width * scale / 32)) * 32)
    h = max(32, int(round(height * scale / 32)) * 32)
    return w, h


def run(command, timeout=180):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError('This H3 image must include ffmpeg and ffprobe.') from exc
    if p.returncode:
        raise ValueError('Local video decode failed: ' + p.stderr[-1500:])
    return p.stdout


def resolve_video(video, input_root):
    getter = getattr(video, 'get_stream_source', None)
    if not callable(getter):
        raise ValueError('Connect the native LoadVideo node directly.')
    raw = getter()
    if not isinstance(raw, (str, Path)) or '://' in str(raw):
        raise ValueError('Select a local uploaded video, not a URL or an in-memory video.')
    root = Path(input_root).resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Source must be an existing file inside ComfyUI input storage.')
    trim = getattr(video, 'get_active_trim_window', None)
    start, duration = trim() if callable(trim) else (0.0, 0.0)
    return path.resolve(), number(start, 'Input trim start', 0, 2592000), number(duration, 'Input trim duration', 0, 2592000)


def source_geometry(stream):
    w, h = int(stream['width']), int(stream['height'])
    if min(w, h) < 16 or w * h > 100_000_000:
        raise ValueError('Unsupported source-video dimensions.')
    try:
        sar = float(Fraction(str(stream.get('sample_aspect_ratio', '1:1')).replace(':', '/')))
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    if not math.isfinite(sar) or sar <= 0:
        sar = 1.0
    w = max(1, round(w * sar))
    rotation = stream.get('tags', {}).get('rotate', 0)
    for side in stream.get('side_data_list', []):
        if 'rotation' in side:
            rotation = side['rotation']
    if round(float(rotation) / 90) % 2:
        w, h = h, w
    if stream.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('Convert HDR source footage to SDR before using this workflow.')
    return w, h


def decode_source(path, start_seconds, window_seconds, megapixels=1.03,
                  input_start=0.0, input_duration=0.0):
    import numpy as np
    import torch
    path = Path(path)
    start = number(start_seconds, 'Start seconds', 0, 2592000)
    duration = number(window_seconds, 'Source window', 124 / 24, 15.1)
    input_start = number(input_start, 'Input trim start', 0, 2592000)
    input_duration = number(input_duration, 'Input trim duration', 0, 2592000)
    metadata = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], 30))
    streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if not streams:
        raise ValueError('The uploaded file has no video stream.')
    stream = streams[0]
    width, height = source_geometry(stream)
    ow, oh = canvas(width, height, megapixels)
    candidates = [metadata.get('format', {}).get('duration'), stream.get('duration')]
    total = next((float(v) for v in candidates if v not in (None, 'N/A')), 0.0)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Video duration is unavailable. Remux the source to a normal MP4.')
    available = min(total - input_start, input_duration) if input_duration > 0 else total - input_start
    duration = min(duration, available - start)
    requested = aligned_length(int(math.floor(duration * FPS + 1e-6)))
    # Decode a bounded source-time window exactly once. fps=24 changes sampling rate,
    # not playback speed. No looping, last-frame padding, or temporal interpolation.
    # Independent rounded axes can differ slightly from the original aspect; pad
    # instead of stretching people or cutting off hands/feet.
    fit = min(ow / width, oh / height)
    cw = min(ow, max(2, int(round(width * fit / 2)) * 2))
    ch = min(oh, max(2, int(round(height * fit / 2)) * 2))
    filters = (f'fps=24,scale={cw}:{ch}:flags=lanczos,setsar=1,'
               f'pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2')
    with tempfile.TemporaryDirectory(prefix='h3-motion-') as temp:
        raw = Path(temp) / 'frames.rgb'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '2',
                   '-protocol_whitelist', 'file,pipe', '-ss', f'{input_start + start:.8f}',
                   '-i', str(path), '-t', f'{duration:.8f}', '-map', f"0:{stream['index']}",
                   '-an', '-sn', '-dn', '-vf', filters, '-frames:v', str(requested),
                   '-threads', '2', '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(raw)]
        run(command)
        frame_bytes = ow * oh * 3
        size = raw.stat().st_size
        if size <= 0 or size % frame_bytes or size > requested * frame_bytes:
            raise ValueError('Source decode returned invalid or unexpectedly many frames.')
        count = aligned_length(size // frame_bytes)
        array = np.fromfile(raw, dtype=np.uint8, count=count * frame_bytes).reshape(count, oh, ow, 3)
        images = torch.from_numpy(array).float().div_(255)
    report = {'source_file': path.name, 'start_seconds': input_start + start,
              'requested_window_seconds': window_seconds, 'source_display_size': [width, height],
              'canvas': [ow, oh], 'frames': count, 'fps': FPS, 'output_seconds': count / FPS,
              'source_motion_control_and_output_share_one_timeline': True,
              'padding_policy': 'Small spatial letterboxing only; never temporal padding.',
              'audio': 'Not decoded or exported.'}
    return images, report


def _box(row, width, height):
    try:
        x, y, w, h = [float(row[k]) for k in ('x', 'y', 'width', 'height')]
    except (TypeError, KeyError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        return None
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x+w), min(float(height), y+h)
    if x2 <= x1 or y2 <= y1:
        return None
    return {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1,
            'label': 'person', 'score': float(row.get('score', 1.0))}


def _iou(a, b):
    x = max(0., min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x']))
    y = max(0., min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    intersection = x*y
    return intersection / max(1., a['width']*a['height'] + b['width']*b['height'] - intersection)


def select_subject(detections, width, height, count, first_person=-1, max_gap=6):
    """Geometric association, not face/identity recognition. Fail instead of changing subjects silently."""
    if type(first_person) is not int or first_person < -1 or type(max_gap) is not int or not 0 <= max_gap <= 12:
        raise ValueError('Invalid subject selection or maximum detection gap.')
    if not isinstance(detections, list) or len(detections) != count:
        raise ValueError('Person detections must have exactly one entry for each source frame.')
    selected, previous, gaps, held, multi = [], None, 0, 0, 0
    for index, frame in enumerate(detections):
        if not isinstance(frame, list):
            raise ValueError('Use RT-DETR person detections, not arbitrary boxes.')
        boxes = [b for row in frame if row.get('label', 'person') == 'person' and (b := _box(row, width, height)) is not None]
        multi += int(len(boxes) > 1)
        if previous is None:
            if not boxes:
                raise ValueError('No performer was detected in the first frame. Change the source start or crop the clip.')
            if first_person == -1:
                chosen = max(boxes, key=lambda b: b['width']*b['height'])
            else:
                boxes.sort(key=lambda b: b['x']+b['width']/2)
                if first_person >= len(boxes):
                    raise ValueError('Selected person index is absent in the first frame.')
                chosen = boxes[first_person]
        else:
            def score(b):
                ax, ay = previous['x']+previous['width']/2, previous['y']+previous['height']/2
                bx, by = b['x']+b['width']/2, b['y']+b['height']/2
                distance = math.hypot((bx-ax)/width, (by-ay)/height)
                overlap = _iou(previous, b)
                return (2*overlap-distance) if overlap > .01 or distance < .15 else -100.
            chosen = max(boxes, key=score) if boxes else None
            if chosen is None or score(chosen) <= -100:
                gaps += 1
                if gaps > max_gap:
                    raise ValueError(f'Performer tracking lost near source frame {index}. Shorten/crop the shot; do not render a broken pose map.')
                chosen = dict(previous)
                held += 1
            else:
                gaps = 0
        previous = dict(chosen)
        # Small crop margin helps the pose model keep extremities. Image coordinates
        # remain the original canvas coordinates; no temporal smoothing is applied.
        pad_x, pad_y = chosen['width']*.08, chosen['height']*.08
        expanded = _box({**chosen, 'x':chosen['x']-pad_x, 'y':chosen['y']-pad_y,
                         'width':chosen['width']+2*pad_x, 'height':chosen['height']+2*pad_y}, width, height)
        selected.append([expanded])
    return selected, {'frames': count, 'multi_person_frames': multi, 'held_crop_frames': held,
                      'selection': 'largest first-frame person' if first_person == -1 else f'first-frame person {first_person}, ordered left to right',
                      'warning': 'Geometric tracking can fail at crossings/occlusions. Inspect the pose preview.'}


def check_alignment(source_shape, pose_shape):
    if len(source_shape) != 4 or len(pose_shape) != 4 or tuple(source_shape[:3]) != tuple(pose_shape[:3]):
        raise ValueError('Source and pose must have identical frame count, height and width.')
    n, h, w = map(int, source_shape[:3])
    if n != aligned_length(n) or w % 32 or h % 32:
        raise ValueError('Motion clip is not on the native H3 temporal/spatial grid.')
    return w, h, n


def pose_health(keypoints, count):
    if not isinstance(keypoints, list) or len(keypoints) != count:
        raise ValueError('Pose extraction did not return one record per source frame.')
    weak = []
    for i, frame in enumerate(keypoints):
        people = frame.get('people', [])
        if len(people) != 1:
            raise ValueError('This character-swap graph requires one selected performer per pose frame.')
        flat = people[0].get('pose_keypoints_2d', [])
        if len(flat) != 54:
            raise ValueError('Unexpected SDPose body-keypoint layout.')
        visible = sum(math.isfinite(float(flat[j])) and float(flat[j]) >= .3 for j in range(2, 54, 3))
        if visible < 4:
            weak.append(i)
    if len(weak) > max(1, int(.1 * count)):
        raise ValueError(f'Pose is weak in {len(weak)}/{count} frames. Inspect/crop the source before running H3.')
    return {'frames': count, 'weak_pose_frames': weak,
            'face_landmarks': 'Not drawn by default, to avoid copying the original facial shape.'}


def check_control_patch(model_patch):
    layers = tuple(getattr(getattr(model_patch, 'model', None), 'injection_layers', ()))
    if layers != tuple(range(0, 50, 5)):
        raise ValueError('Expected H3 Fun ControlNet Union 2.0 with ten injection layers. Do not substitute the five-layer v1 patch.')
    return model_patch


def paired_preview(source, other, label='POSE CONTROL'):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    if tuple(source.shape) != tuple(other.shape):
        raise ValueError('Comparison requires identical dimensions and timing.')
    h, w = source.shape[1:3]
    oh = min(360, h)
    ow = max(2, round(w * oh / h / 2)*2)
    # Build in chunks, not one giant full-resolution concatenation.
    out = torch.empty((source.shape[0], oh+24, ow*2, 3), dtype=torch.float32, device='cpu')
    header = Image.new('RGB', (ow*2, 24), (25, 25, 25))
    draw = ImageDraw.Draw(header)
    draw.text((8, 6), 'SOURCE', fill=(255, 255, 255))
    draw.text((ow+8, 6), label, fill=(255, 255, 255))
    stripe = torch.from_numpy(np.array(header).copy()).float()/255
    for i in range(0, len(source), 4):
        a = F.interpolate(source[i:i+4,...,:3].movedim(-1,1).cpu(), (oh, ow), mode='bilinear', align_corners=False).movedim(1,-1)
        b = F.interpolate(other[i:i+4,...,:3].movedim(-1,1).cpu(), (oh, ow), mode='bilinear', align_corners=False).movedim(1,-1)
        out[i:i+4,:24] = stripe
        out[i:i+4,24:,:ow] = a
        out[i:i+4,24:,ow:] = b
    return out

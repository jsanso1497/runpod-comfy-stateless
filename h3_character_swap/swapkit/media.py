"""Bounded, local-only media reads. Never load an entire long source into tensors."""
from __future__ import annotations
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np

FPS = 24
DEFAULT_FRAMES = 124


def checked_number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(value)


def frame_count(value):
    if type(value) is not int or not 22 <= value <= 362 or value % 17 != 5:
        raise ValueError("Use a valid 17n+5 frame count from 22 to 362; default is 124")
    return value


def run(args, timeout=180):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if p.returncode:
        raise ValueError("Local media processing failed: " + p.stderr.decode("utf-8", errors="replace")[-1500:])
    return p.stdout


def source_file(video):
    if not callable(getattr(video, "get_stream_source", None)):
        raise ValueError("Connect native Load Video directly, using a locally uploaded file")
    source = video.get_stream_source()
    if not isinstance(source, (str, Path)) or "://" in str(source):
        raise ValueError("Only file-backed local video is supported")
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError("The selected video is missing")
    trim = getattr(video, "get_active_trim_window", None)
    start, duration = trim() if callable(trim) else (0.0, 0.0)
    return path, checked_number(start, "Input trim start", 0, 2_592_000), checked_number(duration, "Input trim duration", 0, 2_592_000)


def probe(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=30))
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not streams:
        raise ValueError("No usable video stream")
    stream = streams[0]
    if stream.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ValueError("Convert HDR source to SDR before this workflow; no silent HDR-to-SDR conversion is performed")
    durations = []
    for x in (stream.get("duration"), info.get("format", {}).get("duration")):
        try:
            f = float(x)
            if math.isfinite(f) and f > 0:
                durations.append(f)
        except (TypeError, ValueError):
            pass
    if not durations:
        raise ValueError("Source has no reliable duration; remux it to MP4")
    info["duration"] = min(durations)
    info["video"] = stream
    return info


def displayed_size(stream):
    w, h = int(stream["width"]), int(stream["height"])
    try:
        sar = float(Fraction(stream.get("sample_aspect_ratio", "1:1").replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    if not math.isfinite(sar) or sar <= 0:
        sar = 1.0
    w = max(1, round(w * sar))
    rotation = float(stream.get("tags", {}).get("rotate", 0) or 0)
    for row in stream.get("side_data_list", []):
        if "rotation" in row:
            rotation = float(row["rotation"])
    if round(rotation / 90) % 2:
        w, h = h, w
    if min(w, h) < 16 or max(w / h, h / w) > 5 or w * h > 100_000_000:
        raise ValueError("Unsupported video geometry")
    return w, h


def canvas(w, h, short_edge=768):
    if short_edge not in (384, 576, 768):
        raise ValueError("Choose short edge 384, 576, or 768")
    # Preserve content with tiny alignment padding; no crop or geometry stretch.
    scale = min(short_edge / min(w, h), math.sqrt(short_edge * short_edge * 1.75 / (w * h)))
    return max(32, int(w * scale) // 32 * 32), max(32, int(h * scale) // 32 * 32)


def window(path, start_seconds, frames, input_start=0.0, input_duration=0.0):
    count = frame_count(frames)
    start = checked_number(start_seconds, "Start seconds", 0, 2_592_000)
    info = probe(path)
    available = min(info["duration"] - input_start, input_duration) if input_duration else info["duration"] - input_start
    duration = count / FPS
    # Half a source frame tolerance for media-container rounding; decoded count is checked later.
    if available - start + 0.02 < duration:
        raise ValueError(f"The selected window needs {duration:.6f} seconds; select an earlier start or a shorter valid frame count")
    return info, input_start + start, duration


def decode_clip(path, start_seconds=0.0, frames=124, short_edge=768, input_start=0.0, input_duration=0.0, check_interrupt=lambda: None):
    info, start, duration = window(path, start_seconds, frames, input_start, input_duration)
    w, h = canvas(*displayed_size(info["video"]), short_edge)
    filt = f"fps=24:start_time=0,scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    with tempfile.TemporaryDirectory(prefix="h3swap-clip-") as d:
        raw = Path(d) / "frames.rgb"
        check_interrupt()
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2", "-protocol_whitelist", "file,pipe", "-ss", f"{start:.9f}", "-i", str(path), "-map", f"0:{info['video']['index']}", "-an", "-sn", "-dn", "-vf", filt, "-frames:v", str(frames), "-threads", "2", "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw)])
        check_interrupt()
        if raw.stat().st_size != frames * w * h * 3:
            raise ValueError("Source decode did not produce the exact requested frame count; do not pad or silently shorten the guide")
        images = np.fromfile(raw, dtype=np.uint8).reshape(frames, h, w, 3).astype(np.float32) / 255.0
    audios = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    n = round(duration * 48000)
    waveform = np.zeros((2, n), dtype=np.float32)
    if audios:
        out = run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-protocol_whitelist", "file,pipe", "-ss", f"{start:.9f}", "-i", str(path), "-map", f"0:{audios[0]['index']}", "-vn", "-t", f"{duration:.9f}", "-ac", "2", "-ar", "48000", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
        values = np.frombuffer(out, dtype="<f4")
        if values.size % 2:
            raise ValueError("Malformed audio decode")
        values = values.reshape(-1, 2).T
        m = min(n, values.shape[1])
        waveform[:, :m] = values[:, :m]
    report = {"start_seconds": start, "frames": frames, "fps": 24, "seconds": duration, "width": w, "height": h, "source_audio": bool(audios), "source_unchanged": True}
    return images, waveform, report


def sample_frames(path, start_seconds=0.0, frames=124, input_start=0.0, input_duration=0.0):
    """Decode only the selected window and retain eight small JPEGs for analysis."""
    from PIL import Image
    info, start, duration = window(path, start_seconds, frames, input_start, input_duration)
    positions = sorted(set(round(i * (frames - 1) / 7) for i in range(8)))
    select = "+".join(f"eq(n\\,{n})" for n in positions)
    vf = f"fps=24:start_time=0,select='{select}',scale=576:576:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1"
    with tempfile.TemporaryDirectory(prefix="h3swap-analysis-") as d:
        run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-protocol_whitelist", "file,pipe", "-ss", f"{start:.9f}", "-t", f"{duration:.9f}", "-i", str(path), "-map", f"0:{info['video']['index']}", "-an", "-vf", vf, "-fps_mode", "vfr", "-frames:v", str(len(positions)), "-threads", "2", "-q:v", "3", str(Path(d) / "%02d.jpg")])
        paths = sorted(Path(d).glob("*.jpg"))
        if len(paths) != len(positions):
            raise ValueError("Could not read all selected analysis frames")
        images = [Image.open(p).convert("RGB").copy() for p in paths]
    return images, [n / FPS for n in positions], {"source_start": start, "duration": duration, "sampled_frames": len(images), "audio_analyzed": False}

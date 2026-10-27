"""Media and private ComfyUI client for the disk-backed H3 chain.
Adapted from the previously supplied 2026-10-26 runner. No model imports.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
FPS = 24
HISTORY = 18
VALID_WINDOWS = (56, 73, 90, 107, 124)

class ChainError(RuntimeError):
    pass


class Stopped(ChainError):
    pass


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def positive_number(value, name: str, maximum=3600.0) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= maximum:
        raise ChainError(f"{name} must be greater than zero and at most {maximum}.")
    return number


def inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ChainError(f"Path is outside the expected directory: {path}")
    return path


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def media_info(path: Path, *, count=False):
    command = ["ffprobe", "-v", "error", "-threads", "2"]
    if count:
        command += ["-count_frames"]
    command += ["-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise ChainError("ffprobe failed: " + result.stderr[-1800:])
    info = json.loads(result.stdout)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
               and not s.get("disposition", {}).get("attached_pic")]
    if not streams:
        raise ChainError("No video stream was found.")
    info["video"] = streams[0]
    info["audio"] = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    return info


def media_duration(info) -> float:
    values = []
    for raw in (info["video"].get("duration"), info.get("format", {}).get("duration")):
        try:
            value = float(raw)
            if math.isfinite(value) and value > 0:
                values.append(value)
        except (ValueError, TypeError):
            pass
    if not values:
        raise ChainError("The input has no reliable duration. Remux it to MP4 first.")
    return min(values)


def display_size(stream):
    from fractions import Fraction
    width, height = int(stream["width"]), int(stream["height"])
    try:
        sar = float(Fraction(stream.get("sample_aspect_ratio", "1:1").replace(":", "/")))
        if not math.isfinite(sar) or sar <= 0:
            sar = 1.0
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    width = width * sar
    rotation = float(stream.get("tags", {}).get("rotate", 0) or 0)
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            rotation = float(item["rotation"])
    if round(rotation / 90) % 2:
        width, height = height, width
    if min(width, height) < 16 or max(width / height, height / width) > 6:
        raise ChainError("Unsupported source geometry.")
    return width, height


def canvas_size(width, height, resolution):
    """Use Relay's 32-pixel/area policy, not a different fixed aspect ratio.

    Nearest log-distance grid search follows the pinned Relay source convention.
    See SOURCES.md for the upstream implementation and qualification boundary.
    """
    budget = {"480": 500_000, "768": 1_000_000}[str(resolution)]
    if width * height <= budget and width % 32 == 0 and height % 32 == 0:
        return int(width), int(height)
    ratio = width / height
    ideal_w, ideal_h = math.sqrt(budget * ratio), math.sqrt(budget / ratio)
    options = []
    for w in range(32, min(16384, budget // 32) + 1, 32):
        cap_h = min(16384, budget // w // 32 * 32)
        for h in {max(32, min(cap_h, int(ideal_h // 32) * 32)),
                  max(32, min(cap_h, math.ceil(ideal_h / 32) * 32))}:
            options.append((math.log(w / ideal_w) ** 2 + math.log(h / ideal_h) ** 2,
                            -(w * h), w, h))
    _, _, w, h = min(options)
    return w, h


def run_media(command, log_path: Path, stop_file: Path | None = None, timeout=7200):
    """Stream media through FFmpeg; never capture decoded frames in Python."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if stop_file and stop_file.exists():
                    raise Stopped("Stop requested during disk media processing.")
                if time.monotonic() - started > timeout:
                    raise ChainError("FFmpeg timed out. See " + str(log_path))
                if shutil.disk_usage(log_path.parent).free < 128 * 1024 ** 2:
                    raise ChainError("Less than 128 MiB of disk remains. Media processing stopped.")
                time.sleep(0.2)
            if proc.returncode:
                raise ChainError("FFmpeg failed: " + log_path.read_text(errors="replace")[-2500:])
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise


def verify_video(path, expected_frames=None, expected_size=None):
    from fractions import Fraction
    info = media_info(Path(path), count=True)
    video = info["video"]
    count = int(video.get("nb_read_frames", 0))
    rate = Fraction(video.get("avg_frame_rate", "0/1"))
    if count <= 0 or rate != FPS:
        raise ChainError(f"Invalid output: {count} frames at {rate} fps: {path}")
    if expected_frames is not None and count != expected_frames:
        raise ChainError(f"Expected {expected_frames} frames, got {count}: {path}")
    size = (int(video["width"]), int(video["height"]))
    if expected_size and size != tuple(expected_size):
        raise ChainError(f"Canvas changed: {size} != {expected_size}")
    return {"frames": count, "fps": FPS, "width": size[0], "height": size[1],
            "seconds": count / FPS, "audio": bool(info["audio"])}


def normalize_source(config, destination: Path, logs: Path, stop_file=None):
    source = Path(config["source"])
    info = media_info(source)
    if info["video"].get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ChainError("HDR source is not silently converted. Supply an SDR copy for this recipe.")
    remaining = media_duration(info) - config["start_seconds"]
    requested_frames = math.floor(min(config["duration_seconds"], remaining) * FPS + 1e-7)
    if requested_frames < 22:
        raise ChainError("The selected source section needs at least 22 frames at 24 fps.")
    width, height = canvas_size(*display_size(info["video"]), config["resolution"])
    temporary = destination.with_suffix(".partial.mkv")
    seconds = requested_frames / FPS
    vf = ("setpts=PTS-STARTPTS,fps=fps=24:round=near:eof_action=round,"
          f"scale=w='ceil(max({width},{height}*dar)-1e-6)':"
          f"h='ceil(max({height},{width}/dar)-1e-6)':flags=lanczos,"
          f"setsar=1,crop={width}:{height},trim=end_frame={requested_frames},setpts=N/(24*TB)")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
           "-ss", f"{config['start_seconds']:.9f}", "-i", str(source),
           "-map", f"0:{info['video']['index']}", "-map", "0:a:0?", "-vf", vf,
           "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-threads", "2",
           "-af", f"aresample=48000:async=1:first_pts=0,apad,atrim=end_sample={requested_frames * 2000}",
           "-c:a", "pcm_f32le", "-ar", "48000", "-ac", "2", "-t", f"{seconds:.9f}",
           "-map_metadata", "-1", "-map_chapters", "-1", str(temporary)]
    run_media(cmd, logs / "normalize.log", stop_file)
    result = verify_video(temporary, expected_size=(width, height))
    if result["frames"] > requested_frames:
        raise ChainError("Source normalization exceeded the requested duration.")
    if result["frames"] < 22:
        raise ChainError("Source normalization produced fewer than 22 frames.")
    os.replace(temporary, destination)
    return result


def plan_segments(total_frames: int, window_frames: int = 124):
    """First window delivers W; subsequent windows share one boundary frame.

    A continuation job stages 18 source-context frames plus W-1 new frames.
    Relay's explicit 'segment' policy consumes the matching generated history.
    """
    if type(total_frames) is not int or total_frames < 22:
        raise ChainError("Invalid source frame count.")
    if window_frames not in VALID_WINDOWS:
        raise ChainError("Choose 56, 73, 90, 107, or 124 frames per generation window.")
    rows, covered = [], 0
    while covered < total_frames:
        history = HISTORY if rows else 0
        delivered = min(total_frames - covered, window_frames - int(bool(history)))
        rows.append({"index": len(rows), "start_frame": covered, "new_frames": delivered,
                     "history_frames": history, "input_start": covered - history,
                     "input_frames": delivered + history})
        covered += delivered
    return rows


def extract_segment(source, target, segment, logs, stop_file=None):
    start, length = segment["input_start"], segment["input_frames"]
    # Matroska commonly stores millisecond PTS. Seeking directly to N/24 can
    # drop a boundary frame when its timestamp rounds down. Seek to an exact
    # whole second on our normalized 24 fps timeline, then trim by frame index.
    seek_seconds, offset = divmod(start, FPS)
    temporary = target.with_suffix(".partial.mkv")
    run_media(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
               "-ss", str(seek_seconds), "-i", str(source),
               "-map", "0:v:0", "-map", "0:a:0?",
               "-vf", f"trim=start_frame={offset}:end_frame={offset+length},setpts=N/(24*TB)",
               "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-threads", "2",
               "-af", f"atrim=start_sample={offset*2000}:end_sample={(offset+length)*2000},asetpts=PTS-STARTPTS,apad,atrim=end_sample={length * 2000}",
               "-c:a", "pcm_f32le", "-t", f"{length / FPS:.12f}",
               "-map_metadata", "-1", str(temporary)], logs / f"extract_{segment['index']:04d}.log", stop_file)
    verify_video(temporary, expected_frames=length)
    os.replace(temporary, target)


def validate_api_graph(graph):
    if not isinstance(graph, dict) or not graph:
        raise ChainError("Expected a ComfyUI API-format workflow.")
    for name, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise ChainError("Invalid API node: " + str(name))
        for value in node.get("inputs", {}).values():
            if isinstance(value, list):
                if len(value) != 2 or str(value[0]) not in graph or type(value[1]) is not int or value[1] < 0:
                    raise ChainError("Broken API connection at " + str(name))
    return True


class ComfyClient:
    def __init__(self, url="http://127.0.0.1:8189", poll_seconds=2.0):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1") or parts.username or parts.password:
            raise ChainError("Use this Pod's private loopback ComfyUI URL, not a public endpoint.")
        self.url, self.poll_seconds = url.rstrip("/"), poll_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.client_id = "h3-disk-chain-" + uuid.uuid4().hex
        self.current = None
        self.stop_file = None

    def request(self, path, data=None, timeout=90):
        payload = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise ChainError(f"ComfyUI {path}: HTTP {exc.code}: " + exc.read().decode(errors="replace")[-3000:]) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ChainError("Cannot reach the Pod's ComfyUI backend: " + str(exc)) from exc

    def queue(self, graph):
        result = self.request("/prompt", {"prompt": graph, "client_id": self.client_id})
        if result.get("node_errors") or not result.get("prompt_id"):
            raise ChainError("ComfyUI rejected the short job: " + json.dumps(result))
        self.current = result["prompt_id"]
        return self.current

    def wait(self, prompt_id, timeout=12 * 3600):
        start = time.monotonic()
        while True:
            if self.stop_file and self.stop_file.exists():
                self.cancel_owned()
                raise Stopped("Stop requested. Accepted sections are retained.")
            history = self.request("/history/" + urllib.parse.quote(prompt_id, safe=""))
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if not status.get("completed") or status.get("status_str") != "success":
                    raise ChainError("The chunk failed or was interrupted. " + json.dumps(status)[-4000:])
                self.current = None
                return item
            if time.monotonic() - start > timeout:
                raise ChainError("Timed out waiting for the chunk. It was not silently retried.")
            time.sleep(self.poll_seconds)

    def wait_idle(self, timeout=3600):
        start = time.monotonic()
        while True:
            if self.stop_file and self.stop_file.exists():
                raise Stopped("Stop requested while waiting for ComfyUI.")
            queue = self.request("/queue")
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                return
            if time.monotonic() - start > timeout:
                raise ChainError("Other ComfyUI jobs are still queued. This tool never clears your queue.")
            time.sleep(self.poll_seconds)

    def clear_memory(self):
        self.wait_idle()
        self.request("/free", {"unload_models": True, "free_memory": True})
        # /free is asynchronous. Two small queue fences guarantee the worker
        # has passed its post-job flags/reset/GC block before another model job.
        barriers = []
        for _ in range(2):
            ident = self.queue({"cleanup": {"class_type": "PreviewAny", "inputs": {
                "source": "H3 disk-chain cleanup fence " + uuid.uuid4().hex},
                "_meta": {"title": "H3 disk chain: release previous job cache"}}})
            barriers.append(ident)
            self.wait(ident)
        self.request("/history", {"delete": barriers})

    def cancel_owned(self):
        if not self.current:
            return
        try:
            state = self.request("/queue")
            active = any(str(row[1]) == self.current for row in state.get("queue_running", []))
            if active:
                self.request("/interrupt", {})
            else:
                self.request("/queue", {"delete": [self.current]})
        except ChainError:
            pass


def schema_check(client, graph):
    info = client.request("/object_info")
    required = {n["class_type"] for n in graph.values()} | {"GetVideoComponents", "SaveImage", "PreviewAny"}
    missing = sorted(required - set(info))
    if missing:
        raise ChainError("Required installed nodes are missing: " + ", ".join(missing))
    # Reject missing models before beginning a long disk conversion.
    for kind, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                        ("VAELoader", "vae_name"), ("LoraLoaderModelOnly", "lora_name"),
                        ("CheckpointLoaderSimple", "ckpt_name")):
        spec = info[kind].get("input", {})
        inputs = {**spec.get("required", {}), **spec.get("optional", {})}
        choices = inputs.get(field, [None])[0]
        if isinstance(choices, list):
            for node in graph.values():
                if node["class_type"] == kind and node["inputs"][field] not in choices:
                    raise ChainError("Model is not visible in this Pod: " + node["inputs"][field])


def log_event(run_dir, message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def check_stop(path):
    if path.exists():
        raise Stopped("STOP file found. Completed sections are retained.")

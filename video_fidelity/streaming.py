"""Frame-indexed bounded streaming. Blend predictions of SAME frames, not neighbors."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    overlap_before: int
    hold_after: int
    shot: int


def windows(count: int, cuts: list[int], size: int = 33, overlap: int = 5):
    if count < 1 or size <= 2 * overlap or overlap < 1:
        raise ValueError('Invalid frame count/window/overlap')
    boundaries = [0] + sorted({int(x) for x in cuts if 0 < x < count}) + [count]
    for shot, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        start = lo
        while start < hi:
            stop = min(start + size, hi)
            before = 0 if start == lo else overlap
            after = overlap if stop < hi else 0
            yield Window(start, stop, before, after, shot)
            if stop == hi:
                break
            start = stop - overlap


def source_mix(original: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    """Pixel blend, not a denoise parameter or a hallucination detector."""
    if original.shape != restored.shape or not 0 <= strength <= 1:
        raise ValueError('Source/restoration geometry or strength is invalid')
    if not np.isfinite(restored).all():
        raise ValueError('Non-finite restoration pixels')
    return np.clip(original + strength * (restored - original), 0.0, 1.0)


class OverlapWriter:
    def __init__(self, write_frame):
        self.write_frame = write_frame
        self.pending = None
        self.next_index = 0

    def accept(self, window: Window, frames: np.ndarray) -> None:
        if len(frames) != window.stop - window.start:
            raise ValueError('Model returned a different number of frames')
        if window.start != self.next_index:
            raise ValueError('Frame order changed at streaming boundary')
        before = window.overlap_before
        if before:
            if self.pending is None or len(self.pending) != before:
                raise ValueError('Missing overlap predictions')
            # Raised-cosine weights. These refer to matching frame indices.
            t = np.linspace(0.0, 1.0, before, dtype=np.float32)
            new_weight = (0.5 - 0.5 * np.cos(np.pi * t)).reshape((-1, 1, 1, 1))
            frames[:before] = self.pending * (1.0 - new_weight) + frames[:before] * new_weight
        elif self.pending is not None:
            raise ValueError('Attempt to blend across a scene cut')
        stop = len(frames) - window.hold_after
        for frame in frames[:stop]:
            self.write_frame(self.next_index, frame)
            self.next_index += 1
        self.pending = frames[stop:].copy() if window.hold_after else None

    def finish(self, count: int) -> None:
        if self.pending is not None or self.next_index != count:
            raise ValueError('Incomplete output sequence')

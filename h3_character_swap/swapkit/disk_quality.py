"""Small, source-aware join diagnostics. These are NOT perceptual guarantees.

No learned metric, facial recognition, optical-flow model or cloud service.
Inputs are bounded thumbnails from already generated/source frames. Thresholds
are conservative starting values, not calibrated against the user's footage.
"""
from __future__ import annotations
import numpy as np

POLICY = 'source-aware-join-v1'


def _gray(x):
    x = np.asarray(x, dtype=np.float32)
    return x[..., 0] * .299 + x[..., 1] * .587 + x[..., 2] * .114


def _delta(a, b, mask):
    return float(np.abs(a-b)[mask].mean())


def _steps(frames, mask):
    return [_delta(a, b, mask) for a, b in zip(frames[:-1], frames[1:])]


def _motion(a, b, mask, radius=6):
    """Median coarse block displacement in thumbnail pixels, not dense flow."""
    h, w = a.shape
    vectors = []
    for y in range(radius, h-radius-8, 12):
        for x in range(radius, w-radius-8, 12):
            if np.mean(mask[y:y+8, x:x+8]) < .6:
                continue
            tile = a[y:y+8, x:x+8]
            if float(tile.std()) < .035:
                continue
            best = (float('inf'), 0, 0)
            # Prefer zero displacement on an exact tie.
            shifts = sorted(((dy, dx) for dy in range(-radius, radius+1, 2)
                             for dx in range(-radius, radius+1, 2)),
                            key=lambda p: p[0]*p[0]+p[1]*p[1])
            for dy, dx in shifts:
                err = float(np.mean(np.abs(tile-b[y+dy:y+dy+8, x+dx:x+dx+8])))
                if err < best[0]:
                    best = (err, dx, dy)
            if best[0] < .16:
                vectors.append(best[1:])
    if len(vectors) < 4:
        return None
    return np.median(np.asarray(vectors), axis=0)


def assess_join(previous, current):
    """previous/current are the small QC dictionaries written by DiskSave.

    History is excluded from current['head']; boundary is the first NEW frame.
    There is deliberately no crossfade, interpolation or automatic time warp.
    """
    p, n = _gray(previous['tail']), _gray(current['head'])
    sp, sn = _gray(previous['source_tail']), _gray(current['source_head'])
    pm, nm = previous['mask_tail'][-1] > .5, current['mask_head'][0] > .5
    if p.ndim != 3 or n.ndim != 3 or min(len(p), len(n)) < 1 or p.shape[1:] != n.shape[1:]:
        raise ValueError('Incompatible join diagnostics')
    if not all(np.isfinite(x).all() for x in (p, n, sp, sn)):
        raise ValueError('Non-finite join diagnostics')
    roi = pm | nm
    if roi.sum() < 16:
        return {'policy': POLICY, 'verdict': 'review', 'flags': ['missing_subject_mask'],
                'limitations': 'No trustworthy subject region to assess.'}
    local = _steps(p[-6:], roi) + _steps(n[:6], roi)
    source_local = _steps(sp[-6:], roi) + _steps(sn[:6], roi)
    baseline = float(np.median(local)) if local else 0.
    source_baseline = float(np.median(source_local)) if source_local else 0.
    boundary = _delta(p[-1], n[0], roi)
    source_boundary = _delta(sp[-1], sn[0], roi)
    brightness = abs(float(n[0][roi].mean()-p[-1][roi].mean()))
    source_brightness = abs(float(sn[0][roi].mean()-sp[-1][roi].mean()))
    threshold = max(.12, 4.0*baseline, 2.5*source_boundary)
    flags = []
    if boundary > threshold:
        flags.append('appearance_jump')
    if brightness > max(.16, 3.0*source_brightness, 3.0*baseline):
        flags.append('brightness_jump')
    if len(n) >= 4:
        out_change = float(np.median(_steps(n[:6], roi)))
        src_change = float(np.median(_steps(sn[:6], roi)))
        if out_change < .0015 and src_change > .015:
            flags.append('possible_frozen_continuation')
    # Compare the source before accusing a natural hard cut of being an AI seam.
    source_cut = source_boundary > max(.30, 4.0*source_baseline) and source_brightness > .08
    if source_cut:
        flags.append('possible_source_cut')
    flow = {}
    if min(len(p), len(n), len(sp), len(sn)) >= 3:
        old = _motion(p[-2], p[-1], roi)
        join = _motion(p[-1], n[0], roi)
        new = _motion(n[0], n[1], roi)
        src_old = _motion(sp[-2], sp[-1], roi)
        src_join = _motion(sp[-1], sn[0], roi)
        if all(x is not None for x in (old, join, new, src_old, src_join)):
            jerk = float(np.linalg.norm(join-(old+new)/2))
            source_jerk = float(np.linalg.norm(src_join-src_old))
            flow = {'boundary_vector': join.tolist(), 'preceding_vector': old.tolist(),
                    'following_vector': new.tolist(), 'jump': jerk, 'source_jump': source_jerk}
            if jerk > max(5., 3.*source_jerk) and boundary > max(.03, 2*baseline):
                flags.append('possible_motion_jump')
    return {'policy': POLICY, 'verdict': 'review' if flags else 'pass', 'flags': flags,
            'subject_pixels': int(roi.sum()), 'appearance_delta': boundary,
            'appearance_threshold': threshold, 'neighbor_delta': baseline,
            'source_delta': source_boundary, 'brightness_delta': brightness,
            'source_brightness_delta': source_brightness, 'coarse_motion': flow,
            'limitations': 'Heuristic screen only. A pass does not certify likeness, anatomy, lip sync or seamless motion.'}

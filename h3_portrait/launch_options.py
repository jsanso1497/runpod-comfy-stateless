#!/usr/bin/env python3
"""Validated memory flags, independent of model precision and render quality."""
import math
import os


def options(environ=None):
    env = os.environ if environ is None else environ
    dynamic = env.get('H3_DYNAMIC_VRAM', '0')
    if dynamic not in ('0', '1'):
        raise ValueError('H3_DYNAMIC_VRAM must be 0 or 1.')
    reserve = float(env.get('H3_RESERVE_VRAM', '4'))
    if not math.isfinite(reserve) or not 0 <= reserve <= 32:
        raise ValueError('H3_RESERVE_VRAM must be between 0 and 32 GB.')
    return ['--disable-dynamic-vram' if dynamic == '0' else '--enable-dynamic-vram',
            '--reserve-vram', str(reserve)]


if __name__ == '__main__':
    print('\n'.join(options()))

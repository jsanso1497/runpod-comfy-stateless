#!/usr/bin/env python3
"""Keep ComfyUI runtime dependencies but omit optional bundled demo-template media.

ComfyUI's workflow templates are optional gallery assets. This does NOT strip
transformers, Torch, frontend, SAM, Qwen, or image-processing dependencies.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re

OPTIONAL_DEMO = re.compile(r'^\s*comfyui[-_]workflow[-_]templates(?:\s*\[.*?\])?\s*(?:[<>=!~;].*)?\s*(?:#.*)?$', re.I)


def filter_file(source: Path, output: Path) -> list[str]:
    lines=source.read_text(encoding='utf-8').splitlines(keepends=True)
    skipped=[]; kept=[]
    for line in lines:
        if OPTIONAL_DEMO.fullmatch(line.strip()):
            skipped.append(line.strip())
        else:
            kept.append(line)
    if not skipped:
        raise RuntimeError(f'Expected exactly one ComfyUI demo workflow-template requirement in {source}; upstream changed. Review dependencies rather than silently skipping.')
    if len(skipped)!=1:
        raise RuntimeError(f'Expected one demo workflow-template line, found {len(skipped)}: {skipped}')
    for required in ('torch', 'transformers', 'comfyui-frontend-package'):
        if not any(re.match(r'^\s*'+re.escape(required)+r'(?=[\s<>=!~\[]|$)',line,re.I) for line in kept):
            raise RuntimeError(f'Core dependency {required} is unexpectedly missing; refusing to trim')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(kept),encoding='utf-8')
    return skipped


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();lines=filter_file(a.source,a.output)
    print('Excluded optional bundled workflow-template gallery media:', ', '.join(lines))
    print('All model/runtime requirements retained; ComfyUI UI gallery may have fewer built-in examples.')

if __name__=='__main__':main()

"""Public model choices and authenticated local custom-asset metadata, without URLs."""
from __future__ import annotations
import json
from pathlib import Path
from .config import model_root, read_catalog, state_root

FAMILY_SIGNATURES = {
    'qwen': 'qwen-image-2.1',
    'flux': 'flux2-klein-9b-distilled',
    'h3': 'minimax-h3-ref2va',
    'restoration': 'seedvr2-7b',
}

def private_records() -> list[dict]:
    path=state_root()/'private/assets.json'
    if not path.is_file():return []
    return json.loads(path.read_text()).get('assets',[])

def standard_files(family: str, kind: str) -> list[str]:
    return [a['destination'].split('/',1)[1] for a in read_catalog('assets.json')['assets']
            if a['family']==family and a['destination'].split('/',1)[0]==kind]

def choices(family: str, kind: str, *, installed_only: bool=False) -> list[str]:
    values=standard_files(family,kind)
    values += [r['filename'] for r in private_records() if r.get('family')==family and r.get('kind')==kind
               and r.get('compatibility')=='declared' and r.get('status')=='ready']
    values=list(dict.fromkeys(values))
    if installed_only:values=[v for v in values if (model_root()/kind/v).is_file()]
    return values

def check_selection(family: str, kind: str, filename: str) -> str:
    if filename not in choices(family,kind):
        raise ValueError(f'This file is not registered for {family}/{kind}. Declare its exact family and architecture in the private library; a filename alone is not proof of compatibility.')
    path=model_root()/kind/filename
    if not path.is_file():
        raise FileNotFoundError('Required model is not installed. Open Workbench and prepare this task before running it.')
    return filename

def lora_choices(family: str) -> list[str]:
    return [r['filename'] for r in private_records() if r.get('family')==family and r.get('kind')=='loras'
            and r.get('compatibility')=='declared' and r.get('status')=='ready']

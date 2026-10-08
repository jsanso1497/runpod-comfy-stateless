"""Runtime-only verified asset downloads. Never used during Docker builds.

Private manifest URLs are not persisted, echoed, or embedded in image metadata.
Provider metadata is authoritative for a transfer, while source-pinned hashes
remain mandatory for the standard model bundles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from urllib.parse import quote,urlsplit
from . import provider_downloads as pd
from .config import (FAMILIES, atomic_json, csv, model_root, read_catalog,
                     requested_groups, resolve_secret_aliases, safe_name, safe_path, state_root, workspace)
from .registry import FAMILY_SIGNATURES

ALLOWED_KINDS={'loras','diffusion_models','text_encoders','vae','checkpoints'}

class AssetError(ValueError):
    """A credential-free actionable failure."""

def parse_library(value: str, kind: str) -> list[dict]:
    if not value.strip():return []
    if value.lstrip().startswith('['):
        try: rows=json.loads(value)
        except ValueError:raise AssetError('Private asset JSON is invalid. Use the example schema without adding actual keys.') from None
        if not isinstance(rows,list):raise AssetError('The private asset manifest must be a JSON array.')
    else:
        rows=[]
        for line in value.splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            fields=[s.strip() for s in line.split('|')]
            if len(fields)==1:rows.append({'url':fields[0], 'kind':kind})
            elif len(fields)==2:rows.append({'family':fields[0], 'url':fields[1], 'kind':kind})
            else:raise AssetError('Text library format is one URL, or family | URL, per line.')
    normalized=[]
    for i,row in enumerate(rows,1):
        if isinstance(row,str):row={'url':row}
        if not isinstance(row,dict):raise AssetError(f'Private entry {i} must be an object or URL.')
        row=dict(row)
        unknown=set(row)-{'url','family','kind','name','architecture','sha256','precision','enabled'}
        if unknown:raise AssetError(f'Private entry {i} contains unsupported fields.')
        if not row.get('enabled',True):continue
        try:row['url']=pd.clean_url(row.get('url',''))
        except pd.LinkError as exc:raise AssetError(f'Private entry {i}: {exc}') from None
        row['kind']=row.get('kind',kind)
        if row['kind'] not in ALLOWED_KINDS:raise AssetError(f'Private entry {i}: unsupported asset kind.')
        family=row.get('family')
        if family and family not in FAMILIES:raise AssetError(f'Private entry {i}: unsupported family.')
        if kind=='loras' and row['kind']!='loras':raise AssetError('WB_LORA_URLS accepts only LoRA adapters.')
        if row.get('name'):safe_name(row['name'])
        if row.get('sha256') and not re.fullmatch('[0-9a-fA-F]{64}',row['sha256']):raise AssetError('Asset SHA256 must contain exactly 64 hexadecimal digits.')
        if row.get('precision') and row['precision'].lower() not in ('bf16','fp16','fp32','f16'):
            raise AssetError('This workbench does not register quantized model variants.')
        if row.get('architecture') and family and row['architecture']!=FAMILY_SIGNATURES[family]:
            raise AssetError('The declared architecture does not match its workflow family.')
        normalized.append(row)
    return normalized

def library_value(env_key: str, file_key: str) -> str:
    if os.environ.get(env_key,'').strip():
        if os.environ.get(file_key):print(f'{env_key} takes precedence over its runtime file fallback.',flush=True)
        return os.environ[env_key]
    if os.environ.get(file_key):
        path=Path(os.environ[file_key])
        if not path.is_file():raise AssetError(f'{file_key} does not identify an accessible runtime file.')
        return path.read_text(encoding='utf-8-sig')
    return ''

def inspect_safetensors(path: Path, *, high_precision: bool=False) -> dict:
    try:
        size=path.stat().st_size
        with path.open('rb') as h:
            length,=struct.unpack('<Q',h.read(8))
            if not 2<=length<=min(64*1024*1024,size-8):raise ValueError()
            header=json.loads(h.read(length))
        spans=[]; dtypes=set(); tensors=0
        for name,row in header.items():
            if name=='__metadata__':continue
            start,end=row['data_offsets']
            if not isinstance(start,int) or not isinstance(end,int) or start<0 or end<start:raise ValueError()
            bits={'BOOL':8,'U8':8,'I8':8,'F8_E4M3':8,'F8_E5M2':8,'F8_E4M3FN':8,'I16':16,'U16':16,'F16':16,'BF16':16,'I32':32,'U32':32,'F32':32,'I64':64,'U64':64,'F64':64}
            shape=row['shape']
            if row['dtype'] not in bits or not isinstance(shape,list) or any(type(x) is not int or x<0 for x in shape):raise ValueError()
            count=1
            for dimension in shape:count*=dimension
            if end-start!=(count*bits[row['dtype']]+7)//8:raise ValueError()
            spans.append((start,end));dtypes.add(row['dtype']);tensors+=1
        last=0
        for start,end in sorted(spans):
            if start!=last:raise ValueError()
            last=end
        if not tensors or 8+length+last!=size:raise ValueError()
    except (OSError,ValueError,TypeError,KeyError,struct.error,UnicodeDecodeError):
        raise AssetError('The downloaded file is not a complete safetensors model.') from None
    if high_precision:
        if any(x.startswith(('F8','I8','U8','I4','U4')) for x in dtypes):
            raise AssetError('Quantized model tensors were found. No automatic lower-precision substitution is allowed.')
        metadata=json.dumps(header.get('__metadata__',{})).lower()
        if any(x in metadata for x in ['int8_convrot','nvfp4','awq','gguf']):
            raise AssetError('Quantized model metadata was found. Use the full-precision checkpoint.')
    return {'tensors':tensors,'dtypes':sorted(dtypes),'bytes':size}

def transfer(row: dict, target: Path, transport: pd.Transport, *, high_precision=False) -> dict:
    """Resume to a .part file, verify hash and structure, then atomically publish."""
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix(target.suffix+'.part')
    if target.is_symlink() or part.is_symlink():raise AssetError('Refusing a symlinked model destination.')
    if pd.verified(target,row):return inspect_safetensors(target,high_precision=high_precision)
    size=int(row.get('estimated_bytes') or 0)
    if size<=0:raise AssetError('Provider did not supply a usable model size.')
    reserve=8*1024**3
    last='The model transfer did not finish.'
    for attempt in range(3):
        offset=part.stat().st_size if part.exists() else 0
        if offset and pd.verified(part,row):
            info=inspect_safetensors(part,high_precision=high_precision);part.replace(target);return info
        if shutil.disk_usage(target.parent).free<max(0,size-offset)+reserve:
            raise AssetError('Insufficient storage for the selected asset plus 8 GiB of working space. Increase the data disk.')
        try:
            with transport.request('GET',row['download_url'],stream=True,headers={'Range':f'bytes={offset}-'} if offset else {},redirects=True) as response:
                if response.status_code==416:
                    part.unlink(missing_ok=True);raise pd.LinkError('Partial range changed; transfer will restart.')
                pd.status_error(response.status_code)
                append=response.status_code==206
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise pd.LinkError('Unexpected provider download range.')
                if any(x in response.headers.get('Content-Type','').lower() for x in ('text/html','application/json')):
                    raise pd.LinkError('Provider returned a login or error page instead of model bytes.')
                if not append:
                    offset=0
                    if shutil.disk_usage(target.parent).free<size+reserve:raise AssetError('Provider restarted the transfer but storage is insufficient.')
                count=offset; last_emit=time.monotonic()
                with part.open('ab' if append else 'wb') as handle:
                    for block in response.iter_content(4*1024*1024):
                        if not block:continue
                        handle.write(block);count+=len(block)
                        if count>size*1.02+1024*1024:raise pd.LinkError('Provider transfer exceeded the declared file size.')
                        if shutil.disk_usage(target.parent).free<reserve:raise AssetError('Transfer stopped to preserve working disk space.')
                        if time.monotonic()-last_emit>20:
                            print(f'Asset progress: {100*count/size:.0f} percent.',flush=True);last_emit=time.monotonic()
            if not pd.verified(part,row):
                part.unlink(missing_ok=True);raise pd.LinkError('File checksum mismatch; invalid bytes were discarded.')
            info=inspect_safetensors(part,high_precision=high_precision)
            part.replace(target)
            return info
        except (pd.LinkError,OSError,pd.requests.RequestException) as exc:
            last=str(exc) if isinstance(exc,pd.LinkError) else 'Provider network or local file error.'
            if attempt<2:time.sleep(attempt+1)
    raise AssetError(last)

def resolve_standard(asset:dict,transport:pd.Transport)->dict:
    url=f'https://huggingface.co/{asset["repo_id"]}/resolve/{quote(asset["revision"],safe="")}/{quote(asset["filename"],safe="/")}'
    row=pd.resolve_hf(url,transport)
    if asset.get('sha256'):
        if row.get('sha256') and row['sha256'].lower()!=asset['sha256'].lower():
            raise AssetError('Standard model metadata differs from the source-pinned SHA256. The catalog must be reviewed; no replacement was downloaded.')
        row['sha256']=asset['sha256'].lower();row['git_blob_sha1']=''
    return row

def prepare_standard(groups:list[str])->dict:
    known={a['group'] for a in read_catalog('assets.json')['assets']}|{'ollama'}
    if set(groups)-known:raise AssetError('Unknown model group requested.')
    installed=[];failures=[]; transport=pd.Transport()
    try:
        for asset in read_catalog('assets.json')['assets']:
            if asset['group'] not in groups:continue
            try:
                print('Preparing standard asset: '+asset['id'],flush=True)
                target=safe_path(model_root(),asset['destination'])
                # Cached source-pinned files do not require provider access on each boot.
                if asset.get('sha256') and target.is_file() and pd.file_hash(target)==asset['sha256']:
                    info=inspect_safetensors(target,high_precision=True);resolved=None
                else:
                    row=resolve_standard(asset,transport)
                    info=transfer(row,target,transport,high_precision=True);resolved=row.get('revision')
                installed.append({'id':asset['id'],'group':asset['group'],'destination':asset['destination'],'sha256':pd.file_hash(target),'resolved_revision':resolved,'inspection':info})
            except (AssetError,pd.LinkError,OSError,ValueError,KeyError,TypeError) as exc:
                msg=str(exc) if isinstance(exc,(AssetError,pd.LinkError)) else 'Asset metadata or local storage error.'
                failures.append({'id':asset['id'],'error':msg});print('Asset unavailable: '+asset['id']+'. '+msg,flush=True)
    finally:transport.close()
    oldpath=state_root()/'asset-status.json'
    previous=json.loads(oldpath.read_text()).get('installed',[]) if oldpath.is_file() else []
    lookup={r['id']:r for r in previous}
    for row in installed:lookup[row['id']]=row
    report={'installed':list(lookup.values()),'failures':failures,'last_groups':groups}
    atomic_json(oldpath,report)
    return report

def infer_family(metadata:str)->str|None:
    text=re.sub('[^a-z0-9]+','',str(metadata).lower())
    if 'qwenimage21' in text:return 'qwen'
    if 'flux2klein9b' in text:return 'flux'
    if 'minimaxh3' in text:return 'h3'
    if 'seedvr2' in text:return 'restoration'
    return None

def prepare_private()->dict:
    entries=parse_library(library_value('WB_LORA_URLS','WB_LORA_FILE'),'loras')
    entries+=parse_library(library_value('WB_CHECKPOINTS','WB_CHECKPOINT_FILE'),'diffusion_models')
    index=state_root()/'private/assets.json'
    index.parent.mkdir(parents=True,exist_ok=True);index.parent.chmod(0o700)
    old=json.loads(index.read_text()).get('assets',[]) if index.is_file() else []
    records={r['id']:r for r in old}; failures=[]; transport=pd.Transport()
    try:
        for number,entry in enumerate(entries,1):
            if entry.get('family') and entry['family']!=workspace():continue
            try:
                url=entry['url']; kind=entry['kind']
                allowed=('checkpoint','model','vae','textualinversion') if kind!='loras' else None
                row=pd.resolve_hf(url,transport) if urlsplit(url).hostname=='huggingface.co' else pd.resolve_civitai(url,transport,allowed_types=allowed)
                family=entry.get('family') or infer_family(row.get('reported_base_model',''))
                if family and family!=workspace():continue
                if not family:raise AssetError('Family could not be verified from provider metadata. Prefix the private URL with qwen |, flux |, h3 |, or restoration |.')
                if kind!='loras' and not entry.get('architecture'):
                    raise AssetError('Custom models need an explicit architecture field in the JSON manifest. Use the private-checkpoints example; model families alone do not establish sampler compatibility.')
                if entry.get('architecture') and entry['architecture']!=FAMILY_SIGNATURES[family]:raise AssetError('Custom checkpoint architecture does not match the selected family.')
                if entry.get('sha256') and row.get('sha256') and entry['sha256'].lower()!=row['sha256'].lower():raise AssetError('The private file does not match its declared SHA256.')
                id=hashlib.sha256((url+'|'+kind+'|'+family).encode()).hexdigest()[:16]
                label=entry.get('name') or ('private-'+id)
                filename=f'Private/{family}/{safe_name(label)}.safetensors'
                if any(r.get('filename')==filename and r.get('kind')==kind and key!=id for key,r in records.items()):
                    raise AssetError('Two private entries use the same destination name. Give each asset a unique private name.')
                target=safe_path(model_root(),kind+'/'+filename)
                if entry.get('sha256'):row['sha256']=entry['sha256'].lower();row['git_blob_sha1']=''
                info=transfer(row,target,transport,high_precision=kind!='loras')
                if kind=='loras':pd.check_adapter_file(target)
                records[id]={'id':id,'family':family,'kind':kind,'filename':filename,'architecture':FAMILY_SIGNATURES[family],
                    'compatibility':'declared','status':'ready','sha256':pd.file_hash(target),'inspection':info}
                print(f'Private asset {number} ready. Source URL and filename omitted from log.',flush=True)
            except (AssetError,pd.LinkError,OSError,ValueError,KeyError,TypeError) as exc:
                msg=str(exc) if isinstance(exc,(AssetError,pd.LinkError)) else 'Provider metadata or local storage error.'
                failures.append({'entry':number,'error':msg});print(f'Private asset {number} unavailable: '+msg,flush=True)
    finally:transport.close()
    report={'assets':list(records.values()),'failures':failures}
    atomic_json(index,report,private=True)
    return report

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups',default=None)
    parser.add_argument('--private',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    a=parser.parse_args();resolve_secret_aliases()
    if a.check_config:
        parse_library(library_value('WB_LORA_URLS','WB_LORA_FILE'),'loras')
        parse_library(library_value('WB_CHECKPOINTS','WB_CHECKPOINT_FILE'),'diffusion_models')
        print('Private configuration syntax checked; contents were not printed.');return 0
    from filelock import FileLock
    state_root().mkdir(parents=True,exist_ok=True)
    with FileLock(str(state_root()/'downloads.lock'),timeout=1):
        if a.private:report=prepare_private()
        else:report=prepare_standard(csv(a.groups) if a.groups is not None else requested_groups())
    return 1 if report.get('failures') else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssetError,pd.LinkError,ValueError) as exc:
        print('Asset configuration failed: '+str(exc),file=sys.stderr);raise SystemExit(2)

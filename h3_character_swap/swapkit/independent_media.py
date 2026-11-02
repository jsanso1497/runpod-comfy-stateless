"""Frame-exact independent sections and disk-streamed assembly, without H3 history.

Only source conditioning is shared between manual generations. Joining is an edit,
not another diffusion pass. No colour matching, interpolation, or face morphing is
silently applied. Work inside local input/output storage; never execute shell text.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from fractions import Fraction

from .config import atomic_json, data_root, file_sha
from .disk_common import media_info, media_duration, display_size
from .recovery import ffmetadata, json_text, sidecars

FPS = 24
VIDEO_EXT = {'.mp4', '.mkv', '.mov', '.webm', '.m4v'}
WINDOWS = (56, 73, 90, 107, 124)


def local_media(name, storage=None):
    """Accept input/foo, output/foo or absolute paths *inside* those roots."""
    root = Path(storage or data_root()).resolve()
    if not isinstance(name, str) or not name.strip() or any(c in name for c in ('\x00', '\n', '\r')):
        raise ValueError('Choose a local video in input/ or output/.')
    path = Path(name.strip())
    if not path.is_absolute():
        if path.parts[0] not in ('input', 'output') or '..' in path.parts:
            raise ValueError('Use input/filename.mkv or output/subfolder/filename.mp4, not a URL or glob.')
        path = root / path
    path = path.resolve()
    if not any(path.is_relative_to(root / x) and path != root / x for x in ('input', 'output')):
        raise ValueError('Video must remain inside the input or output directory.')
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
        raise ValueError('Local video not found: ' + str(path))
    return path


def choices(storage=None, include_output=True):
    root = Path(storage or data_root()).resolve()
    result = []
    for category in ('input', 'output') if include_output else ('input',):
        directory = root / category
        if not directory.is_dir(): continue
        # Bound listing work, do not follow directory symlinks into other trees.
        for parent, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(parent)/d).is_symlink())
            for name in sorted(files):
                p = Path(parent)/name
                if p.suffix.lower() in VIDEO_EXT and not p.is_symlink() and '.partial.' not in name:
                    result.append(p.relative_to(root).as_posix())
                    if len(result) >= 5000: return sorted(result)
    return sorted(result) or ['input/upload_a_video_first.mp4']


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Name must be 1-64 letters, numbers, underscores or hyphens.')
    return value


def run_ffmpeg(args, log, cancel=lambda: None, timeout=7200):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    cancel()
    with log.open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while proc.poll() is None:
                cancel()
                if time.monotonic()-started > timeout: raise TimeoutError('Media operation timed out: '+str(log))
                if shutil.disk_usage(log.parent).free < 128*1024**2:
                    raise ValueError('Insufficient disk space. Existing videos have not been overwritten.')
                time.sleep(.1)
            if proc.returncode:
                raise ValueError('FFmpeg failed: '+log.read_text(errors='replace')[-2500:])
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
    cancel()


def _ff():
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-threads', '2']


def video_info(path, count=True):
    info = media_info(Path(path), count=count)
    v = info['video']
    if v.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR needs an explicitly approved SDR conversion before this workflow.')
    if count:
        try: n = int(v.get('nb_read_frames') or v.get('nb_frames'))
        except (TypeError, ValueError): raise ValueError('Cannot count video frames: '+str(path))
        if n < 1: raise ValueError('Video has no decodable frames.')
        info['frames'] = n
    return info


def require_24(info):
    try: rate = Fraction(info['video'].get('avg_frame_rate', '0/1'))
    except (ValueError, ZeroDivisionError): rate = 0
    if rate != FPS:
        raise ValueError('Use 24 fps clips from workflow 11. Joining does not silently retime other frame rates.')


def plan_chunks(total, window=124, overlap=12):
    if type(total) is not int or total < 1 or window not in WINDOWS:
        raise ValueError('Invalid source frame count or window. Choose 56, 73, 90, 107 or 124.')
    if type(overlap) is not int or not 0 <= overlap <= min(24, window//2):
        raise ValueError('Overlap must be 0..24 frames and no more than half a window.')
    rows=[]; start=0
    while True:
        end=min(start+window,total)
        rows.append({'index':len(rows)+1,'start_frame':start,'end_frame':end,'frames':end-start})
        if end==total: break
        start=end-overlap
    return rows


def _extract_frames(source, target, start, length, log, cancel):
    # Count from zero, not fractional Matroska timestamps. Source is already 24 fps.
    vf=f'trim=start_frame={start}:end_frame={start+length},setpts=N/(24*TB)'
    args=_ff()+['-i',str(source),'-map','0:v:0','-map','0:a:0?','-vf',vf,
                '-r','24','-fps_mode','cfr','-c:v','ffv1','-level','3','-g','1','-pix_fmt','gbrp16le','-threads','2',
                '-af',f'atrim=start_sample={start*2000}:end_sample={(start+length)*2000},asetpts=PTS-STARTPTS',
                '-c:a','pcm_s16le','-map_metadata','-1',str(target)]
    run_ffmpeg(args, log, cancel)
    got=video_info(target);require_24(got)
    if got['frames']!=length: raise ValueError('Frame-exact extraction failed; no completed manifest published.')
    return got


def split_source(source, set_name, duration_seconds=48., start_seconds=0., window=124, overlap=12,
                 storage=None, cancel=lambda:None):
    """Conform once on disk. Complete tiny tail is kept, not skipped or extended."""
    root=Path(storage or data_root()).resolve();source=local_media(str(source),root)
    safe_name(set_name)
    if not math.isfinite(start_seconds) or start_seconds<0: raise ValueError('Invalid start time.')
    if not math.isfinite(duration_seconds) or not 0<=duration_seconds<=3600: raise ValueError('Duration must be 0..3600 seconds; 0 keeps the remainder.')
    plan_chunks(1,window,overlap)
    info=video_info(source,False)
    remaining=media_duration(info)-start_seconds
    if remaining<=0: raise ValueError('Start time is beyond the source.')
    duration=remaining if duration_seconds==0 else min(duration_seconds,remaining)
    expected=math.floor(duration*FPS+1e-7)
    if expected<1: raise ValueError('Selected source interval is shorter than one frame.')
    parent=root/'input/h3_independent';parent.mkdir(parents=True,exist_ok=True)
    dest=parent/set_name
    if dest.exists(): raise ValueError('This clip set already exists. Choose a new set_name; nothing was overwritten.')
    dest.mkdir()
    try:
        w,h=display_size(info['video']);w=max(2,round(w/2)*2);h=max(2,round(h/2)*2)
        if max(w,h)>8192: raise ValueError('Source is over 8192 pixels. Prepare a smaller source explicitly.')
        normalized=dest/'_source_24fps.mkv'
        args=_ff()+['-ss',f'{start_seconds:.9f}','-i',str(source),'-map','0:v:0','-map','0:a:0?',
                    '-vf',f'setpts=PTS-STARTPTS,fps=fps=24:round=near:eof_action=round,scale={w}:{h}:flags=lanczos,setsar=1,trim=end_frame={expected},setpts=N/(24*TB)',
                    '-r','24','-fps_mode','cfr','-c:v','ffv1','-level','3','-g','1','-pix_fmt','gbrp16le','-threads','2',
                    '-af',f'aresample=48000:async=1:first_pts=0,apad,atrim=end_sample={expected*2000}',
                    '-c:a','pcm_s16le','-ar','48000','-ac','2','-t',f'{expected/FPS:.9f}',
                    '-map_metadata','-1',str(normalized)]
        run_ffmpeg(args,dest/'conform.log',cancel)
        conformed=video_info(normalized);require_24(conformed)
        if abs(conformed['frames']-expected)>1:
            raise ValueError('Source conform duration differs from the requested interval.')
        total=conformed['frames'];rows=plan_chunks(total,window,overlap)
        source_hash=file_sha(source)
        manifest={'format':'h3-independent-source-v1','source':str(source), 'source_sha256':source_hash,
                  'set_name':set_name,'start_seconds':start_seconds,'frames':total,'fps':FPS,
                  'window_frames':window,'overlap_frames':overlap,'seconds':total/FPS,
                  'source_audio':str(normalized),'clips':rows,
                  'note':'Editorial overlap only. No generated history is carried into another render.'}
        for row in rows:
            cancel();target=dest/f"clip_{row['index']:04d}.mkv"
            _extract_frames(normalized,target,row['start_frame'],row['frames'],dest/f"clip_{row['index']:04d}.log",cancel)
            row['file']=target.relative_to(root).as_posix();row['sha256']=file_sha(target)
            atomic_json(target.with_suffix('.source.json'),{'format':'h3-independent-clip-v1',
                'source_sha256':source_hash,'set_name':set_name,'source_start_seconds':start_seconds,
                'total_frames':total,'overlap_frames':overlap,**row})
        atomic_json(dest/'chunks.json',manifest)
        (dest/'ORDER.txt').write_text('\n'.join(r['file'] for r in rows)+'\n')
        return manifest, dest/'chunks.json'
    except BaseException:
        # A failed, new set contains only products made by this call. Keep its logs
        # for diagnosis, but never present it as a ready set or overwrite on retry.
        (dest/'INCOMPLETE.txt').write_text('Preparation did not finish. Use a new set name or remove this incomplete set after inspecting its logs.\n')
        raise


def ordered_paths(text, storage=None):
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if not 1<=len(lines)<=500: raise ValueError('Supply 1..500 video filenames, one per line in timeline order.')
    result=[local_media(x,storage) for x in lines]
    if len(set(result))!=len(result): raise ValueError('A video is listed twice. Remove duplicate filenames.')
    return result


def edit_ranges(counts, overlap, cut='Middle of overlap'):
    if type(overlap) is not int or not 0<=overlap<=24: raise ValueError('Overlap must be 0..24 frames.')
    if cut not in ('Middle of overlap','Keep earlier take','Keep later take'): raise ValueError('Unknown cut placement.')
    if not counts or any(type(n) is not int or n<1 for n in counts): raise ValueError('Missing frame counts.')
    if len(counts)>1 and any(n<=overlap for n in counts): raise ValueError('Each joined clip must be longer than the overlap.')
    offset={'Middle of overlap':overlap//2,'Keep earlier take':overlap,'Keep later take':0}[cut]
    rows=[]
    for i,n in enumerate(counts):
        start=offset if i else 0
        end=n-(overlap-offset if i<len(counts)-1 else 0)
        if end<=start: raise ValueError('Overlap would remove this entire clip.')
        rows.append({'start':start,'end':end,'frames':end-start})
    return rows


def _concat_list(paths, target):
    # Only internally generated filenames, never caller strings, enter ffconcat.
    lines=['ffconcat version 1.0']
    for path in paths:
        if "'" in str(path) or '\n' in str(path): raise ValueError('Unsafe temporary path.')
        lines.append("file '"+str(path)+"'")
    Path(target).write_text('\n'.join(lines)+'\n')


def _metadata_args(metadata, directory, input_index):
    if not metadata: return [], ['-map_metadata','-1']
    file=ffmetadata(Path(directory)/'workflow.ffmeta',metadata)
    return ['-f','ffmetadata','-i',str(file)], ['-map_metadata',str(input_index)]


def make_mp4(source, target, crf=16, cancel=lambda:None):
    if type(crf) is not int or not 0<=crf<=30: raise ValueError('CRF must be 0..30.')
    # Matroska uppercases custom tag names. Re-emit canonical lowercase keys for
    # ComfyUI MP4 import, rather than trusting a case-sensitive player to adapt.
    tags=video_info(source,False).get('format',{}).get('tags',{})
    metadata={k.lower():json.loads(v) for k,v in tags.items() if k.lower() in ('workflow','prompt')}
    with tempfile.TemporaryDirectory(prefix='.mp4-meta-',dir=Path(target).parent) as td:
        mi,mo=_metadata_args(metadata,td,1)
        args=_ff()+['-i',str(source)]+mi+['-map','0:v:0','-map','0:a:0?',
            '-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-preset','medium','-crf',str(crf),'-threads','2',
            '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','iec61966-2-1',
            '-c:a','aac','-b:a','192k']+mo+['-movflags','+faststart+use_metadata_tags',str(target)]
        run_ffmpeg(args,Path(target).with_suffix('.encode.log'),cancel)


def validate_provenance(paths, infos, overlap, *, source=None, start_seconds=0.):
    """Check generated receipts when present. Never infer source order visually."""
    records=[]
    for p in paths:
        rec=p.parent/'receipt.json'
        if not rec.exists(): records.append(None);continue
        data=json.loads(rec.read_text())
        if data.get('format')=='h3-independent-render-v1':
            expected=data.get('master_sha256' if p.name=='master.mkv' else 'preview_sha256' if p.name=='preview.mp4' else '')
            if expected and file_sha(p)!=expected: raise ValueError('Rendered clip changed after its receipt was saved.')
        records.append(data if data.get('format')=='h3-independent-render-v1' else None)
    if not any(records): return 'No independent-render receipts: order and source timing are user-declared.'
    if not all(records): raise ValueError('Do not mix tracked independent renders with untracked clips in one join.')
    if len({r['model_profile'] for r in records})!=1: raise ValueError('Cannot join mixed BF16/baseline renders in this tracked take.')
    refs=[r.get('reference_sha256', {}) for r in records]
    if any(refs) and len({json_text(sorted(v.values())) for v in refs})!=1:
        raise ValueError('Character references differ between these clips. Join one consistent character take at a time.')
    origins=[r.get('source_clip') for r in records]
    if not all(origins): return 'Independent renders checked; no split-source receipts available.'
    ids={(r['source_sha256'],r['set_name'],r['source_start_seconds'],r['total_frames']) for r in origins}
    if len(ids)!=1: raise ValueError('Rendered chunks belong to different source sets.')
    for i,(origin,info) in enumerate(zip(origins,infos)):
        if origin['frames']!=info['frames']: raise ValueError('Rendered clip length differs from its original interval.')
        if origin['overlap_frames']!=overlap: raise ValueError('Overlap does not match workflow 11. Use the recorded overlap.')
        if i and origins[i-1]['end_frame']-origin['start_frame']!=overlap:
            raise ValueError('Chunks are reordered, missing, duplicated, or not consecutive.')
    if source is not None:
        if file_sha(source)!=origins[0]['source_sha256']: raise ValueError('Original audio source does not match the chunk set.')
        expected=origins[0]['source_start_seconds']+origins[0]['start_frame']/FPS
        if abs(start_seconds-expected)>1e-5: raise ValueError(f'Audio start must be {expected:.9f} seconds for these chunks.')
    return 'Receipt-verified source order, overlap, lengths and precision profile. Visual continuity is not verified.'


def join_videos(ordered_videos, join_name, overlap_frames=12, cut='Middle of overlap',
                audio_mode='Original source', original_source='', source_start_seconds=0.,
                crf=16, metadata=None, storage=None, cancel=lambda:None):
    root=Path(storage or data_root()).resolve();safe_name(join_name)
    if not math.isfinite(source_start_seconds) or source_start_seconds<0: raise ValueError('Invalid source audio start.')
    if audio_mode not in ('Original source','No audio'): raise ValueError('Choose continuous source audio or No audio.')
    paths=ordered_paths(ordered_videos,root)
    infos=[video_info(p) for p in paths]
    for info in infos: require_24(info)
    sizes={(v['video']['width'],v['video']['height']) for v in infos}
    if len(sizes)!=1: raise ValueError('All clips must have identical resolution. No automatic resizing is applied.')
    w,h=next(iter(sizes))
    if w%2 or h%2: raise ValueError('Output width and height must be even for H.264.')
    for info in infos:
        v=info['video']
        if Fraction(v.get('sample_aspect_ratio','1:1').replace(':','/'))!=1 or any(abs(float(x.get('rotation',0)))%360 for x in v.get('side_data_list',[])):
            raise ValueError('Conform rotated or anamorphic clips before joining.')
    source=local_media(original_source,root) if audio_mode=='Original source' else None
    source_info=video_info(source,False) if source else None
    if source and source_info['audio'] is None: raise ValueError('Original source contains no audio; choose No audio.')
    rows=edit_ranges([i['frames'] for i in infos],int(overlap_frames),cut)
    total=sum(r['frames'] for r in rows)
    if source and source_start_seconds+total/FPS>media_duration(source_info)+1/FPS:
        raise ValueError('Joined duration extends beyond the original source audio/video interval.')
    validation=validate_provenance(paths,infos,int(overlap_frames),source=source,start_seconds=source_start_seconds)
    metadata=metadata or {};json_text(metadata)
    parent=root/'output/h3_character_swap/joined';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/join_name
    if destination.exists(): raise ValueError('Join name already exists. Choose a new join_name; nothing was overwritten.')
    destination.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix='.join-',dir=destination) as td:
            td=Path(td);trimmed=[]
            for i,(p,r) in enumerate(zip(paths,rows)):
                target=td/f'part_{i:04d}.mkv'
                args=_ff()+['-i',str(p),'-map','0:v:0','-an','-vf',
                    f"trim=start_frame={r['start']}:end_frame={r['end']},setpts=N/(24*TB),format=gbrp16le",
                    '-r','24','-fps_mode','cfr','-c:v','ffv1','-level','3','-g','1','-pix_fmt','gbrp16le','-threads','2','-map_metadata','-1',str(target)]
                run_ffmpeg(args,destination/f'trim_{i:04d}.log',cancel)
                if video_info(target)['frames']!=r['frames']: raise ValueError('Trimmed frame count mismatch.')
                trimmed.append(target)
            listing=td/'concat.txt';_concat_list(trimmed,listing)
            args=_ff()+['-f','concat','-safe','0','-i',str(listing)]
            if source: args+=['-ss',f'{source_start_seconds:.9f}','-i',str(source)]
            mi,mo=_metadata_args(metadata,td,2 if source else 1);args+=mi
            args+=['-map','0:v:0','-c:v','copy']
            if source:
                args+=['-map','1:a:0','-af',f'aresample=48000:async=1:first_pts=0,apad,atrim=end_sample={total*2000}',
                       '-c:a','pcm_s16le','-ar','48000','-ac','2']
            else: args+=['-an']
            master=destination/'final.mkv';temp=destination/'final.partial.mkv'
            args+=mo+['-t',f'{total/FPS:.9f}',str(temp)]
            run_ffmpeg(args,destination/'assemble.log',cancel)
            if video_info(temp)['frames']!=total: raise ValueError('Final joined frame count mismatch.')
            os.replace(temp,master);sidecars(master,metadata)
        mp4=destination/'final.mp4'
        make_mp4(master,destination/'final.partial.mp4',int(crf),cancel)
        if video_info(destination/'final.partial.mp4')['frames']!=total: raise ValueError('MP4 frame count mismatch.')
        os.replace(destination/'final.partial.mp4',mp4);sidecars(mp4,metadata)
        report={'format':'h3-independent-join-v1','frames':total,'fps':FPS,'seconds':total/FPS,
                'size':[w,h],'overlap_frames':overlap_frames,'cut':cut,'validation':validation,
                'source_audio':str(source) if source else None,'source_start_seconds':source_start_seconds,
                'clips':[{'path':str(p),**r} for p,r in zip(paths,rows)],
                'master':str(master),'video':str(mp4), 'seamless_guarantee':False,
                'note':'Hard cuts only. No interpolation, crossfade, retiming or skin/colour correction.'}
        atomic_json(destination/'join.json',report)
        return report
    except BaseException:
        (destination/'INCOMPLETE.txt').write_text('Join did not finish. Source videos were not modified. Inspect logs before deleting this partial export.\n')
        raise

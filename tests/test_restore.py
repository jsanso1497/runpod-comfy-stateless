"""CPU tests. The fake AI test verifies plumbing, not SeedVR2 inference quality."""
import base64
from fractions import Fraction
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'video_restore'))
import engine
import server


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp('media') / 'test.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi',
        '-i', 'testsrc2=size=1920x1080:rate=30000/1001:duration=0.4',
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.4',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
        '-color_range', 'tv', '-c:a', 'aac', '-shortest', str(path)], check=True)
    return path


def test_manifest_has_verified_entries():
    info = json.loads((ROOT / 'config/video_restore/models.json').read_text())
    assert set(info['models']) == {'7b', '3b', 'vae'}
    assert all(len(x['sha256']) == 64 for x in info['models'].values())


def test_frame_rate_and_metadata(clip):
    meta = engine.validate_source(clip)
    assert meta['fps'] == '30000/1001'
    assert meta['audio']
    engine.verify_cadence(clip, 0, 0.25, Fraction(meta['fps']))


def test_hdr_is_rejected(monkeypatch, clip):
    data = engine.probe(clip)
    data['streams'][0]['color_transfer'] = 'smpte2084'
    monkeypatch.setattr(engine, 'probe', lambda p: data)
    with pytest.raises(ValueError, match='HDR'):
        engine.validate_source(clip)


def test_vfr_metadata_rejected(monkeypatch, clip):
    data = engine.probe(clip)
    data['streams'][0]['avg_frame_rate'] = '27000/1001'
    monkeypatch.setattr(engine, 'probe', lambda p: data)
    with pytest.raises(ValueError, match='variable-frame-rate'):
        engine.validate_source(clip)


def test_profiles_and_cli():
    hardware = {'vram_gib': 80}
    profile = engine.select_profile(hardware, '7b')
    assert profile['batch'] == 9 and profile['overlap'] < profile['batch']
    weights = {'directory': '/models', 'dit': {'filename': 'seedvr2_ema_7b_fp16.safetensors'}}
    cmd = engine.seed_command(Path('/in.mkv'), Path('/out'), weights, hardware, 'compare_4k', profile, 'sdpa')
    assert cmd[cmd.index('--resolution') + 1] == '2160'
    assert cmd[cmd.index('--output_format') + 1] == 'png'
    assert '--compile_dit' not in cmd
    assert cmd[cmd.index('--input_noise_scale') + 1] == '0'


def test_cleanup_pipeline_real_ffmpeg(clip, tmp_path):
    report = engine.run_test(clip, tmp_path / 'cleanup',
        {'mode': 'cleanup_only', 'duration': 0.2, 'start': 0, 'shadow': 'none'},
        {}, {'vram_gib': 80}, progress=lambda s: None)
    assert report['status'] == 'complete'
    assert report['frames'] == 6
    for check in report['checks'].values():
        assert check['frames'] == 6 and check['fps'] == '30000/1001'
        assert check['sample_aspect_ratio'] == '1:1' and check['audio_present']


def test_fake_ai_4k_plumbing_only(monkeypatch, clip, tmp_path):
    fake = tmp_path / 'fake_ai.py'
    fake.write_text('''import subprocess,sys\nfrom pathlib import Path\nsource,dest,width,height=sys.argv[1:]\nout=Path(dest)/"input"\nout.mkdir(parents=True)\nsubprocess.run(["ffmpeg","-v","error","-y","-i",source,"-vf",f"scale={width}:{height}:flags=bilinear","-start_number","0",str(out/"input_%06d.png")],check=True)\n''')
    monkeypatch.setattr(engine, 'SEED', tmp_path)
    monkeypatch.setattr(engine, 'seed_command', lambda source, dest, weights, hardware, mode, profile, attention:
        [sys.executable, str(fake), str(source), str(dest), '3840', '2160'])
    report = engine.run_test(clip, tmp_path / 'ai',
        {'mode': 'compare_4k', 'duration': 0.2, 'start': 0, 'shadow': 'gentle'},
        {}, {'vram_gib': 80, 'attention': 'sdpa'}, progress=lambda s: None)
    assert report['status'] == 'complete'
    assert report['checks']['04_AI_4K.mp4']['width'] == 3840
    assert report['checks']['04_AI_4K.mp4']['height'] == 2160
    assert report['checks']['03_AI_1080p_from_4K.mp4']['width'] == 1920
    assert report['checks']['05_Original_vs_AI_3840x1080.mp4']['width'] == 3840
    assert (tmp_path / 'ai' / '06_Detail_comparison.png').exists()
    assert not (tmp_path / 'ai' / 'raw_seedvr2').exists()


def test_cancel_prevents_start(tmp_path):
    cancel = threading.Event(); cancel.set()
    runner = engine.Runner(tmp_path, cancel, lambda s: None)
    with pytest.raises(engine.Cancelled):
        runner.run([sys.executable, '-c', 'print(1)'])


def test_http_auth_csrf_and_private_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, 'HOME', tmp_path)
    monkeypatch.setenv('RESTORE_PASSWORD', 'test-password-only-1234')
    instance = server.make_server('127.0.0.1', 0, initialize=False)
    thread = threading.Thread(target=instance.serve_forever, daemon=True); thread.start()
    try:
        auth = 'Basic ' + base64.b64encode(b'james:test-password-only-1234').decode()
        conn = http.client.HTTPConnection('127.0.0.1', instance.server_port, timeout=10)
        conn.request('GET', '/api/state'); response = conn.getresponse()
        assert response.status == 401; response.read()
        conn.request('GET', '/healthz'); response = conn.getresponse()
        assert response.status == 200; response.read()
        conn.request('GET', '/api/state', headers={'Authorization': auth}); response = conn.getresponse()
        data = json.loads(response.read()); assert 'csrf' in data
        conn.request('POST', '/api/run', '{}', {'Authorization': auth, 'Content-Type': 'application/json'})
        response = conn.getresponse(); assert response.status == 403; response.read(); conn.close()
        conn = http.client.HTTPConnection('127.0.0.1', instance.server_port, timeout=10)
        conn.request('GET', '/files/../../etc/passwd', headers={'Authorization': auth})
        response = conn.getresponse(); assert response.status == 404; response.read(); conn.close()
    finally:
        instance.shutdown(); instance.server_close()
        instance.state.executor.shutdown(wait=True)

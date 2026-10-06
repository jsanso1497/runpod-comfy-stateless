#!/usr/bin/env python3
"""Private Ollama supervision, resumable pulls, explicit ready/failure status."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import requests
from node import ollama_client as oc
from node.logic import settings

ROOT = Path('/workspace/h3-portrait')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def present(session, model):
    return any(oc.canonical(r.get('name') or r.get('model', '')) == oc.canonical(model)
               for r in oc.api(session, '/api/tags').get('models', []))


def pull_model(session, model, *, total_timeout=7200, attempts=3, stall_seconds=180,
               emit=print, status=lambda **kw: None, clock=time.monotonic, sleep=time.sleep):
    """Retry into Ollama's existing partial store; never delete partial weights.

    The deadline covers all attempts, not one full budget for every retry.
    A visible model tag, rather than an empty `ollama list` during a pull,
    determines completion. An empty list does not prove that seeding never ran.
    """
    if present(session, model):
        emit('H3 OLLAMA: cached model verified in /api/tags: ' + model, flush=True)
        return
    deadline = clock() + total_timeout
    last_error = None
    for attempt in range(1, attempts + 1):
        if clock() >= deadline:
            break
        status(phase='pulling', model=model, attempt=attempt)
        emit(f'H3 OLLAMA PULL {attempt}/{attempts}: {model} (partial downloads are reused)', flush=True)
        last_progress = clock()
        last_print = -1e20
        previous = None
        success = False
        try:
            with session.post(oc.BASE + '/api/pull', json={'model': model, 'stream': True},
                              stream=True, timeout=(5, min(stall_seconds, max(1, deadline-clock()))),
                              allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RuntimeError(f'Local Ollama pull returned HTTP {response.status_code}.')
                for line in response.iter_lines():
                    now = clock()
                    if now >= deadline:
                        raise TimeoutError('Model pull exceeded its total time budget.')
                    if not line:
                        if now-last_progress > stall_seconds:
                            raise TimeoutError('Model download stopped making progress.')
                        continue
                    msg = json.loads(line)
                    if msg.get('error'):
                        # Daemon errors contain model/download details, not credentials from this client.
                        raise RuntimeError('Ollama pull: ' + str(msg['error'])[:500])
                    signature = (msg.get('status'), msg.get('digest'), msg.get('completed'))
                    if signature != previous:
                        previous, last_progress = signature, now
                    elif now-last_progress > stall_seconds:
                        raise TimeoutError('Model download stalled; retrying the same partial download.')
                    completed, total = msg.get('completed', 0), msg.get('total', 0)
                    if now-last_print >= 10 or msg.get('status') == 'success':
                        detail = f'{completed/1e9:.2f}/{total/1e9:.2f} GB' if total else ''
                        emit(f'H3 OLLAMA: {model} | {msg.get("status", "downloading")} {detail}', flush=True)
                        status(phase='pulling', model=model, attempt=attempt,
                               progress=msg.get('status'), completed=completed, total=total)
                        last_print = now
                    if msg.get('status') == 'success':
                        success = True
                        break
            if success and present(session, model):
                emit('H3 OLLAMA MODEL INSTALLED: ' + model, flush=True)
                return
            raise RuntimeError('Pull ended without a registered complete model; partial data was retained.')
        except (requests.RequestException, ValueError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            emit(f'H3 OLLAMA RETRY: {type(exc).__name__}: {exc}', flush=True)
            if attempt < attempts and clock() < deadline:
                sleep(min(5 * attempt, max(0, deadline-clock())))
    raise RuntimeError(f'Could not finish {model}. Partial download kept. '
                       f'Inspect {ROOT}/ollama-status.json and retry the same model. Last error: {last_error}')


def daemon_env(cfg, environ=None):
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items()
           if not k.endswith(('_TOKEN', '_KEY', '_PASSWORD', '_SECRET'))
           and k.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'OLLAMA_ORIGINS')}
    store = Path(environ.get('OLLAMA_MODELS') or ROOT/'ollama-models')
    if not store.is_absolute() or str(store) == '/':
        raise ValueError('OLLAMA_MODELS must be an absolute model-cache directory, not /.')
    env.update(OLLAMA_HOST='127.0.0.1:11434', OLLAMA_MODELS=str(store), OLLAMA_NO_CLOUD='1',
               OLLAMA_KEEP_ALIVE='0', OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
               OLLAMA_CONTEXT_LENGTH=str(cfg['context_length']))
    return env


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = settings()
    child = None
    ready = ROOT/'ollama-ready.json'
    ready.unlink(missing_ok=True)
    def status(**values):
        atomic_json(ROOT/'ollama-status.json', dict(pid=os.getpid(), version='1.5.3', updated=time.time(), **values))
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status(phase='starting')
    try:
        env = daemon_env(cfg)
        Path(env['OLLAMA_MODELS']).mkdir(parents=True, exist_ok=True)
        with oc.session() as session, (ROOT/'ollama.log').open('a', buffering=1) as log:
            try:
                oc.api(session, '/api/version', timeout=2)
            except Exception:
                pass
            else:
                raise RuntimeError('Port 11434 is occupied. This template must own its private Ollama daemon.')
            child = subprocess.Popen(['/usr/bin/ollama', 'serve'], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+90
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Ollama exited; inspect ollama.log.')
                try:
                    oc.api(session, '/api/version', timeout=3)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Ollama did not start.')
                    time.sleep(1)
            models = list(dict.fromkeys((cfg['analysis_model'], cfg['ollama_model'])))
            for model in models:
                pull_model(session, model, total_timeout=cfg['pull_timeout_seconds'], status=status)
            info = oc.pipeline_info(session, cfg)
            oc.clear_owned_residency(session, info, cfg['unload_timeout_seconds'])
            ready_info = {'version': '1.5.3', 'pid': os.getpid(), 'models': models, 'pipeline': info}
            atomic_json(ready, ready_info)
            status(phase='ready', models=models, model_cache=env['OLLAMA_MODELS'])
            print('H3 PORTRAIT OLLAMA READY: analysis='+cfg['analysis_model']+' | director='+cfg['ollama_model']+
                  ' | profile='+cfg.get('profile', 'full')+'; all required model tags verified', flush=True)
            child.wait()
            raise RuntimeError('Ollama stopped.')
    except KeyboardInterrupt:
        status(phase='stopped')
    except Exception as exc:
        status(phase='failed', error=str(exc)[:1200])
        raise
    finally:
        ready.unlink(missing_ok=True)
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError): os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError): os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()

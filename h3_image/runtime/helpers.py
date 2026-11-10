"""Small process and secret-handling helpers, independent of the old H3 Lab."""
import os
import signal
import subprocess


def validate_password(password):
    if not isinstance(password, str) or not 16 <= len(password.encode('utf-8')) <= 72 or '\n' in password or '\r' in password:
        raise ValueError('COMFY_PASSWORD must contain 16-72 UTF-8 bytes, without newlines.')
    if 'RUNPOD_SECRET_' in password:
        raise ValueError('COMFY_PASSWORD is an unresolved RunPod secret reference.')


def terminal_environment():
    result = dict(os.environ)
    for name in list(result):
        upper = name.upper()
        if upper.endswith(('_TOKEN', '_PASSWORD', '_SECRET', '_API_KEY')) or upper.startswith('RUNPOD_SECRET_'):
            result.pop(name, None)
    result['TERM'] = 'xterm-256color'
    return result


def stop_process(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=20)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

#!/usr/bin/env python3
"""Own one loopback Ollama daemon. Download the vision model, not H3 weights."""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ollama_h3_client", HERE / "local_nodes/ComfyUI-OllamaH3/client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def write_status(root, state, **extra):
    data = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    tmp = root / "status.json.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, root / "status.json")
    print(f"[OllamaH3] {state}: {extra.get('message', extra.get('model', ''))}", flush=True)


def local_env(root):
    env = os.environ.copy()
    for key in list(env):
        if key.endswith(("_TOKEN", "_PASSWORD", "_SECRET", "_API_KEY")) or key in ("HTTP_PROXY", "ALL_PROXY", "OLLAMA_ORIGINS"):
            env.pop(key, None)
    env.update(OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MODELS=str(root / "models"),
               OLLAMA_NO_CLOUD="1", OLLAMA_KEEP_ALIVE="0", OLLAMA_NUM_PARALLEL="1",
               OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_MAX_QUEUE="2")
    return env


def stop(child):
    if child is not None and child.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def ready(s, child, timeout=90):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if child.poll() is not None:
            raise RuntimeError("Ollama daemon exited before readiness; read /workspace/ollama/service.log")
        try:
            return client.api(s, "/api/version", timeout=3)
        except Exception:
            time.sleep(1)
    raise RuntimeError("Timed out waiting for local Ollama")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "check-install"])
    args = parser.parse_args()
    cfg = client.settings()
    test = args.command == "check-install"
    root = Path(tempfile.mkdtemp(prefix="ollama-smoke-")) if test else Path("/workspace/ollama")
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    if not test and not cfg["enabled"]:
        write_status(root, "disabled", message="Set ENABLE_OLLAMA=1 to enable; reviewed prompts still work.")
        return
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Do not accidentally adopt an unrelated local server and unload its models.
        with client.session() as s:
            try:
                client.api(s, "/api/version", timeout=2)
            except Exception:
                pass
            else:
                raise RuntimeError("Port 11434 is already occupied. Stop the other Ollama service first.")
            env = local_env(root)
            with (root / "service.log").open("a", buffering=1) as log:
                child = subprocess.Popen(["/usr/bin/ollama", "serve"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                version = ready(s, child)
                if test:
                    print(f"Ollama server/API smoke test PASS: {version.get('version')} (no model downloaded)", flush=True)
                    return
                write_status(root, "downloading", model=cfg["model"], ollama_version=version.get("version"))
                # Named tags may move. Pull validates content; an optional manifest digest enforces an exact model.
                result = subprocess.run(["/usr/bin/ollama", "pull", cfg["model"]], env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=cfg["pull_timeout_seconds"])
                if result.returncode:
                    raise RuntimeError("Vision-model pull failed; inspect service.log. Existing ComfyUI workflows remain available.")
                info = client.model_info(s, cfg["model"], cfg["expected_model_digest"])
                write_status(root, "ready", **info, ollama_version=version.get("version"),
                             message="Vision model downloaded; it loads only for a prompt request.")
                print(f"[OllamaH3] Resolved model digest: {info['digest']}", flush=True)
                child.wait()
                raise RuntimeError("Ollama daemon stopped; restart its service or the Pod application before generating another prompt")
    except KeyboardInterrupt:
        if not test:
            write_status(root, "stopped")
    except Exception as exc:
        if not test:
            write_status(root, "error", message=str(exc))
        raise
    finally:
        stop(child)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[OllamaH3] SERVICE ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

"""Start the classic FileBrowser behind the existing authenticated gateway.

No public listener and no user password is passed to this service. The gateway
asserts the one verified identity. This is a single-owner Pod convenience, not a
security boundary against other processes/custom nodes in the same container.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .config import boolean, data_root

BINARY = "/usr/local/bin/filebrowser"
VERSION = "2.63.23"
HOST = "127.0.0.1"
PORT = 8081
BASE_URL = "/files"
UPSTREAM = f"http://{HOST}:{PORT}"
IDENTITY_HEADER = "X-H3-Filebrowser-User"
USERNAME = "workbench"
STATE_NAME = ".h3-filebrowser"


def enabled() -> bool:
    return boolean("H3_SWAP_FILE_BROWSER", True)


def child_environment(home: Path) -> dict[str, str]:
    # A whitelist avoids leaking HF, registry, RunPod and gateway credentials.
    result = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ}
    result.update(HOME=str(home), TMPDIR=tempfile.gettempdir())
    return result


def _run(args: list[str], state: Path, secret: str = "") -> None:
    result = subprocess.run(args, cwd=state, env=child_environment(state),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=30, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout)[-1500:]
        if secret:
            detail = detail.replace(secret, "[redacted]")
        raise RuntimeError(f"FileBrowser configuration failed ({result.returncode}): {detail}")


def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("FileBrowser state directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def configure(root: Path, binary: str = BINARY, port: int = PORT) -> tuple[list[str], dict[str, str]]:
    """Idempotently initialize a NEW private DB; never edit the old manual DB."""
    root = root.resolve(strict=True)
    if not root.is_dir() or not Path(binary).is_file():
        raise ValueError("FileBrowser binary or workspace is missing; rebuild the H3 image")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Invalid private FileBrowser port")
    state = root / STATE_NAME
    _private_dir(state)
    db, config = state / "filebrowser.db", state / "cli.json"
    for path in (db, config):
        if path.is_symlink():
            raise ValueError("FileBrowser state files must not be symlinks")
    # Explicit empty CLI config suppresses inherited config files and FB_* vars.
    config.write_text("{}\n", encoding="utf-8")
    config.chmod(0o600)
    settings = ["--address", HOST, "--port", str(port), "--baseURL", BASE_URL,
                "--root", str(root), "--auth.method=proxy", "--auth.header", IDENTITY_HEADER,
                "--disableExec=true", "--followExternalSymlinks=false", "--signup=false",
                "--disableThumbnails=true", "--disablePreviewResize=true",
                "--disableImageResolutionCalc=true", "--branding.disableExternal=true",
                "--log", "stdout"]
    permissions = ["--scope", ".", "--perm.admin=false", "--perm.execute=false",
                   "--perm.share=false", "--perm.create=true", "--perm.modify=true",
                   "--perm.rename=true", "--perm.delete=true", "--perm.download=true",
                   "--hideDotfiles=true", "--lockPassword=true"]
    common = [binary, "--config", str(config)]
    if not db.exists():
        # An interrupted initialization never publishes a half-created DB.
        with tempfile.TemporaryDirectory(prefix="init-", dir=state) as tmp:
            staged = Path(tmp) / "filebrowser.db"
            command = common + ["--database", str(staged)]
            _run(command + ["config", "init"] + settings, state)
            placeholder = secrets.token_urlsafe(48)
            _run(command + ["users", "add", USERNAME, placeholder] + permissions, state, placeholder)
            _run(command + ["rules", "add", r"(^|/)\.h3-filebrowser(/|$)", "--regex"], state)
            staged.chmod(0o600)
            os.replace(staged, db)
    command = common + ["--database", str(db)]
    _run(command + ["config", "set"] + settings, state)
    _run(command + ["users", "update", USERNAME] + permissions, state)
    db.chmod(0o600)
    return (command + ["--address", HOST, "--port", str(port), "--baseURL", BASE_URL,
                       "--root", str(root), "--disableExec=true"], child_environment(state))


async def supervise(stop: asyncio.Event, status: dict, logdir: Path) -> None:
    """Start at Pod boot; limited retries do not bring down a working H3 service."""
    entry = status.setdefault("filebrowser", {})
    entry.update(enabled=enabled(), ready=False, path=BASE_URL + "/", version=VERSION)
    if not entry["enabled"]:
        entry["stage"] = "disabled"
        return
    failures = 0
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "filebrowser.log"
    while not stop.is_set() and failures < 3:
        proc = None
        try:
            entry.update(stage="starting", ready=False)
            args, env = await asyncio.to_thread(configure, data_root())
            with log_path.open("a", buffering=1) as log:
                proc = await asyncio.create_subprocess_exec(*args, cwd=str(data_root() / STATE_NAME),
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                from aiohttp import ClientSession, ClientTimeout
                async with ClientSession(timeout=ClientTimeout(total=2), trust_env=False) as session:
                    for _ in range(30):
                        if stop.is_set():
                            return
                        if proc.returncode is not None:
                            raise RuntimeError("FileBrowser exited during startup; inspect filebrowser.log")
                        try:
                            async with session.get(UPSTREAM + BASE_URL + "/") as response:
                                if response.status == 200:
                                    break
                        except (OSError, asyncio.TimeoutError):
                            pass
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError("FileBrowser local health check timed out; inspect filebrowser.log")
                entry.update(stage="ready", ready=True)
                entry.pop("error", None)
                print("[H3 FILES READY] Open /files/ on the authenticated HTTP 8188 service", flush=True)
                while not stop.is_set():
                    if proc.returncode is not None:
                        raise RuntimeError("FileBrowser exited; inspect filebrowser.log")
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        pass
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failures += 1
            entry.update(stage="error", ready=False, error=str(exc), failed_starts=failures)
            print("[H3 FILES ERROR] " + str(exc), flush=True)
            if failures < 3:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5 * failures)
                except asyncio.TimeoutError:
                    pass
        finally:
            entry["ready"] = False
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()


def smoke(binary: str = BINARY) -> None:
    """Real binary/HTTP/config round trip, run in Docker build without a GPU."""
    with tempfile.TemporaryDirectory(prefix="h3-filebrowser-smoke-") as tmp:
        root = Path(tmp)
        (root / "sample.txt").write_text("H3 FileBrowser smoke test\n")
        # Choose a free ephemeral port for this build-only test.
        with socket.socket() as listener:
            listener.bind((HOST, 0))
            port = listener.getsockname()[1]
        args, env = configure(root, binary, port)
        # Test restart configuration too, before opening Bolt DB in the service.
        args, env = configure(root, binary, port)
        base = f"http://{HOST}:{port}{BASE_URL}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(args, cwd=root / STATE_NAME, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError("Real FileBrowser exited in smoke test")
                    try:
                        with opener.open(base + "/", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Real FileBrowser did not become ready")
                request = urllib.request.Request(base + "/api/login", data=b"", method="POST",
                                                 headers={IDENTITY_HEADER: USERNAME})
                with opener.open(request, timeout=3) as response:
                    token = response.read().decode().strip().strip('"')
                if token.count(".") != 2:
                    raise RuntimeError("FileBrowser did not return its expected session token")
                with opener.open(urllib.request.Request(base + "/api/resources/", headers={"X-Auth": token}), timeout=3) as response:
                    result = json.loads(response.read())
                    names = [item["name"] for item in result["items"]]
                    assert "sample.txt" in names and STATE_NAME not in names, names
                try:
                    opener.open(urllib.request.Request(base + "/api/resources/" + STATE_NAME + "/", headers={"X-Auth": token}), timeout=3)
                except urllib.error.HTTPError as exc:
                    assert exc.code in (403, 404), exc.code
                else:
                    raise RuntimeError("Internal FileBrowser database directory is not protected")
                print("FileBrowser real-binary smoke passed: restart, proxy login, listing and state protection")
            except Exception:
                log.seek(0)
                print(log.read().decode(errors="replace")[-4000:])
                raise
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if not options.smoke:
        parser.error("Use --smoke; normal startup is managed by swapkit.runtime")
    smoke()

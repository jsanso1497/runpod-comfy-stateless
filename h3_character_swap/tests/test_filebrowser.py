"""Offline tests; FileBrowser subprocess is mocked, gateway HTTP is real loopback.

Docker runs the separate real-binary smoke test after download. These tests do
not claim an installed FileBrowser, a Docker run, a browser session, or a GPU.
"""
import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from multidict import CIMultiDict

from swapkit import filebrowser as fb, gateway

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "this-is-a-test-password-only"
AUTH = "Basic " + base64.b64encode(("workbench:" + PASSWORD).encode()).decode()


def mock_cli(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_bytes(b"not executable in offline tests")
    calls = []
    def run(args, state, secret=""):
        calls.append(list(args))
        if "init" in args:
            Path(args[args.index("--database") + 1]).write_bytes(b"mock-db")
    monkeypatch.setattr(fb, "_run", run)
    return str(binary), calls


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("H3_SWAP_FILE_BROWSER", raising=False)
    assert fb.enabled()


def test_disable(monkeypatch):
    monkeypatch.setenv("H3_SWAP_FILE_BROWSER", "0")
    assert not fb.enabled()


def test_invalid_enable_flag(monkeypatch):
    monkeypatch.setenv("H3_SWAP_FILE_BROWSER", "yes")
    with pytest.raises(ValueError):
        fb.enabled()


def test_only_private_listener():
    assert fb.HOST == "127.0.0.1" and fb.PORT == 8081
    assert fb.BASE_URL == "/files"


def test_child_has_no_secrets_or_untrusted_fb_overrides(monkeypatch, tmp_path):
    for key in ("WB_PASSWORD", "HF_TOKEN", "RUNPOD_API_KEY", "FB_ADDRESS", "FB_AUTH_METHOD", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "not-for-filebrowser")
    env = fb.child_environment(tmp_path)
    assert not any(v == "not-for-filebrowser" for v in env.values())
    assert env["HOME"] == str(tmp_path)


def test_initialization_idempotent_and_permissions_reapplied(tmp_path, monkeypatch):
    binary, calls = mock_cli(tmp_path, monkeypatch)
    args, env = fb.configure(tmp_path, binary)
    db = tmp_path / fb.STATE_NAME / "filebrowser.db"
    assert db.read_bytes() == b"mock-db"
    assert db.stat().st_mode & 0o777 == 0o600
    assert (db.parent.stat().st_mode & 0o777) == 0o700
    init_count = len([a for a in calls if "init" in a])
    fb.configure(tmp_path, binary)
    assert len([a for a in calls if "init" in a]) == init_count == 1
    assert args[args.index("--address") + 1] == "127.0.0.1"
    assert "--disableExec=true" in args
    assert env == fb.child_environment(db.parent)
    for a in calls:
        if "users" in a:
            assert "--perm.admin=false" in a
            assert "--perm.share=false" in a
            assert "--perm.execute=false" in a
    assert len([a for a in calls if "rules" in a]) == 1
    assert len([a for a in calls if "update" in a]) == 2


def test_failed_init_does_not_publish_partial_db(tmp_path, monkeypatch):
    binary, calls = mock_cli(tmp_path, monkeypatch)
    real = fb._run
    def fail(args, state, secret=""):
        real(args, state, secret)
        if "add" in args and "users" in args:
            raise RuntimeError("simulated setup failure")
    monkeypatch.setattr(fb, "_run", fail)
    with pytest.raises(RuntimeError):
        fb.configure(tmp_path, binary)
    assert not (tmp_path / fb.STATE_NAME / "filebrowser.db").exists()


def test_manual_database_is_not_modified(tmp_path, monkeypatch):
    legacy = tmp_path / ".filebrowser" / "filebrowser.db"
    legacy.parent.mkdir()
    legacy.write_bytes(b"leave this alone")
    binary, calls = mock_cli(tmp_path, monkeypatch)
    fb.configure(tmp_path, binary)
    assert legacy.read_bytes() == b"leave this alone"
    assert all(str(legacy) not in a for a in calls)


@pytest.mark.parametrize("name", ["filebrowser.db", "cli.json"])
def test_symlinked_state_file_rejected(tmp_path, monkeypatch, name):
    binary, calls = mock_cli(tmp_path, monkeypatch)
    state = tmp_path / fb.STATE_NAME
    state.mkdir()
    (state / name).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        fb.configure(tmp_path, binary)
    assert not calls


def test_symlinked_state_directory_rejected(tmp_path, monkeypatch):
    binary, calls = mock_cli(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / fb.STATE_NAME).symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        fb.configure(tmp_path, binary)
    assert not calls


@pytest.mark.parametrize("port", [80, 0, -1, 65536, "8081", True])
def test_invalid_port_rejected(tmp_path, monkeypatch, port):
    binary, calls = mock_cli(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        fb.configure(tmp_path, binary, port)


def test_header_filter_strips_spoofed_identity():
    headers = CIMultiDict({fb.IDENTITY_HEADER: "admin", "Authorization": AUTH, "Range": "bytes=0-9"})
    result = gateway.filtered(headers, True)
    assert fb.IDENTITY_HEADER not in result
    assert "Authorization" not in result
    assert result["Range"] == "bytes=0-9"


def test_real_gateway_http_auth_route_and_streaming(monkeypatch):
    monkeypatch.setenv("H3_SWAP_FILE_BROWSER", "1")
    async def scenario():
        records = []
        async def backend(request):
            records.append((request.raw_path, dict(request.headers), await request.read()))
            return web.Response(status=206, body=b"test bytes", headers={"Content-Range": "bytes 0-9/10"})
        upstream = web.Application()
        upstream.router.add_route("*", "/{tail:.*}", backend)
        async with TestServer(upstream) as server:
            monkeypatch.setattr(fb, "UPSTREAM", str(server.make_url("/"))[:-1])
            app = gateway.make_app(PASSWORD, {"ready": True, "filebrowser": {"ready": True}})
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/files/")
                assert response.status == 401 and not records
                response = await client.get("/files", headers={"Authorization": AUTH}, allow_redirects=False)
                assert response.status == 307 and response.headers["Location"] == "/files/"
                response = await client.get("/files/api/raw/test%20clip.mp4?x=1", headers={
                    "Authorization": AUTH, fb.IDENTITY_HEADER: "administrator", "Range": "bytes=0-9"})
                assert response.status == 206 and await response.read() == b"test bytes"
                assert response.headers["Content-Range"] == "bytes 0-9/10"
                path, headers, data = records[-1]
                assert path == "/files/api/raw/test%20clip.mp4?x=1"
                assert headers[fb.IDENTITY_HEADER] == "workbench"
                assert "Authorization" not in headers
                assert headers["Range"] == "bytes=0-9"
                before = len(records)
                response = await client.post("/files/api/resources/test", data=b"blocked", headers={
                    "Authorization": AUTH, "Origin": "https://attacker.example", fb.IDENTITY_HEADER: "workbench"})
                assert response.status == 403 and len(records) == before
                response = await client.post("/files/api/resources/test", data=b"upload", headers={"Authorization": AUTH})
                assert response.status == 206 and records[-1][2] == b"upload"
    asyncio.run(scenario())


@pytest.mark.parametrize("enabled,ready,expected", [("0",True,404),("1",False,503)])
def test_gateway_disabled_or_starting(monkeypatch, enabled, ready, expected):
    monkeypatch.setenv("H3_SWAP_FILE_BROWSER", enabled)
    async def scenario():
        app = gateway.make_app(PASSWORD, {"ready": True, "filebrowser": {"ready": ready}})
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/files/", headers={"Authorization": AUTH})
            assert response.status == expected
    asyncio.run(scenario())


def test_supervisor_disabled_does_not_spawn(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_SWAP_FILE_BROWSER", "0")
    spawn = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    async def scenario():
        status = {}
        await fb.supervise(asyncio.Event(), status, tmp_path)
        assert status["filebrowser"]["stage"] == "disabled"
    asyncio.run(scenario())
    spawn.assert_not_called()


def test_binary_pinned_and_smoke_retained():
    docker = (ROOT / "Dockerfile").read_text()
    assert "/v2.63.23/linux-amd64-filebrowser.tar.gz" in docker
    assert "b14db2bb8033caa3f80205eb6578b2ed0744ebd9e716b790bc4a9703ce909e88" in docker
    assert "python -m swapkit.filebrowser --smoke" in docker
    assert "EXPOSE 8188" in docker and "EXPOSE 808" not in docker
    assert "python -m swapkit.check --static" in docker


def test_runtime_manages_service_without_changing_comfy_ports():
    runtime = (ROOT / "swapkit/runtime.py").read_text()
    assert "asyncio.create_task(supervise_filebrowser(stop,status,logdir))" in runtime
    assert 'await asyncio.gather(files_task, return_exceptions=True)' in runtime
    assert '"--listen","127.0.0.1","--port","8189"' in runtime
    assert '"OLLAMA_CONTEXT_LENGTH":"32768"' in runtime

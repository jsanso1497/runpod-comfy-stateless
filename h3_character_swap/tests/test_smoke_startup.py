"""Regression checks for CI smoke startup, without weights or a GPU."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from swapkit import check


def install_fake_child(monkeypatch, *, exit_code=None, startup_log=""):
    process = Mock()
    process.poll.return_value = exit_code
    process.wait.return_value = 0
    launches = []

    def launch(args, **kwargs):
        user_dir = Path(args[args.index("--user-directory") + 1])
        # ComfyUI validates this argument before importing its native nodes.
        assert user_dir.is_dir(), "--user-directory must exist BEFORE Popen"
        assert user_dir.parent.name.startswith("h3-swap-smoke-")
        assert user_dir.name == "user"
        (user_dir / "write-probe").write_text("writable", encoding="utf-8")
        assert "--cpu" in args
        assert args[args.index("--listen") + 1] == "127.0.0.1"
        assert "--disable-all-custom-nodes" not in args
        assert "--quick-test-for-ci" not in args
        assert kwargs["cwd"] == "/opt/ComfyUI"
        if startup_log:
            kwargs["stdout"].write(startup_log)
            kwargs["stdout"].flush()
        launches.append(user_dir)
        return process

    monkeypatch.setattr(check.subprocess, "Popen", launch)
    return process, launches


def schema_success():
    return {
        "ok": True, "missing_nodes": [], "problems": [],
        "gpu_tested": False, "scope": "test-only schema",
    }


def test_smoke_creates_user_directory_before_launch(monkeypatch, capsys):
    process, launches = install_fake_child(monkeypatch)
    report = Mock(return_value=schema_success())
    monkeypatch.setattr(check, "live_report", report)
    check.smoke()
    report.assert_called_once_with(
        {"akatz", "bfs"}, "http://127.0.0.1:8289/object_info", False,
    )
    assert len(launches) == 1
    assert not launches[0].parent.exists()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=20)
    assert '"gpu_tested": false' in capsys.readouterr().out


def test_smoke_still_rejects_missing_nodes(monkeypatch, capsys):
    process, launches = install_fake_child(monkeypatch, startup_log="node import failed\n")
    monkeypatch.setattr(check, "live_report", lambda *args: {
        "ok": False, "missing_nodes": ["H3RelayWindowPlan"], "problems": [],
    })
    with pytest.raises(ValueError, match="H3RelayWindowPlan"):
        check.smoke()
    assert "node import failed" in capsys.readouterr().out
    assert not launches[0].parent.exists()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=20)


def test_smoke_still_fails_on_early_process_exit(monkeypatch, capsys):
    process, launches = install_fake_child(
        monkeypatch, exit_code=2, startup_log="main.py: error: example startup error\n",
    )
    report = Mock()
    monkeypatch.setattr(check, "live_report", report)
    with pytest.raises(ValueError, match="CPU ComfyUI startup failed"):
        check.smoke()
    report.assert_not_called()
    assert "example startup error" in capsys.readouterr().out
    assert not launches[0].parent.exists()
    process.terminate.assert_called_once()


def test_smoke_retries_until_schema_is_available(monkeypatch):
    process, launches = install_fake_child(monkeypatch)
    report = Mock(side_effect=[OSError("not listening yet"), schema_success()])
    sleep = Mock()
    monkeypatch.setattr(check, "live_report", report)
    monkeypatch.setattr(check.time, "sleep", sleep)
    check.smoke()
    assert report.call_count == 2
    sleep.assert_called_once_with(1)
    assert not launches[0].parent.exists()
    process.terminate.assert_called_once()


def test_smoke_uses_a_fresh_temporary_directory_for_each_run(monkeypatch):
    _, launches = install_fake_child(monkeypatch)
    monkeypatch.setattr(check, "live_report", lambda *args: schema_success())
    check.smoke()
    check.smoke()
    assert len(launches) == 2
    assert launches[0] != launches[1]
    assert all(not p.parent.exists() for p in launches)

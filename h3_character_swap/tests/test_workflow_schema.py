"""Regression coverage for the actual 2026-10-22 null-definitions failure.

These tests exercise the real graph/schema reader. Mocked HTTP and child
processes are explicitly test fixtures, not a live ComfyUI or GPU benchmark.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from swapkit import check, graphs


def canvas(kind="LoadVideo", **extra):
    return {"nodes": [{"id": 1, "type": kind, "inputs": [], "outputs": []}],
            "links": [], **extra}


@pytest.mark.parametrize("extra", [
    {}, {"definitions": None}, {"definitions": {}},
    {"definitions": {"subgraphs": None}},
    {"definitions": {"subgraphs": []}},
])
def test_absent_null_and_empty_optional_fields(extra):
    graph = canvas(**extra)
    original = copy.deepcopy(graph)
    assert check.graph_nodes(graph) == {"LoadVideo"}
    assert graph == original


@pytest.mark.parametrize("value", [[], "invalid", 0, False])
def test_malformed_non_null_definitions_are_not_silently_ignored(value):
    with pytest.raises(ValueError, match="definitions must be an object or null"):
        check.graph_nodes(canvas(definitions=value))


@pytest.mark.parametrize("value", [{}, "invalid", 0, False])
def test_malformed_non_null_subgraphs_are_rejected(value):
    with pytest.raises(ValueError, match="subgraphs must be a list or null"):
        check.graph_nodes(canvas(definitions={"subgraphs": value}))


@pytest.mark.parametrize("graph", [
    None, {}, {"nodes": None}, {"nodes": {}},
    {"nodes": [None]}, {"nodes": [{"type": None}]},
    {"nodes": [{"type": ""}]},
])
def test_malformed_graph_or_node_is_rejected(graph):
    with pytest.raises(ValueError):
        check.graph_nodes(graph)


def test_nested_subgraphs_with_null_definitions_keep_backend_types():
    leaf = {"id": "leaf", **canvas("VAEDecode", definitions=None)}
    middle = {"id": "middle", **canvas("leaf", definitions={"subgraphs": [leaf]})}
    graph = canvas("middle", definitions={"subgraphs": [middle]})
    assert check.graph_nodes(graph) == {"VAEDecode"}


def test_sibling_subgraph_references_are_not_missing_backend_nodes():
    a = {"id": "subgraph-a", **canvas("subgraph-b", definitions=None)}
    b = {"id": "subgraph-b", **canvas("LoadImage", definitions=None)}
    graph = canvas("subgraph-a", definitions={"subgraphs": [a, b]})
    assert check.graph_nodes(graph) == {"LoadImage"}


def test_frontend_nodes_are_ignored_but_unknown_backend_nodes_are_not():
    kinds = ["Note", "MarkdownNote", "Reroute", "PrimitiveNode", "NotInstalled"]
    graph = {"nodes": [{"id": i, "type": k} for i, k in enumerate(kinds)], "definitions": None}
    assert check.graph_nodes(graph) == {"NotInstalled"}


@pytest.mark.parametrize("subgraph", [None, {}, {"id": None}, {"id": ""}, {"id": 5}])
def test_malformed_subgraph_definition_is_rejected(subgraph):
    with pytest.raises(ValueError, match="subgraph id"):
        check.graph_nodes(canvas(definitions={"subgraphs": [subgraph]}))


def write_workflows(root):
    folder = root / "workflows"
    folder.mkdir()
    prompt, _ = graphs.prompt_studio()
    bfs, _ = graphs.bfs_graph()
    prompt["definitions"] = None
    bfs["definitions"] = {"subgraphs": None}
    exports = {
        "00_Analyze_Clip_Build_Prompts.json": prompt,
        "01_Akatz_Mask_Check.json": canvas("SAM3_Detect", definitions=None),
        "02_Akatz_Native_Replacement.json": canvas("H3RelayWindowPlan", definitions=None),
        "03_BFS_Guide_Comparison.json": bfs,
    }
    for name, graph in exports.items():
        (folder / name).write_text(json.dumps(graph))
    # API-format files must not be mistaken for editor canvases.
    (folder / "02_Akatz_Native_Replacement.api.json").write_text("{}")
    return exports


def backend_schema(exports):
    # Construct expected types independently from the production traverser.
    return {n["type"]: {"input": {"required": {}}}
            for g in exports.values() for n in g["nodes"]}


def test_schema_inspection_accepts_null_without_skipping_nodes(tmp_path, monkeypatch):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    schema = backend_schema(exports)
    report = check.inspect_schema(schema, {"akatz", "bfs"}, False)
    assert report["ok"] and not report["missing_nodes"]
    del schema["H3RelayWindowPlan"]
    report = check.inspect_schema(schema, {"akatz", "bfs"}, False)
    assert report["ok"] is False
    assert report["missing_nodes"] == ["H3RelayWindowPlan"]


def test_variant_filter_still_works(tmp_path, monkeypatch):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    schema = backend_schema(exports)
    del schema["H3SwapBFSPrompt"]
    assert check.inspect_schema(schema, {"akatz"}, False)["ok"]
    assert "H3SwapBFSPrompt" in check.inspect_schema(schema, {"akatz", "bfs"}, False)["missing_nodes"]


def test_static_build_check_exercises_same_traverser(tmp_path, monkeypatch):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    (tmp_path / "assets.lock.json").write_text(json.dumps({"assets": [
        {"sha256": "a" * 64, "revision": "b" * 40, "bytes": 1}
    ]}))
    seen = []
    original = check.graph_nodes
    def recording(graph):
        seen.append(graph)
        return original(graph)
    monkeypatch.setattr(check, "graph_nodes", recording)
    report = check.static_report()
    assert report["ok"] and report["canvases"] == 4
    assert len(seen) == 4
    bad = exports["00_Analyze_Clip_Build_Prompts.json"]
    bad["definitions"] = []
    (tmp_path / "workflows/00_Analyze_Clip_Build_Prompts.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="definitions must be an object or null"):
        check.static_report()


def test_model_visibility_still_checked(tmp_path, monkeypatch):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    (tmp_path / "assets.lock.json").write_text(json.dumps({"assets": [
        {"group": "common", "destination": "diffusion_models/model.safetensors"}
    ]}))
    schema = backend_schema(exports)
    schema["UNETLoader"]["input"]["required"]["unet_name"] = [[]]
    report = check.inspect_schema(schema, {"akatz", "bfs"}, True)
    assert report["ok"] is False
    assert report["problems"] == ["Model not visible: diffusion_models/model.safetensors"]
    schema["UNETLoader"]["input"]["required"]["unet_name"] = [["model.safetensors"]]
    assert check.inspect_schema(schema, {"akatz", "bfs"}, True)["ok"]


def test_smoke_reaches_real_schema_reader_with_null_definitions(tmp_path, monkeypatch, capsys):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    schema = backend_schema(exports)
    child = Mock()
    child.poll.return_value = None
    child.wait.return_value = 0
    def launch(args, **kwargs):
        assert Path(args[args.index("--user-directory") + 1]).is_dir()
        assert "--cpu" in args
        return child
    monkeypatch.setattr(check.subprocess, "Popen", launch)
    def urlopen(url, timeout):
        assert url == "http://127.0.0.1:8289/object_info"
        return io.BytesIO(json.dumps(schema).encode())
    monkeypatch.setattr(check.urllib.request, "urlopen", urlopen)
    # Unlike the original startup regression, live_report is NOT mocked.
    check.smoke()
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["gpu_tested"] is False
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=20)


def test_smoke_still_fails_for_real_missing_schema_type(tmp_path, monkeypatch):
    exports = write_workflows(tmp_path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    schema = backend_schema(exports)
    del schema["H3RelayWindowPlan"]
    child = Mock()
    child.poll.return_value = None
    child.wait.return_value = 0
    monkeypatch.setattr(check.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(check.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(schema).encode()))
    with pytest.raises(ValueError, match="H3RelayWindowPlan"):
        check.smoke()
    child.terminate.assert_called_once()

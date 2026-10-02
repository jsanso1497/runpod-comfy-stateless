#!/usr/bin/env python3
"""Small RunPod compatibility wrapper for ComfyUI's origin middleware.

This keeps ComfyUI's own request guard. It does NOT enable wildcard CORS,
remove authentication, or disable Manager's security checks.

Commands: apply COMFY_HOME | check --port 8188 | self-test
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

VERSION = "1.1.0"
HEADER = "X-ComfyUI-HTTP-Fix"
BEGIN = "# BEGIN RUNPOD_COMFY_HTTP_FIX"
END = "# END RUNPOD_COMFY_HTTP_FIX"
INJECTION = '''# BEGIN RUNPOD_COMFY_HTTP_FIX
# Keep upstream protection; repair safe UI navigation and this Pod's origin.
_runpod_original_origin_factory = create_origin_only_middleware

def create_origin_only_middleware():
    from runpod_http_support import wrap_origin_middleware
    return wrap_origin_middleware(_runpod_original_origin_factory())
# END RUNPOD_COMFY_HTTP_FIX


'''


def canonical_origin(value: str) -> str | None:
    """Accept an origin, not an arbitrary URL or untrusted forwarded header."""
    if not value or any(c.isspace() for c in value):
        return None
    try:
        p = urlsplit(value)
        if (p.scheme not in ("http", "https") or not p.hostname
                or p.username is not None or p.password is not None
                or p.path not in ("", "/") or p.query or p.fragment):
            return None
        port = p.port
    except ValueError:
        return None
    host = p.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    default = 443 if p.scheme == "https" else 80
    if port is not None and port != default:
        host += ":" + str(port)
    return p.scheme + "://" + host


def public_origin() -> str | None:
    explicit = os.environ.get("COMFY_PUBLIC_ORIGIN", "").strip()
    if explicit:
        origin = canonical_origin(explicit)
        if origin is None:
            raise ValueError("COMFY_PUBLIC_ORIGIN must be one http(s) origin, without a path or credentials.")
        return origin
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    port = os.environ.get("COMFY_PORT", "8188").strip()
    if pod_id:
        if not re.fullmatch(r"[a-z0-9-]+", pod_id) or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("Invalid RUNPOD_POD_ID or COMFY_PORT.")
        return f"https://{pod_id}-{int(port)}.proxy.runpod.net"
    return None


def normalize_headers(method: str, path: str, headers, allowed_origin: str | None) -> dict[str, str]:
    """Normalize only the public UI landing page and an exact trusted origin.

    The public index page is safe to navigate to from another website. All API,
    upload, websocket and state-changing requests retain the upstream checks.
    Exact-origin normalization also handles proxies that change Host to a local
    address. X-Forwarded-Host is deliberately NOT trusted.
    """
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    origin = canonical_origin(h.get("origin", ""))
    if allowed_origin and origin == allowed_origin:
        h["host"] = urlsplit(allowed_origin).netloc
        h["sec-fetch-site"] = "same-origin"

    # The HTML shell is public. This exception does not extend to /view,
    # /userdata, /system_stats, /prompt, /upload or websocket requests.
    safe_landing = (method.upper() in ("GET", "HEAD")
                    and path in ("/", "/index.html")
                    and h.get("upgrade", "").lower() != "websocket")
    if safe_landing:
        h.pop("origin", None)
        if h.get("sec-fetch-site", "").lower() == "cross-site":
            h.pop("sec-fetch-site", None)
    return h


def wrap_origin_middleware(upstream):
    from aiohttp import web
    allowed_origin = public_origin()

    @web.middleware
    async def runpod_origin_middleware(request, handler):
        original = {k.lower(): v for k, v in request.headers.items()}
        normalized = normalize_headers(request.method, request.path, request.headers, allowed_origin)
        adjusted = request if normalized == original else request.clone(headers=normalized)
        response = await upstream(adjusted, handler)
        # FileResponse is unprepared here; an accepted websocket may be prepared.
        if not response.prepared:
            response.headers[HEADER] = VERSION
        if response.status == 403:
            # Do not log Authorization, cookies, query strings or other secrets.
            logging.warning(
                "[COMFY-HTTP-403] method=%s path=%s fetch_site=%s host=%s origin=%s",
                request.method, request.path[:160],
                request.headers.get("Sec-Fetch-Site", "")[:80],
                request.headers.get("Host", "")[:160],
                request.headers.get("Origin", "")[:160],
            )
        return response

    return runpod_origin_middleware


def patch_source(source: str) -> str:
    if BEGIN in source:
        if source.count(BEGIN) != 1 or source.count(END) != 1:
            raise ValueError("Unexpected existing HTTP patch markers; refusing to change server.py.")
        start = source.index(BEGIN)
        finish = source.index(END, start) + len(END)
        source = source[:start] + source[finish:].lstrip("\r\n")
    tree = ast.parse(source)
    factories = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "create_origin_only_middleware"]
    servers = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PromptServer"]
    if len(factories) != 1 or len(servers) != 1:
        raise ValueError("ComfyUI's HTTP layout changed. Review server.py before applying this patch.")
    if factories[0].lineno >= servers[0].lineno:
        raise ValueError("Unexpected middleware order in server.py.")
    lines = source.splitlines(keepends=True)
    index = servers[0].lineno - 1
    result = "".join(lines[:index]) + INJECTION + "".join(lines[index:])
    compile(result, "server.py", "exec")
    return result


def apply(comfy_home: Path) -> None:
    target = comfy_home / "server.py"
    if not target.is_file():
        raise FileNotFoundError(f"ComfyUI server.py was not found: {target}")
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    backup = comfy_home / "server.py.runpod-http-backup"
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    support = comfy_home / "runpod_http_support.py"
    if support.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, support)
    temp = target.with_suffix(".py.runpod-new")
    temp.write_text(patched, encoding="utf-8")
    temp.replace(target)
    print(f"HTTP compatibility patch {VERSION}: installed; API-origin checks retained.", flush=True)
    origin = public_origin()
    print(f"Expected public origin: {origin or 'not set; public landing-page fix still active'}", flush=True)


def check(port: int, timeout: int) -> int:
    """Check the real local server, not just whether the process exists.

    These are local-only probes. No request, token or content is sent to a
    third-party site, and no inference or state-changing request is submitted.
    """
    root = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))

    def fetch(path: str, headers=None):
        req = Request(root + path, headers={"User-Agent": "RunPod-ComfyUI-Readiness/" + VERSION,
                                           **(headers or {})})
        try:
            with opener.open(req, timeout=30) as response:
                return response.status, dict(response.headers), response.read(32 * 1024 * 1024)
        except HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read(4096)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _, body = fetch("/system_stats")
            if status == 200 and isinstance(json.loads(body), dict):
                break
        except (OSError, URLError, ValueError):
            pass
        time.sleep(3)
    else:
        print("[HTTP CHECK FAILED] ComfyUI did not become ready. Inspect the preceding startup error.", flush=True)
        return 1

    origin = public_origin()
    host = urlsplit(origin).netloc if origin else "127.0.0.1:" + str(port)
    checks = [
        ("local UI", "/", {}, 200),
        ("cross-site UI navigation", "/", {
            "Host": host, "Origin": "https://console.runpod.io",
            "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document"}, 200),
        ("cross-site API remains blocked", "/system_stats", {
            "Host": host, "Origin": "https://untrusted.invalid",
            "Sec-Fetch-Site": "cross-site"}, 403),
    ]
    if origin:
        checks.append(("same-origin API through rewritten Host", "/system_stats", {
            "Host": "127.0.0.1:" + str(port), "Origin": origin,
            "Sec-Fetch-Site": "same-origin"}, 200))

    ok = True
    for name, path, headers, expected in checks:
        try:
            status, response_headers, _ = fetch(path, headers)
            marker = {k.lower(): v for k, v in response_headers.items()}.get(HEADER.lower())
            passed = status == expected and marker == VERSION
            ok = ok and passed
            print(f"[HTTP CHECK {'PASS' if passed else 'FAIL'}] {name}: HTTP {status}; expected {expected}; patch={marker}", flush=True)
        except Exception as exc:
            ok = False
            print(f"[HTTP CHECK FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)

    try:
        status, _, body = fetch("/object_info")
        data = json.loads(body) if status == 200 else {}
        required = {"SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel",
                    "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"}
        missing = sorted(required.difference(data))
        if missing:
            ok = False
            print("[NODE CHECK FAIL] Missing nodes: " + ", ".join(missing), flush=True)
        else:
            print("[NODE CHECK PASS] SeedVR2 and native video nodes are registered.", flush=True)
    except Exception as exc:
        ok = False
        print(f"[NODE CHECK FAIL] {type(exc).__name__}: {exc}", flush=True)

    if ok:
        print("[READY] ComfyUI, SeedVR2 nodes and local proxy-header tests passed. GPU inference is not yet tested.", flush=True)
        if origin:
            print("Open the CURRENT Pod's HTTPS service: " + origin + "/", flush=True)
        print("If the browser still shows 403, compare response header X-ComfyUI-HTTP-Fix and [COMFY-HTTP-403] logs. Local success does not test RunPod's public proxy.", flush=True)
    else:
        print("[HTTP CHECK FAILED] Do not render yet. Keep the diagnostic lines and inspect the startup log.", flush=True)
    return 0 if ok else 1


class PolicyTests(unittest.TestCase):
    origin = "https://examplepod-8188.proxy.runpod.net"

    def test_cross_site_landing(self):
        h = normalize_headers("GET", "/", {"Origin": "https://console.runpod.io", "Sec-Fetch-Site": "cross-site"}, self.origin)
        self.assertNotIn("sec-fetch-site", h)
        self.assertNotIn("origin", h)

    def test_cross_site_api_unchanged(self):
        h = normalize_headers("GET", "/system_stats", {"Origin": "https://untrusted.invalid", "Sec-Fetch-Site": "cross-site"}, self.origin)
        self.assertEqual(h["sec-fetch-site"], "cross-site")
        self.assertEqual(h["origin"], "https://untrusted.invalid")

    def test_cross_site_writes_unchanged(self):
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            for path in ("/", "/prompt", "/upload/image"):
                with self.subTest(method=method, path=path):
                    h = normalize_headers(method, path, {"Origin": "https://untrusted.invalid", "Sec-Fetch-Site": "cross-site"}, self.origin)
                    self.assertEqual(h["sec-fetch-site"], "cross-site")
                    self.assertIn("origin", h)

    def test_exact_origin_forwarding(self):
        h = normalize_headers("POST", "/prompt", {"Host": "127.0.0.1:8188", "Origin": self.origin, "Sec-Fetch-Site": "cross-site"}, self.origin)
        self.assertEqual(h["host"], "examplepod-8188.proxy.runpod.net")
        self.assertEqual(h["sec-fetch-site"], "same-origin")

    def test_websocket_does_not_use_landing_exception(self):
        h = normalize_headers("GET", "/", {"Upgrade": "websocket", "Origin": "https://untrusted.invalid", "Sec-Fetch-Site": "cross-site"}, self.origin)
        self.assertEqual(h["sec-fetch-site"], "cross-site")

    def test_forwarded_host_not_trusted(self):
        h = normalize_headers("POST", "/prompt", {"Host": "127.0.0.1:8188", "X-Forwarded-Host": "examplepod-8188.proxy.runpod.net", "Origin": "https://untrusted.invalid"}, self.origin)
        self.assertEqual(h["host"], "127.0.0.1:8188")

    def test_no_origin_does_not_get_trusted(self):
        h = normalize_headers("GET", "/userdata", {"Sec-Fetch-Site": "cross-site"}, self.origin)
        self.assertEqual(h["sec-fetch-site"], "cross-site")

    def test_origin_parser(self):
        self.assertEqual(canonical_origin(self.origin + "/"), self.origin)
        self.assertEqual(canonical_origin("https://example.com:443"), "https://example.com")
        for bad in ("*", "null", "https://user:pass@example.com", "https://example.com/path", "https://example.com?q=x", "https://example.com#foo", "https://example.com:abc"):
            with self.subTest(origin=bad):
                self.assertIsNone(canonical_origin(bad))

    def test_patch_idempotent(self):
        source = "def create_origin_only_middleware():\n    return None\n\nclass PromptServer:\n    pass\n"
        once = patch_source(source)
        self.assertEqual(patch_source(once), once)
        self.assertEqual(once.count(BEGIN), 1)

    def test_unknown_upstream_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_source("class PromptServer:\n    pass\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("comfy_home", type=Path)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--port", type=int, default=8188)
    check_parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.command == "self-test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
        return 0 if result.wasSuccessful() else 1
    if args.command == "apply":
        apply(args.comfy_home)
        return 0
    if not 1 <= args.port <= 65535 or args.timeout < 1:
        parser.error("Invalid port or timeout.")
    return check(args.port, args.timeout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"HTTP support error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

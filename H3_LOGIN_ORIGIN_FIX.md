# H3 Studio login-origin repair

Access-layer version: 2026-11-07.3. This patch changes access handling and its tests only.
No workflows, model locks, model files, prompts, or GPU settings are changed.

## Cause and correction

The preceding gateway compared the browser's Origin directly to request.host.
That can reject a valid login when the HTTPS proxy rewrites Host to an internal
address. The same mismatch also affected protected uploads and WebSockets.
The exact user-side proxy headers have not been captured, but this failure was
reproduced locally with the original code and a rewritten Host.

The launcher now builds each exact public HTTPS origin from RunPod's automatic
RUNPOD_POD_ID variable: one for port 8188 and one for port 8888. The gateway uses
those server-configured origins for checks and forwards the matching public Host
to ComfyUI and Jupyter. Raw Forwarded and X-Forwarded-* headers are not trusted
as configuration. Default HTTPS port 443 is normalized.

Password checks, signed login nonces, secure HttpOnly session cookies,
cross-origin rejection, private Jupyter token authentication, and no-cache
headers remain enabled. Another Pod's origin and the other service port are not
accepted. A random UTM parameter does not bypass authentication.

## Install on the Mac

Save update_h3_login_origin_fix.py in Downloads. In GitHub Desktop select
runpod-comfy-stateless, then Repository > Open in Terminal:

```bash
python3 "$HOME/Downloads/update_h3_login_origin_fix.py"
```

Python 3.8 or newer is sufficient for the installer. It refuses unknown local
edits, backs up changed files under Git's private metadata, and does not stage,
commit, push, build, deploy, or access RunPod. --dry-run previews changes.

Review and commit in GitHub Desktop, then manually Push origin. Start a NEW
Build H3 Still Developer Workspace run for the updated commit. Use the new
successful build's image digest in RunPod. An old image or an old build rerun
does not acquire this repair.

Keep ports 8188,8888 and COMFY_PASSWORD unchanged. Keep /workspace and its
model cache, inputs, outputs, private adapters and saved workflows intact.
No new secret or normal RunPod configuration is needed.

Open the active Pod's HTTPS connection in its own tab. Use the fresh path:

```text
/__h3/login-v3
```

The sign-in page should show Studio 2026-11-07.3. Enter the same comfy_password;
there is no username. Old /__h3/login links still work, and submit to the new
endpoint automatically. The fresh path avoids cached responses at the old URL;
the container cannot purge an existing provider or browser cache.

## Nonstandard proxy addresses only

RunPod normally supplies RUNPOD_POD_ID automatically. For a custom domain or a
nonstandard proxy, set these optional variables to each exact external HTTPS
origin (scheme + host + optional port, without a path or query):

```text
H3_COMFY_PUBLIC_ORIGIN=https://YOUR-COMFY-HOST
H3_FILES_PUBLIC_ORIGIN=https://YOUR-FILES-HOST
```

Do not paste those placeholders literally. Do not set a wildcard, turn off XSRF,
remove the password, or expose internal ports 8189/8889. A stopped/edited Pod
must retain /workspace. RUNPOD_POD_HOSTNAME is not the public browser origin.

The sign-in error and authenticated Connection diagnostics now show the expected
public service origin. Only the public target is logged, not passwords,
cookies, tokens, full Referer URLs or URL query strings.

## Validation

See the packaged validation results for actually executed tests. New checks cover
rewritten Host, TLS termination, default ports, both services, denied cross-site
requests, password and nonce failures, chunked uploads, and WebSocket forwarding.
The full browser smoke now places a real HTTPS-to-HTTP Host-rewriting proxy in
front of the gateway, instead of testing direct TLS access only.

The local full Chromium run was blocked by the environment's administrator
navigation policy (ERR_BLOCKED_BY_ADMINISTRATOR). Local protocol tests passed
against real Jupyter, including a real terminal command over secure WebSockets.
The GitHub build must still pass the full Chromium/Jupyter UI test in its image.
No live RunPod, new Docker build, or GPU inference test was performed here.

## References

- https://docs.runpod.io/pods/templates/environment-variables
- https://docs.runpod.io/integrations/overview
- https://docs.aiohttp.org/en/stable/web_advanced.html#deploying-behind-a-proxy
- https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html

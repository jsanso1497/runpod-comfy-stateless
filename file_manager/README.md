# Browser file access on HTTP 8888

The complete repository installs JupyterLab 4.6.4 and Jupyter Server 2.21.1 in a
separate /opt/file-manager-venv. This does not replace ComfyUI's Python libraries.
Both Docker entrypoints supervise the browser and their existing application.

Set FILEBROWSER_PASSWORD from a RunPod secret (12 or more characters) and expose
HTTP ports 8188,8888. FILEBROWSER_PORT defaults to 8888. No default login or
passwordless mode is provided. Use the RunPod HTTPS proxy; leave TCP ports blank.

Open HTTP Service 8888 and enter the actual password at the Jupyter login screen.
The initial file-tree root is /workspace. JupyterLab provides uploads, downloads,
text editing and a server-side terminal. It is not a security sandbox; access is
intended for the trusted container owner. This does not protect ComfyUI port 8188.

Password hashes/config and runtime logs are in /run/runpod-file-manager outside
the browsable root, with private file permissions. Secrets are not printed or put
in CLI arguments; unrelated API keys are removed from Jupyter's child environment.
Password validation occurs before any inference-model downloads. The service
starts first, so files can be accessed while models are preparing. If either
critical service exits, the supervisor terminates the other service's process
group. Inference process exit behavior otherwise remains unchanged.

The Docker build executes the same smoke_test.py against the pinned Jupyter stack.
It checks real password authentication, denial without login, CSRF protection,
and upload/download/edit/delete through Jupyter's HTTP API in a temporary directory.
These test files do not access a user's /workspace.

Sources verified for this integration:
- https://jupyter-server.readthedocs.io/en/latest/operators/security.html
- https://jupyter-server.readthedocs.io/en/latest/api/jupyter_server.auth.html
- https://pypi.org/project/jupyterlab/4.6.4/
- https://pypi.org/project/jupyter-server/2.21.1/
- https://docs.runpod.io/pods/configuration/expose-ports
- https://docs.runpod.io/pods/templates/secrets

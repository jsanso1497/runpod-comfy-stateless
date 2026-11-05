"""Token-protected Jupyter behind RunPod HTTPS. This file contains no secret values."""
import os
from pathlib import Path

c = get_config()  # noqa: F821; provided by Jupyter's config loader
# Remove the token before notebook kernels and terminals inherit the environment.
token = os.environ.pop('JUPYTER_TOKEN', '')
if len(token) < 16 or not token.strip() or token.lstrip().startswith('{{'):
    raise ValueError('Refusing to start Jupyter without a resolved token of at least 16 characters.')
if any(ch in token for ch in ('\n', '\r', '\x00')):
    raise ValueError('Invalid multiline Jupyter token.')
c.IdentityProvider.token = token
del token
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.port_retries = 0
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.root_dir = str(Path(os.environ.get('WB_DATA_ROOT', '/workspace')).resolve())
c.ServerApp.default_url = '/lab'
c.ServerApp.allow_remote_access = True
c.ServerApp.trust_xheaders = True
c.ServerApp.allow_origin = ''
c.ServerApp.disable_check_xsrf = False
c.ServerApp.allow_unauthenticated_access = False
c.ServerApp.cookie_options = {'secure': True, 'httponly': True, 'samesite': 'Lax'}
c.ServerApp.log_level = 30
c.LabApp.open_browser = False

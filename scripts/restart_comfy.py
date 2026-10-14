#!/usr/bin/env python3
"""Ask the running Workbench supervisor to restart its ComfyUI child safely."""
import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    headers = {'Content-Type': 'application/json'}
    mode = os.getenv('WB_AUTH_MODE', 'basic')
    if mode == 'basic':
        password = os.getenv('WB_PASSWORD', '')
        if len(password) < 16 or password.startswith('{{'):
            raise ValueError('WB_PASSWORD is not resolved in this shell.')
        headers['Authorization'] = 'Basic ' + base64.b64encode(('workbench:' + password).encode()).decode()
    elif mode != 'none':
        raise ValueError('WB_AUTH_MODE must be basic or none.')
    req = Request('http://127.0.0.1:8188/_workbench/restart-comfy', method='POST', data=b'{}', headers=headers)
    try:
        with urlopen(req, timeout=40) as response:
            print(json.loads(response.read(65536)).get('state', 'Restart requested'))
    except HTTPError as error:
        try:
            message = json.loads(error.read(65536)).get('error', 'Request rejected.')
        except (ValueError, UnicodeError):
            message = 'Request rejected. Check the rebuilt Workbench image and authentication.'
        raise ValueError(message) from None
    except URLError:
        raise ValueError('Workbench is unavailable on port 8188. Restart the Pod/container, not a second ComfyUI server.') from None


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        print('ERROR: ' + str(error))
        raise SystemExit(1)

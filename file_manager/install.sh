#!/usr/bin/env bash
set -Eeuo pipefail
# Separate environment prevents Jupyter dependency resolution from replacing Torch,
# transformers, aiohttp or any of the pinned ComfyUI inference dependencies.
python -m venv --without-pip /opt/file-manager-venv
env -u PIP_CONSTRAINT -u PYTHONPATH -u PYTHONHOME \
  python -m pip --python /opt/file-manager-venv/bin/python install --no-cache-dir \
  -r /opt/file-manager/requirements.txt
env -u PIP_CONSTRAINT -u PYTHONPATH -u PYTHONHOME \
  python -m pip --python /opt/file-manager-venv/bin/python check
/opt/file-manager-venv/bin/python -m unittest discover -s /opt/file-manager/tests -p 'test_*.py'
/opt/file-manager-venv/bin/python /opt/file-manager/smoke_test.py

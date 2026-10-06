"""Importable by repository and /opt installers without installing a Python wheel."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_model_safety_policy', Path(__file__).parent / 'node/policy.py')
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
protect_graph = _policy.protect_graph
native_view = _policy.native_view

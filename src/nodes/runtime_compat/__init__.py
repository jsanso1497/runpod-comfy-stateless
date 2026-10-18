"""Load the Workbench CUDA compatibility shim before any model execution."""
from workbench.cuda_compat import install_compat
REPORT = install_compat()
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

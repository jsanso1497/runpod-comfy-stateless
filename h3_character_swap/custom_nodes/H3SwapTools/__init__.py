from swapkit.nodes import NODE_CLASS_MAPPINGS as BASE_NODES, NODE_DISPLAY_NAME_MAPPINGS as BASE_NAMES
from swapkit.disk_nodes import NODE_CLASS_MAPPINGS as DISK_NODES, NODE_DISPLAY_NAME_MAPPINGS as DISK_NAMES
from swapkit.precision import profile, ECONOMY
if profile() == ECONOMY:
    ANALYSIS_NODES, ANALYSIS_NAMES = {}, {}
    BASE_NODES = {k:v for k,v in BASE_NODES.items() if 'Analyze' not in k and 'BFSPrompt' not in k}
    BASE_NAMES = {k:v for k,v in BASE_NAMES.items() if k in BASE_NODES}
else:
    from swapkit.analysis_nodes import NODE_CLASS_MAPPINGS as ANALYSIS_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANALYSIS_NAMES
from swapkit.preview_nodes import NODE_CLASS_MAPPINGS as PREVIEW_NODES, NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NAMES
from swapkit.independent_nodes import NODE_CLASS_MAPPINGS as INDEPENDENT_NODES, NODE_DISPLAY_NAME_MAPPINGS as INDEPENDENT_NAMES
from swapkit.economy_nodes import NODE_CLASS_MAPPINGS as ECON_NODES, NODE_DISPLAY_NAME_MAPPINGS as ECON_NAMES
NODE_CLASS_MAPPINGS = {**ECON_NODES, **BASE_NODES, **DISK_NODES, **ANALYSIS_NODES, **PREVIEW_NODES, **INDEPENDENT_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**ECON_NAMES, **BASE_NAMES, **DISK_NAMES, **ANALYSIS_NAMES, **PREVIEW_NAMES, **INDEPENDENT_NAMES}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from swapkit.recovery import register_journal
register_journal()

from swapkit.job_history import register as register_job_history
register_job_history()

"""H3 Studio Lab: additive nodes; legacy Studio implementations remain untouched."""
from .nodes import NODE_CLASS_MAPPINGS
from .experiments import EXPERIMENT_NODES
NODE_CLASS_MAPPINGS.update(EXPERIMENT_NODES)
NODE_DISPLAY_NAME_MAPPINGS = {name: name.replace('H3Lab', 'H3 Lab - ') for name in NODE_CLASS_MAPPINGS}
WEB_DIRECTORY = './web'

# Extra model roots are registered on every start without changing bootstrap's YAML.
import folder_paths
from .common import ROOT
for _kind in ('checkpoints', 'refmods'):
    folder_paths.add_model_folder_path(_kind, str(ROOT / 'models' / _kind), is_default=True)

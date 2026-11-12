"""H3 Video Still Studio: one graph, family-correct routing, no prompt LLM."""
from .nodes import NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {name: name.replace('H3V', 'H3 Video Studio - ') for name in NODE_CLASS_MAPPINGS}
WEB_DIRECTORY = './web'

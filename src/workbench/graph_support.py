"""Node requirements for normal workflows and current frontend subgraphs."""
from __future__ import annotations
VIRTUAL = {'Note', 'MarkdownNote', 'Reroute', 'PrimitiveNode'}


def containers(value):
    if isinstance(value, dict):
        if isinstance(value.get('nodes'), list):
            yield value
        for child in value.values():
            yield from containers(child)
    elif isinstance(value, list):
        for child in value:
            yield from containers(child)


def required_node_types(graph):
    parts = list(containers(graph))
    definitions = {str(p['id']) for p in parts[1:] if 'id' in p}
    return {n['type'] for p in parts for n in p['nodes']} - definitions - VIRTUAL

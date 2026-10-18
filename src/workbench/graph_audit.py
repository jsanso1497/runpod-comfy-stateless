"""Recursive link and DAG validation, including native ComfyUI subgraphs."""
from graphlib import TopologicalSorter, CycleError


def audit_graph(graph, label='graph'):
    errors = []
    def visit(g, name, inner=False):
        nodes = {n['id']: n for n in g.get('nodes', [])}
        if len(nodes) != len(g.get('nodes', [])):
            errors.append(name+': duplicate node IDs')
        pseudo = set()
        if inner:
            pseudo = {g.get('inputNode', {}).get('id'), g.get('outputNode', {}).get('id')} - {None}
        rows = []
        for row in g.get('links', []):
            if isinstance(row, dict):
                try:
                    row = [row[k] for k in ('id','origin_id','origin_slot','target_id','target_slot','type')]
                except KeyError:
                    errors.append(name+': malformed subgraph link'); continue
            if not isinstance(row, list) or len(row) != 6:
                errors.append(name+': malformed link'); continue
            rows.append(row)
        links = {r[0]: r for r in rows}
        if len(links) != len(rows):
            errors.append(name+': duplicate link IDs')
        deps = {n: set() for n in nodes}
        for ident, src, out, dst, inp, kind in rows:
            if src not in nodes and src not in pseudo or dst not in nodes and dst not in pseudo:
                errors.append(f'{name}: dangling endpoint on {ident}'); continue
            try:
                if type(out) is not int or type(inp) is not int or min(out, inp) < 0:
                    raise ValueError()
                if src in nodes and ident not in (nodes[src]['outputs'][out].get('links') or []):
                    raise ValueError()
                if dst in nodes and nodes[dst]['inputs'][inp].get('link') != ident:
                    raise ValueError()
                if inner and src == g.get('inputNode', {}).get('id') and out >= len(g.get('inputs', [])):
                    raise ValueError()
                if inner and dst == g.get('outputNode', {}).get('id') and inp >= len(g.get('outputs', [])):
                    raise ValueError()
            except (KeyError, IndexError, ValueError, TypeError):
                errors.append(f'{name}: invalid socket or reciprocal link {ident}')
            if src in nodes and dst in nodes:
                deps[dst].add(src)
        for n in nodes.values():
            for p in n.get('inputs', []):
                if p.get('link') is not None and p['link'] not in links:
                    errors.append(f"{name}: dangling input at {n['id']}")
            for p in n.get('outputs', []):
                if any(i not in links for i in p.get('links') or []):
                    errors.append(f"{name}: dangling output at {n['id']}")
        try:
            tuple(TopologicalSorter(deps).static_order())
        except CycleError:
            errors.append(name+': execution cycle')
        for sg in g.get('definitions', {}).get('subgraphs', []):
            visit(sg, name+'/'+str(sg.get('name', sg.get('id', 'subgraph'))), True)
    visit(graph, label)
    return errors

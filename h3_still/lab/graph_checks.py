"""Structural graph checks, optionally compared with real installed /object_info."""

def validate_graph(graph,info=None):
    errors=[];nodes={n['id']:n for n in graph['nodes']};links={e[0]:e for e in graph['links']}
    if len(nodes)!=len(graph['nodes']):errors.append('Duplicate node IDs')
    if len(links)!=len(graph['links']):errors.append('Duplicate edge IDs')
    for ident,s,slot,t,index,kind in graph['links']:
        if s not in nodes or t not in nodes:errors.append('Missing endpoint: '+str(ident));continue
        a,b=nodes[s],nodes[t]
        if slot>=len(a.get('outputs',[])) or index>=len(b.get('inputs',[])):errors.append('Missing port '+str(ident));continue
        out,inp=a['outputs'][slot],b['inputs'][index]
        if ident not in (out.get('links') or []):errors.append('Nonreciprocal output '+str(ident))
        if inp.get('link')!=ident:errors.append('Nonreciprocal input '+str(ident))
        if out['type']!=kind or inp['type']!=kind:errors.append('Port type mismatch '+str(ident))
    for n in nodes.values():
        for inp in n.get('inputs',[]):
            if inp.get('link') is not None and inp['link'] not in links:errors.append('Dangling input '+str(n['id']))
        for port in n.get('outputs',[]):
            if any(e not in links for e in (port.get('links') or [])):errors.append('Dangling output '+str(n['id']))
        if info is None or n['type'] in ('Note','MarkdownNote'):continue
        schema=info.get(n['type'])
        if not schema:errors.append('Missing registered node: '+n['type']);continue
        fields={**schema.get('input',{}).get('required',{}),**schema.get('input',{}).get('optional',{})}
        connected={p['name'] for p in n.get('inputs',[]) if p.get('link') is not None}
        widget=n.get('widgets_values_named',{})
        for key in connected:
            if key not in fields:errors.append(n['type']+' has no input '+key)
        # These generated graphs provide named widgets; legacy fixture graphs can rely on positional values.
        if widget:
            for key in schema.get('input',{}).get('required',{}):
                if key not in connected and key not in widget:errors.append(n['type']+' missing required value '+key)
        for p in n.get('inputs',[]):
            if p['name'] in fields:
                typ=fields[p['name']][0]
                typ='COMBO' if isinstance(typ,list) else typ
                if isinstance(typ,str) and typ!=p['type'] and typ!='*':errors.append(n['type']+'.'+p['name']+' expects '+typ+' not '+p['type'])
        outputs=schema.get('output',[])
        for i,p in enumerate(n.get('outputs',[])):
            if i>=len(outputs):errors.append(n['type']+' output index missing: '+str(i))
            elif isinstance(outputs[i],str) and outputs[i]!=p['type']:errors.append(n['type']+' output type mismatch')
    # Detect cycles independently of saved LiteGraph order.
    remaining=set(nodes);done=set()
    while remaining:
        ready={n for n in remaining if all(e[1] in done for e in links.values() if e[3]==n)}
        if not ready:errors.append('Cyclic graph');break
        remaining-=ready;done|=ready
    return errors

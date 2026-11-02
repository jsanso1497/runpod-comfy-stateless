"""Create matching baseline/BF16 factories from the pinned native Akatz graph."""
from __future__ import annotations
import copy
import uuid
from .graphs import Graph, set_widget, validate_graph
from .recovery import SIMPLE_PROMPT


def relink(canvas, api):
    """Rebuild reciprocal links from the executable API graph after bounded edits."""
    by_id={str(n['id']):n for n in canvas['nodes']}
    for node in canvas['nodes']:
        for item in node.get('inputs',[]):item['link']=None
        for item in node.get('outputs',[]):item['links']=[]
    links=[]
    for dest_id,recipe in api.items():
        dest=by_id[str(dest_id)]
        for field,value in recipe['inputs'].items():
            if not (isinstance(value,list) and len(value)==2 and str(value[0]) in by_id and type(value[1]) is int):continue
            src=by_id[str(value[0])];slot=value[1]
            if not 0<=slot<len(src.get('outputs',[])):raise ValueError('Invalid native output index')
            matches=[i for i,p in enumerate(dest.get('inputs',[])) if p['name']==field]
            if matches:idx=matches[0]
            else:
                idx=len(dest.setdefault('inputs',[]))
                port={'name':field,'type':src['outputs'][slot]['type'],'link':None}
                if field in dest.get('widgets_values_named',{}):port['widget']={'name':field}
                dest['inputs'].append(port)
            lid=len(links)+1;kind=src['outputs'][slot]['type']
            dest['inputs'][idx]['link']=lid;src['outputs'][slot]['links'].append(lid)
            links.append([lid,src['id'],slot,dest['id'],idx,kind])
    canvas['links']=links;canvas['last_link_id']=len(links)
    canvas['last_node_id']=max([0]+[n['id'] for n in canvas['nodes'] if type(n['id']) is int])
    validate_graph(canvas)
    return canvas,api


def low_resolution(original,api):
    canvas,recipe=copy.deepcopy(original),copy.deepcopy(api)
    for node in canvas['nodes']:
        if node['type']=='H3RelayPrepareSource':set_widget(node,'resolution','480')
    for node in recipe.values():
        if node['class_type']=='H3RelayPrepareSource':node['inputs']['resolution']='480'
    canvas['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,str(canvas['id'])+'/480-preview'))
    canvas.setdefault('extra',{})['h3_480_note']='480 is the lower pixel-budget preset, not a different weight precision. Sampling and models are unchanged within this image.'
    return canvas,recipe


def independent_native(original,original_api):
    from .preview_graph import native_preview
    canvas,api=native_preview(original,original_api)
    nodes={str(n['id']):n for n in canvas['nodes']}
    for name in ('start','end','video'):
        api.pop(name,None)
    canvas['nodes']=[n for n in canvas['nodes'] if str(n['id']) not in ('start','end','video')
                     and n['type'] not in ('Note','MarkdownNote')]
    source=nodes['source'];source.update(type='H3SwapIndependentSource',
        title='1. SELECT ONE CLIP FROM WORKFLOW 11 (no generated history)',
        inputs=[],widgets_values=['input/h3_independent/shot48_independent/clip_0001.mkv'],
        widgets_values_named={'source_clip':'input/h3_independent/shot48_independent/clip_0001.mkv'})
    source['properties']['Node name for S&R']='H3SwapIndependentSource'
    api['source']={'class_type':'H3SwapIndependentSource','inputs':dict(source['widgets_values_named'])}
    set_widget(nodes['prepare'],'duration_seconds',0.)
    api['prepare']['inputs']['duration_seconds']=0.
    for key,value in [('history_frames','0'),('schedule','segment')]:
        set_widget(nodes['plan'],key,value);api['plan']['inputs'][key]=value
    for name in ('window','append'):
        api[name]['inputs'].pop('previous',None)
        api[name]['inputs'].pop('initial_history',None)
        nodes[name]['inputs']=[p for p in nodes[name]['inputs'] if p['name'] not in ('previous','initial_history')]
    api['window']['inputs']['iteration_index']=0
    set_widget(nodes['window'],'iteration_index',0)
    api['finish']['inputs']['state']=['append',0]
    for name,node in nodes.items():
        if node['type']=='H3SwapWindowPreview':
            api[name]['inputs']['iteration_index']=0
            node['inputs']=[p for p in node['inputs'] if p['name']!='iteration_index']
            set_widget(node,'iteration_index',0)
    save=nodes['save']
    save.update(type='H3SwapSaveIndependentChunk',title='3. SAVE: lossless master, preview and reopenable workflow',
                inputs=[{'name':'images','type':'IMAGE','link':None},
                        {'name':'source_video','type':'VIDEO','link':None}],
                outputs=[{'name':'preview_video','type':'VIDEO','links':[]},
                         {'name':'join_filename','type':'STRING','links':[]}],
                widgets_values_named={'take_name':'shot48_take1','crf':16},widgets_values=['shot48_take1',16],
                size=[600,500])
    save['properties']['Node name for S&R']='H3SwapSaveIndependentChunk'
    api['save']={'class_type':'H3SwapSaveIndependentChunk','inputs':{
        'images':['finish',0],'source_video':['source',0],'take_name':'shot48_take1','crf':16}}
    n=900002
    if str(n) in nodes:raise ValueError('Independent graph node id conflict')
    canvas['nodes'].append({'id':n,'type':'PreviewAny','title':'COPY THIS MASTER FILENAME INTO JOIN WORKFLOW 13',
        'pos':[save['pos'][0],save['pos'][1]+550],'size':[650,300],'order':len(canvas['nodes']),
        'mode':0,'flags':{},'properties':{'Node name for S&R':'PreviewAny'},
        'inputs':[{'name':'source','type':'*','link':None}],
        'outputs':[{'name':'output','type':'STRING','links':[]}], 'widgets_values':[], 'widgets_values_named':{}})
    api[str(n)]={'class_type':'PreviewAny','inputs':{'source':['save',1]}}
    note=('One clip, one independent native generation. Use the SAME original reference, seed and SAM settings for all clips. '
          'No StartLoop, previous frames, second character reference or prompt enhancer. '
          'The default source is a 124-frame clip from workflow 11. Short final tails are padded only for H3 then trimmed back. '
          'Change source_clip for the next section; do not feed generated outputs forward. '
          'Review the decoded pane, then copy each chosen master filename into workflow 13 in chronological order. '
          'Keep 12-frame overlap in the join only when prepared with 12-frame overlap. Lower-memory pixels: choose 480 on Prepare Source.')
    canvas['nodes'].append({'id':900003,'type':'Note','title':'INDEPENDENT CLIP WORKFLOW',
        'pos':[source['pos'][0],source['pos'][1]-450],'size':[1000,330],'order':0,'mode':0,'flags':{},
        'properties':{'Node name for S&R':'Note'},'inputs':[],'outputs':[],
        'widgets_values':[note],'widgets_values_named':{'text':note}})
    canvas['groups']=[];canvas['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,'H3 independent native 2026-11-02'))
    canvas.setdefault('extra',{})['h3_independent']={'history_frames':0,'max_source_frames':124,'no_identity_guarantee':True}
    canvas['extra']['ds']={'scale':.6,'offset':[50,500]}
    return relink(canvas,api)


def prepare_graph():
    g=Graph('H3 independent preparation 2026-11-02')
    a=g.add('H3SwapPrepareIndependentClips','1. PREPARE SOURCE CLIPS: no GPU generation',0,0,
        outputs=[('clip_list_and_instructions','STRING')],
        widgets={'source_video':'input/source.mp4','set_name':'shot48_independent',
                 'duration_seconds':48.,'start_seconds':0.,'window_frames':'124','overlap_frames':12},size=(600,470))
    p=g.add('PreviewAny','2. CLIP LIST / SETTINGS TO KEEP FOR JOINING',700,0,inputs=[('source','*')],
            outputs=[('output','STRING')],size=(850,600));g.connect(a,0,p,'source','STRING')
    return g.result()


def join_graph():
    g=Graph('H3 independent join 2026-11-02')
    a=g.add('H3SwapJoinIndependentClips','1. JOIN CHOSEN MASTERS: hard cuts, no regeneration',0,0,
        outputs=[('joined_video','VIDEO'),('report','STRING')],widgets={
            'ordered_videos':'','join_name':'shot48_join01','overlap_frames':12,'cut_placement':'Middle of overlap',
            'audio_mode':'Original source','original_source':'input/source.mp4','source_start_seconds':0.,'crf':16},size=(760,850))
    p=g.add('PreviewAny','2. FRAME COUNTS / OUTPUT PATHS / VALIDATION',850,0,inputs=[('source','*')],
            outputs=[('output','STRING')],size=(850,800));g.connect(a,1,p,'source','STRING')
    note=('Paste one output/.../master.mkv per line in timeline order. Pick one accepted take per source clip. '
          'Overlap=12 matches workflow 11 defaults; use 0 for non-overlapping clips. '
          'Middle of overlap keeps half from each take; no crossfade or morphing. '
          'Select the original FULL source for uninterrupted audio and the same original start time. '
          '24 fps, equal dimensions and valid order are required. No silent resizing, retiming, colour matching or identity correction. '
          'Saves lossless final.mkv plus final.mp4 and workflow JSONs. A correct frame count does not guarantee an invisible join.')
    g.add('Note','JOIN RULES',0,930,widgets={'text':note},size=(1600,230))
    canvas,api=g.result();api.pop('3',None);return canvas,api


def recover_graph():
    g=Graph('H3 recover video workflow 2026-11-02')
    a=g.add('H3SwapRecoverPastWorkflow','1. SELECT A PAST GENERATION',0,0,
        outputs=[('recovered_paths','STRING')],widgets={'generation_video':'output/select_a_past_video.mp4'},size=(650,250))
    p=g.add('PreviewAny','2. RECOVERED JSON PATHS OR EXPLANATION',750,0,inputs=[('source','*')],
            outputs=[('output','STRING')],size=(850,650));g.connect(a,0,p,'source','STRING')
    return g.result()

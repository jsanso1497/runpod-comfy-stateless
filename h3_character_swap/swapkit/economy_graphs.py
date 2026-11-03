"""Factory generation, keeping native samplers and per-chunk workflows intact."""
from pathlib import Path
import json
from .config import atomic_json
from .recovery import SIMPLE_PROMPT


def batch_graph():
    from .graphs import Graph
    g=Graph('H3 Economy automatic independent batch 2026-11-02.2')
    c=g.add('H3SwapEconomyBatch','1. ONE SOURCE + ONE REFERENCE: preview first, then automatic chunks',0,0,
        outputs=[('status_and_review_link','STRING')],widgets={
        'action':'Start new run','run_name':'economy_test10','source_video':'input/source.mp4',
        'character_image':'input/character.png','duration_seconds':10.,'start_seconds':0.,
        'resolution':'768','window_frames':'124','overlap_frames':12,'target_person':'person',
        'selection':'largest','selection_index':0,'mask_expansion':64,'seed':904239,
        'review_policy':'Approve first chunk','cleanup_policy':'Stage offload','prompt':SIMPLE_PROMPT},size=(680,910))
    p=g.add('PreviewAny','2. STATUS: /_swap/economy on the same 8188 service',770,0,
        inputs=[('source','*')],outputs=[('output','STRING')],size=(730,750));g.connect(c,0,p,'source','STRING')
    g.add('Note','INDEPENDENT, NOT HISTORY-CHAINED',770,800,widgets={'text':
        'Queue once. Open /_swap/economy and approve the FIRST completed clip. The rest are generated and joined automatically. '
        'Every chunk starts from ORIGINAL source and reference; no generated history is carried forward. '
        'Cuts may still be visible. First test 10 seconds, then a NEW run for 48 seconds. '
        'Right-click the ACTUAL chunk or final output in ComfyUI History > Open workflow, not this launcher status. '
        'Each chunk stores its exact native generation canvas; final output stores the assembly canvas. '
        'Stage offload keeps native stage cleanup but avoids duplicate full-cache teardown. Full cleanup is a slower fallback. '
        'Use a new run name for changed settings. Use workflow 10 for the older continuation mode.'},size=(730,330))
    canvas,api=g.result();api.pop('3',None);return canvas,api


def build_economy(root):
    from .precision import profile, ECONOMY
    root=Path(root)
    if profile(root)!=ECONOMY:return
    graph,api=batch_graph()
    graph.setdefault('extra',{})['h3_build_profile']=ECONOMY
    for suffix,value in [('.json',graph),('.api.json',api)]:
        atomic_json(root/'workflows'/('15_H3_Economy_Auto_Independent'+suffix),value)
    # No Ollama or BFS workflows are shown or registered in this dedicated image.
    for p in (root/'workflows').glob('*.json'):
        if p.name.startswith(('00_','03_','04_','09_','02C_')):p.unlink()
    record=root/'workflow-provenance.json'
    r=json.loads(record.read_text()) if record.exists() else {}
    r['economy']={'profile':ECONOMY,'llm_assistant':False,'auto_mode':'independent short native jobs',
        'conditioning':'original reference every job, no generated history','comfy_history':'exact editable job canvases',
        'gpu_validated':False,'quality_equivalence_validated':False}
    atomic_json(record,r)

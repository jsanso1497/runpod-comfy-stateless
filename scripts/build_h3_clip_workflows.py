#!/usr/bin/env python3
"""Generate two small clip-preparation workflows and matching API exports."""
from __future__ import annotations
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]


def build(multiple=False):
    values = {'mode': 'Consecutive chunks' if multiple else 'Selected chunk',
              'start_seconds': 0.0, 'first_chunk': 1, 'chunk_count': 0,
              'final_chunk': 'Skip incomplete tail', 'audio': 'Keep source audio',
              'output_prefix': 'H3_clips', 'make_loadable': True, 'make_zip': True, 'crf': 16}
    note = ('H3 SOURCE CLIP PREPARATION\n\n'
            'Upload a long video in Load Video. Three minutes or longer is fine.\n\n'
            'Exactly 124 frames / 24 fps = 5.1666667 seconds. The 5.17 label is rounded; cutting arithmetic is not.\n\n'
            'ONE SECTION: mode=Selected chunk; set start_seconds; first_chunk=1.\n'
            'ONE NUMBERED CHUNK: mode=Selected chunk; start_seconds=0; first_chunk=N.\n'
            'ALL CHUNKS: mode=Consecutive chunks; start_seconds=0; first_chunk=1; chunk_count=0.\n'
            'A RANGE: Consecutive chunks; choose first_chunk and chunk_count.\n\n'
            'Short tail is skipped by default. Hold last frame is optional and explicitly marked in the manifest. No motion is invented.\n\n'
            'Outputs: output/h3_clips/<unique run>/. Two or more clips also get a ZIP. Workbench > Files exposes MP4, ZIP and CSV.\n\n'
            'Refresh ComfyUI after export. In the HMOTION Load Video dropdown choose an h3clip_ file; set source start_seconds=0 and leave window_seconds=5.25. No re-upload needed.\n\n'
            'CPU-only FFmpeg; no H3, pose, or Ollama model loading in this workflow. Source is unchanged. H.264 CRF 16 by default, not lossless. SDR inputs only.\n\n'
            'This creates source chunks. It does not automatically render a whole movie, maintain identity between generations, or stitch generated outputs.\n\n'
            'Requires the rebuilt H3 image containing H3 Clip Preparation. The JSON alone does not install its new node.')
    def node(ident, kind, title, pos, size, inputs, outputs, widgets):
        return {'id':ident, 'type':kind, 'title':title, 'pos':pos, 'size':size,
                'flags':{}, 'order':ident-1, 'mode':0, 'inputs':inputs, 'outputs':outputs,
                'properties':{'Node name for S&R':kind}, 'widgets_values':list(widgets.values()),
                'widgets_values_named':widgets}
    nodes = [
        node(1, 'LoadVideo', '1. UPLOAD LONG SOURCE VIDEO', [0, 0], [430, 420],
             [{'name':'file','type':'COMBO','link':None,'widget':{'name':'file'}}],
             [{'name':'VIDEO','type':'VIDEO','links':[1]}], {'file':''}),
        node(2, 'WBH3ExportClips', '2. EXPORT 124-FRAME CLIPS', [500,0], [520,650],
             [{'name':'video','type':'VIDEO','link':1}],
             [{'name':'export_report','type':'STRING','links':[2]}], values),
        node(3, 'PreviewAny', '3. EXPORT REPORT + CLIP NAMES', [1080,0], [620,650],
             [{'name':'source','type':'*','link':2}], [], {}),
        node(4, 'MarkdownNote', 'USAGE', [0,490], [440,650], [], [], {'text':note}),
    ]
    graph = {'id':str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-clips-2026-10-22/'+str(multiple))),
             'revision':0, 'last_node_id':4, 'last_link_id':2, 'version':0.4,
             'nodes':nodes, 'links':[[1,1,0,2,0,'VIDEO'],[2,2,0,3,0,'STRING']], 'groups':[], 'config':{},
             'extra':{'ds':{'scale':0.75,'offset':[60,60]},'h3_clips_release':'2026-10-22.2',
                      'cpu_media_tests_run':True, 'live_comfyui_tested':False}}
    api = {'1':{'class_type':'LoadVideo','inputs':{'file':''},'_meta':{'title':nodes[0]['title']}},
           '2':{'class_type':'WBH3ExportClips','inputs':{'video':['1',0],**values},'_meta':{'title':nodes[1]['title']}},
           '3':{'class_type':'PreviewAny','inputs':{'source':['2',0]},'_meta':{'title':nodes[2]['title']}}}
    return graph, api


def main():
    for multiple, name in [(False,'01_H3_Extract_Selected_124_Frame_Clip'), (True,'02_H3_Split_Long_Video_Into_124_Frame_Clips')]:
        graph, api = build(multiple)
        for relative, data in [('workflows/h3/HCLIPS_Video_Preparation/'+name+'.json',graph),
                               ('automation/HCLIPS/'+name+'.api.json',api)]:
            path = ROOT/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()

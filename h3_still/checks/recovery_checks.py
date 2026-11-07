from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PACKAGE/'runtime'))
from fixtures import graph
from workflow_tools import portable_graph,validate_graph,get_node,PRESETS
from core_workflows import native_resize
from recovery_launcher import terminal_environment
MODELS=json.loads((PACKAGE/'models.lock.json').read_text())['models']

def edit(profile='edit_readme'):
    return portable_graph(graph(True),profile+'.json',profile,MODELS)

class RecoveryTests(unittest.TestCase):
    def test_five_presets(self):self.assertEqual(len(PRESETS),5)
    def test_no_color_or_rmbg_dependency(self):
        for profile in ('edit_readme','edit_original','edit_two'):
            g=edit(profile)
            self.assertFalse(any(n['type'].startswith('AILab_') for n in g['nodes']))
            self.assertFalse(any('COLORCODE' in str(n.get('inputs')) for n in g['nodes']))
            validate_graph(g,profile)
    def test_two_separate_image_paths(self):
        g=edit('edit_two');n=get_node(g,'MiniMaxH3ReferenceToVideo')
        links={l[0]:l for l in g['links']}
        ids=[links[p['link']][1] for p in n['inputs'] if p['name'].startswith('ref_images.') and p.get('link')]
        self.assertEqual(len(ids),2);self.assertEqual(len(set(ids)),2)
    def test_missing_native_image_rejected(self):
        g=edit();n=get_node(g,'ImageScaleToTotalPixels');n['inputs'][0]['link']=None
        with self.assertRaises(ValueError):validate_graph(g,'edit_readme')
    def test_no_fake_color_string_node(self):
        for n in edit('edit_two')['nodes']:self.assertNotEqual(n['type'],'PrimitiveString')
    def test_input_graph_preserved(self):
        g=graph(True);before=copy.deepcopy(g);native_resize(g);self.assertEqual(g,before)
    def test_used_mask_output_not_discarded(self):
        g=graph(True);n=get_node(g,'AILab_ImageResize');n['outputs'].append({'name':'MASK','links':[999]})
        with self.assertRaises(ValueError):native_resize(g)
    def test_padding_mode_not_silently_changed(self):
        g=graph(True);n=get_node(g,'AILab_ImageResize');n['widgets_values_named']['output_mode']='pad'
        with self.assertRaises(ValueError):native_resize(g)
    def test_width_height_still_and_conditioning_match(self):
        g=edit('edit_two');nodes={n['id']:n for n in g['nodes']};edges={l[0]:l for l in g['links']}
        pairs=[]
        for kind in ('MiniMaxH3ReferenceToVideo','FizgigH3StillLatent'):
            n=get_node(g,kind);pairs.append([edges[p['link']][1:3] for p in n['inputs'] if p['name'] in ('width','height')])
        self.assertEqual(pairs[0],pairs[1])
    def test_secrets_not_in_terminal_env(self):
        with patch.dict(os.environ,{'COMFY_PASSWORD':'secret','HF_TOKEN':'secret','JUPYTER_TOKEN':'secret','RUNPOD_SECRET_TEST':'secret','WB_LORA_URLS':'secret'}):
            env=terminal_environment()
            for name in ('COMFY_PASSWORD','HF_TOKEN','JUPYTER_TOKEN','RUNPOD_SECRET_TEST','WB_LORA_URLS'):self.assertNotIn(name,env)
    def test_native_resize_settings_preserved(self):
        n=get_node(edit(),'ImageScaleToTotalPixels')
        self.assertEqual(n['widgets_values'],['lanczos',2.5,32])
    def test_docker_has_shell_tools_and_jupyter(self):
        s=(PACKAGE/'Dockerfile').read_text()
        for token in ('bash curl wget procps','jupyterlab==4.5.11','jupyter-server==2.21.1','EXPOSE 8188 8888','service_launcher.py'):
            self.assertIn(token,s)
    def test_launcher_not_dependent_on_cuda_before_jupyter(self):
        s=(PACKAGE/'runtime/recovery_launcher.py').read_text()
        self.assertNotIn('import torch',s)
        self.assertLess(s.index("'-m','jupyterlab'"),s.index("PACKAGE/'runtime/launch.py'"))
    def test_auth_still_enabled(self):
        s=(PACKAGE/'runtime/recovery_launcher.py').read_text()
        self.assertIn("'password_required':True",s);self.assertIn("'disable_check_xsrf':False",s)
        self.assertNotIn("'allow_origin':'*'",s)

if __name__=='__main__':unittest.main()

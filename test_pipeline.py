import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import node
from node import logic,ollama_client as oc
from test_portrait import response

class Blocker:
    def __init__(self,reason):self.reason=reason

class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();node._PROMPT_CACHE.clear()
        self.comfy=types.ModuleType('comfy');self.mm=types.ModuleType('comfy.model_management')
        for k in ('unload_all_models','soft_empty_cache','throw_exception_if_processing_interrupted'):setattr(self.mm,k,MagicMock())
        self.comfy.model_management=self.mm
        self.fp=types.ModuleType('folder_paths');self.fp.models_dir=self.tmp.name;self.fp.get_output_directory=lambda:self.tmp.name;self.fp.get_full_path=lambda *args:'/tmp/file.safetensors'
        self.gu=types.ModuleType('comfy_execution.graph_utils');self.gu.ExecutionBlocker=Blocker
        self.modules={'comfy':self.comfy,'comfy.model_management':self.mm,'folder_paths':self.fp,'comfy_execution.graph_utils':self.gu}
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'settings.json').read_text())
        self.refs=[{'sha256':'a','filename':'head.png','image':object()}]
        self.kw=dict(references=self.refs,instruction='Image 1 is me walking.',aspect='9:16',quality='Standard',seconds=5,seed=42,use_loras=False,mode='Generate video',prompt_variation=0)
    def tearDown(self):self.tmp.cleanup()
    def run_direct(self,kw=None):
        with patch.dict(sys.modules,self.modules),patch.object(logic,'settings',return_value=self.cfg),patch.object(oc,'pipeline_info',return_value={'model':'m','digest':'d','capabilities':['vision'],'analysis':{'model':'a','digest':'a-digest','capabilities':['vision']}}),patch.object(oc,'generate',return_value=response(1)),patch.object(oc,'api',return_value={'models':[]}):
            return node.H3PortraitDirector().direct(**(kw or self.kw))
    def test_job_keeps_original_images(self):
        job,text=self.run_direct();self.assertIs(job['references'][0]['image'],self.refs[0]['image']);self.assertEqual(job['recipe']['steps'],20)
    def test_draft_blocks_downstream(self):
        k=dict(self.kw,mode='Draft only');job,text=self.run_direct(k);self.assertIsInstance(job,Blocker);self.assertIn('<Picture 1>',text)
    def test_unloads_comfy_before_ollama(self):
        self.run_direct();self.mm.unload_all_models.assert_called_once();self.mm.soft_empty_cache.assert_called_once()
    def test_cache_preserves_prompt_when_only_video_seed_changes(self):
        with patch.dict(sys.modules,self.modules),patch.object(logic,'settings',return_value=self.cfg),patch.object(oc,'pipeline_info',return_value={'model':'m','digest':'d','capabilities':['vision'],'analysis':{'model':'a','digest':'a-digest','capabilities':['vision']}}),patch.object(oc,'generate',return_value=response(1)) as g,patch.object(oc,'api',return_value={'models':[]}):
            d=node.H3PortraitDirector();d.direct(**self.kw);d.direct(**dict(self.kw,seed=43));self.assertEqual(g.call_count,1)
    def test_no_lora_fails_before_ollama(self):
        with patch.dict(sys.modules,self.modules),patch.object(logic,'settings',return_value=self.cfg),patch.object(logic,'active_loras',side_effect=ValueError('No LoRA')),patch.object(oc,'pipeline_info') as call:
            with self.assertRaises(ValueError):node.H3PortraitDirector().direct(**dict(self.kw,use_loras=True))
            call.assert_not_called()
    def test_missing_lora_file_fails_before_ollama(self):
        self.fp.get_full_path=lambda *args:None
        with patch.dict(sys.modules,self.modules),patch.object(logic,'settings',return_value=self.cfg),patch.object(logic,'active_loras',return_value=[{'lora_name':'foo'}]),patch.object(oc,'pipeline_info') as call:
            with self.assertRaises(ValueError):node.H3PortraitDirector().direct(**dict(self.kw,use_loras=True))
            call.assert_not_called()
    def test_saves_prompt_records(self):
        self.run_direct();self.assertEqual(len(list(Path(self.tmp.name).rglob('*.json'))),1)
    def test_no_grammar_requirement_in_director(self):
        k=dict(self.kw,instruction='[SINGLE_PERSON_V1 primary_headshot] Her face. Have her walk in a lobby.')
        job,text=self.run_direct(k);self.assertIn(k['instruction'],text)

class NativeCallTests(unittest.TestCase):
    def test_all_refs_go_to_native_node_in_order_with_vae(self):
        mod=types.ModuleType('comfy_extras.nodes_minimax_h3');mod.MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=MagicMock(return_value=('cond','latent')))
        pkg=types.ModuleType('comfy_extras');pkg.nodes_minimax_h3=mod
        images=[object() for _ in range(5)]
        job={'recipe':logic.geometry('2:3','Standard',5),'prompt':'text','references':[{'image':i} for i in images]}
        with patch.dict(sys.modules,{'comfy_extras':pkg,'comfy_extras.nodes_minimax_h3':mod}):
            result=node.H3PortraitConditioning().encode(job,'clip','VIDEO_VAE','AUDIO_VAE')
        self.assertEqual(result,('cond','latent'))
        k=mod.MiniMaxH3ReferenceToVideo.execute.call_args.kwargs
        self.assertEqual(list(k['ref_images'].values()),images);self.assertEqual(k['vae'],'VIDEO_VAE');self.assertEqual(k['ref_image_size'],'match')
    def test_sampler_recipe_uses_native_classes(self):
        mod=types.ModuleType('comfy_extras.nodes_custom_sampler')
        for name,result in [('RandomNoise','noise'),('BasicGuider','guide'),('KSamplerSelect','sampler'),('BasicScheduler','sigmas'),('SamplerCustomAdvanced','output')]:
            setattr(mod,name,types.SimpleNamespace(execute=MagicMock(return_value=(result,))))
        pkg=types.ModuleType('comfy_extras');pkg.nodes_custom_sampler=mod
        with patch.dict(sys.modules,{'comfy_extras':pkg,'comfy_extras.nodes_custom_sampler':mod}):
            self.assertEqual(node.H3PortraitSampler().sample({'recipe':logic.geometry('9:16','Standard',5),'seed':42},'model','cond','latent'),('output',))
        mod.BasicScheduler.execute.assert_called_once_with('model','simple',20,1.0)
        mod.SamplerCustomAdvanced.execute.assert_called_once_with('noise','guide','sampler','sigmas','latent')

if __name__=='__main__':unittest.main()

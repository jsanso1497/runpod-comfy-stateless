"""Regression tests for the delivered recipe and real local export/name helpers.

Upstream models and GPUs are not downloaded by these tests.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import catalog
import check_workflows
import install_nodes
spec=importlib.util.spec_from_file_location('quality_helpers',ROOT/'scripts/local_nodes/ComfyUI-Quality/__init__.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
CFG=ROOT/'config'
def graph(name):return json.loads((CFG/'workflows'/name).read_text())
def nodes(g,kind):return [n for n in g['nodes'] if n['type']==kind]
def source(g,n,field):
 i=next(i for i in n['inputs'] if i['name']==field)
 return next((l[1:3] for l in g['links'] if l[0]==i.get('link')),None)

class QualityRecipeTests(unittest.TestCase):
 def test_refmod_pin(self):
  row=next(r for r in json.loads((CFG/'custom_nodes.json').read_text()) if r['name']=='ComfyUI-MiniMaxH3Mod')
  self.assertEqual(row['ref'],'f9462081e28794389b5a6c5067eb327412ad8ee7');install_nodes.validate(row)
 def test_rebalance_pin(self):
  row=next(r for r in json.loads((CFG/'custom_nodes.json').read_text()) if r['name']=='Rebalance-Pack')
  self.assertEqual(row['ref'],'53c147c72c2fbd9444765af87118caee81a26d01');install_nodes.validate(row)
 def test_no_selected_turbo(self):
  rows=catalog.catalog(CFG,Path('/tmp/comfy'),set(json.loads((CFG/'runtime.json').read_text())['profiles']))
  self.assertFalse(any('turbo' in r['filename'].lower() for r in rows))
 def test_raw_krea_uses_real_guidance(self):
  for name in ['10_Krea2_Same_Subject.json','11_Krea2_Subject_into_Scene.json']:
   g=graph(name);s=nodes(g,'KSampler')[0]['widgets_values_named']
   self.assertEqual((s['steps'],s['cfg'],s['denoise']),(20,3.0,1.0))
   self.assertTrue(nodes(g,'UNETLoader')[0]['widgets_values_named']['unet_name'].startswith('krea2_raw_bf16'))
   enc=nodes(g,'Krea2EditGroundedEncode');self.assertEqual(len(enc),2)
   self.assertEqual(enc[1]['widgets_values_named']['prompt'],'')
   self.assertEqual(enc[1]['widgets_values_named']['grounding_px'],1024)
 def test_h3_baselines_standard_schedule(self):
  for name in ['20_H3_Reference_BF16.json','21_H3_Ollama_Two_Refs_BF16.json','22_H3_RefMod_Ollama_Quality.json','23_H3_Saved_RefMods_Quality.json']:
   g=graph(name);self.assertEqual(nodes(g,'BasicScheduler')[0]['widgets_values_named']['steps'],25)
   self.assertEqual(nodes(g,'UNETLoader')[0]['widgets_values_named']['unet_name'],'minimax_h3_ref2va_bf16.safetensors')
 def test_integrated_original_pixels_to_qwen(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');b=nodes(g,'EverydayOllamaH3Prompt')[0];h=nodes(g,'MiniMaxH3ReferenceToVideo')[0]
  self.assertIsNone(source(g,h,'vae'))
  self.assertEqual(source(g,h,'ref_images.ref_image_0'),[b['id'],1])
  self.assertEqual(source(g,h,'ref_images.ref_image_1'),[b['id'],2])
 def test_each_refmod_has_one_ordered_subject(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');b=nodes(g,'EverydayOllamaH3Prompt')[0]
  ext=nodes(g,'MiniMaxH3RefModExtract')
  self.assertEqual([source(g,n,'refs_image.ref_image_0') for n in ext],[[b['id'],1],[b['id'],2]])
 def test_refmod_sampling_uses_both_applies(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');a=nodes(g,'MiniMaxH3RefModApply');guider=nodes(g,'BasicGuider')[0]
  self.assertEqual(source(g,a[1],'conditioning'),[a[0]['id'],0])
  self.assertEqual(source(g,guider,'conditioning'),[a[1]['id'],0])
 def test_compression_change_is_rejected(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');nodes(g,'MiniMaxH3RefModExtract')[0]['widgets_values_named']['mode']='Compressed Reference'
  with self.assertRaises(ValueError):check_workflows.validate_graph(g)
 def test_strength_fading_is_rejected(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');nodes(g,'MiniMaxH3RefModApply')[0]['widgets_values_named']['retention']=0.7
  with self.assertRaises(ValueError):check_workflows.validate_graph(g)
 def test_truncation_is_rejected(self):
  g=graph('22_H3_RefMod_Ollama_Quality.json');nodes(g,'MiniMaxH3RefModExtract')[0]['widgets_values_named']['budget_policy']='truncate'
  with self.assertRaises(ValueError):check_workflows.validate_graph(g)
 def test_refmod_creator_no_diffusion_loading(self):
  g=graph('06_H3_Create_Full_RefMod.json');types={n['type'] for n in g['nodes']}
  self.assertFalse(types & {'UNETLoader','CLIPLoader','SamplerCustomAdvanced','EverydayOllamaH3Prompt'})
  self.assertEqual(nodes(g,'MiniMaxH3RefModExtract')[0]['widgets_values_named']['ref_resolution'],2048)
 def test_saved_references_attached_once(self):
  g=graph('23_H3_Saved_RefMods_Quality.json')
  self.assertEqual(len(nodes(g,'MiniMaxH3RefModTextEncode')),1);self.assertFalse(nodes(g,'MiniMaxH3RefModApply'))
  self.assertEqual(len(nodes(g,'QualityCheckRefModBundle')),1)
 def test_loader_no_duplicates(self):
  g=graph('23_H3_Saved_RefMods_Quality.json');w=nodes(g,'MiniMaxH3RefModsLoader')[0]['widgets_values_named']
  self.assertTrue(all(w[f'copies_{i}']==1 and w[f'strength_{i}']==1.0 for i in range(1,9)))
 def test_no_generated_code_path(self):
  for p in catalog.workflow_sources(CFG):
   self.assertFalse(any(n['type'].startswith('OmniNode') for n in json.loads(p.read_text())['nodes']))
 def test_abliterated_model_and_resources(self):
  d=json.loads((CFG/'ollama.json').read_text());self.assertEqual(d['model'],'huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16')
  self.assertEqual(d['image_max_edge'],1536);self.assertEqual(d['context_length'],16384)
  self.assertEqual(d['expected_model_digest'],'');self.assertNotIn('expected_digest',d)
 def test_named_widget_arrays_match(self):
  for p in catalog.workflow_sources(CFG):
   for n in json.loads(p.read_text())['nodes']:
    if 'widgets_values_named' in n:
     self.assertEqual(n['widgets_values'],list(n['widgets_values_named'].values()),(p.name,n['type']))
 def test_all_new_profiles_have_workflow_mapping(self):
  d=json.loads((CFG/'runtime.json').read_text())
  for name in ['06_H3_Create_Full_RefMod.json','07_Backup_RefMods.json','22_H3_RefMod_Ollama_Quality.json','23_H3_Saved_RefMods_Quality.json']:
   self.assertEqual(d['workflow_profiles'][name],'h3')
  self.assertNotIn('ideogram4', d['profiles'])
 def test_docker_no_dangling_run_continuation(self):
  s=(ROOT/'Dockerfile').read_text()
  self.assertNotRegex(s,r'\\\s*\n\s*\nRUN\b')
  self.assertIn('COPY scripts/entrypoint.sh scripts/hardware_check.py',s)
  self.assertNotIn('FROM ghcr.io/jsanso1497/runpod-comfy-stateless:latest',s)
 def test_new_local_pack_copied(self):
  self.assertIn('cp -a "$SCRIPTS/local_nodes/".',(ROOT/'scripts/prepare_image.sh').read_text())
 def test_latest_tag_still_published(self):
  path=ROOT/'.github/workflows/build-image.yml'
  if not path.exists():self.skipTest('GitHub workflow intentionally excluded from Docker build context')
  s=path.read_text();self.assertIn('type=raw,value=latest',s);self.assertIn('push: true',s)

class RefModHelperTests(unittest.TestCase):
 def test_stable_reference_name(self):
  import torch
  t=torch.zeros(1,8,8,3)
  self.assertEqual(q.reference_name(t,'person'),q.reference_name(t.clone(),'person'))
 def test_different_pixels_different_filename(self):
  import torch
  self.assertNotEqual(q.reference_name(torch.zeros(1,8,8,3),'person'),q.reference_name(torch.ones(1,8,8,3),'person'))
 def test_name_has_no_path_components(self):
  import torch
  n=q.reference_name(torch.zeros(1,8,8,3),'../../person / bad')
  self.assertNotIn('/',n);self.assertNotIn('..',n)
 def test_missing_second_subject_clear_error(self):
  with self.assertRaises(ValueError):q.reference_name(None,'subject_2')
 def test_empty_saved_selection_rejected(self):
  with self.assertRaises(ValueError):q.QualityCheckRefModBundle().check([])
 def test_selected_bundle_is_preserved(self):
  a=[('sample',1.0)];self.assertIs(q.QualityCheckRefModBundle().check(a)[0],a)
 def test_archive_contains_only_expected_reference_files(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=p/'refs';r.mkdir();(r/'identities').mkdir()
   (r/'identities/a.safetensors').write_bytes(b'fixture');(r/'a.json').write_text('{}');(r/'secret.txt').write_text('not a reference')
   filename,n=q.make_archive(r,p/'exports')
   self.assertEqual(n,2);self.assertRegex(filename,q.ARCHIVE_RE)
   with zipfile.ZipFile(p/'exports'/filename) as z:
    self.assertIn('refmods/identities/a.safetensors',z.namelist());self.assertNotIn('refmods/secret.txt',z.namelist())
 def test_archive_does_not_follow_symlinks(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=p/'refs';r.mkdir();(r/'a.safetensors').write_bytes(b'ok')
   (p/'outside.safetensors').write_bytes(b'private');(r/'bad.safetensors').symlink_to(p/'outside.safetensors')
   filename,n=q.make_archive(r,p/'out');self.assertEqual(n,1)
 def test_archive_is_size_bounded(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=p/'refs';r.mkdir();(r/'large.safetensors').write_bytes(b'12345')
   with self.assertRaises(ValueError):q.make_archive(r,p/'out',max_bytes=4)
 def test_archive_empty_directory_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):q.make_archive(Path(t),Path(t)/'out')
 def test_download_filename_validation(self):
  for name in ['../../secret','refmods-foo.zip','password.txt','refmods-20261004-123456-abcdef012345.zip/extra']:
   self.assertIsNone(q.ARCHIVE_RE.fullmatch(name))

if __name__=='__main__':unittest.main(verbosity=2)

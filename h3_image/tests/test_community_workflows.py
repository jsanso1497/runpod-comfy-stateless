import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out


builder=module('community_builder',PACKAGE/'community/build_workflows.py')
manage=module('community_manage',PACKAGE/'community/manage.py')


class CommunityGraphTests(unittest.TestCase):
    def test_all_graphs(self):
        for profile in builder.PROFILES:
            with self.subTest(file=profile[0]):
                graph,api=builder.build(profile)
                self.assertTrue(builder.validate(graph,api))

    def test_reproducible_gui_and_api(self):
        for p in builder.PROFILES:
            graph,api=builder.build(p)
            self.assertEqual(graph,json.loads((PACKAGE/'workflows'/(p[0]+'.json')).read_text()))
            self.assertEqual(api,json.loads((PACKAGE/'api_templates'/(p[0]+'.json')).read_text()))

    def test_nine_additive_workflows(self):
        self.assertEqual(len(builder.PROFILES),9)
        self.assertEqual(len({p[0] for p in builder.PROFILES}),9)
        self.assertTrue(all(int(p[0][:2])>=10 for p in builder.PROFILES))

    def test_manual_prompt_only(self):
        for p in builder.PROFILES:
            g,api=builder.build(p)
            self.assertEqual(api['40']['inputs']['prompt'],'')
            self.assertEqual(api['40']['class_type'],'H3ImageManualConditioning')
            self.assertFalse(any(n['type'].startswith(('TextGenerate','H3LabPrompt')) for n in g['nodes']))

    def test_fixed_seed_all_variants(self):
        for p in builder.PROFILES:
            g,api=builder.build(p)
            n=next(n for n in g['nodes'] if n['id']==43)
            self.assertEqual(n['widgets_values_named']['control_after_generate'],'fixed')
            self.assertEqual(api['43']['inputs']['noise_seed'],18071042)

    def test_same_base_model_set(self):
        for p in builder.PROFILES:
            _,api=builder.build(p)
            self.assertEqual(api['30']['inputs']['unet_name'],builder.BASE.MODEL)
            self.assertEqual(api['31']['inputs']['clip_name'],builder.BASE.ENCODER)
            self.assertEqual(api['32']['inputs']['vae_name'],builder.BASE.VAE)

    def test_sixteen_mp_all(self):
        for p in builder.PROFILES:
            _,api=builder.build(p)
            self.assertEqual(api['40']['inputs']['output_megapixels'],16.0)

    def test_euler_control_and_ersde_counts(self):
        for p in builder.PROFILES:
            _,api=builder.build(p)
            self.assertEqual(api['41']['inputs']['scheduler'],'simple')
            self.assertEqual(api['41']['inputs']['steps'],p[3])
            self.assertEqual(api['41']['inputs']['denoise'],1.0)
            self.assertIn(p[3],(4,6))
            self.assertEqual(api['44']['inputs']['sampler'],['51',0])
            self.assertEqual(api['44']['inputs']['sigmas'],['41',0])

    def test_lcm_noise_exposed_both_ends(self):
        for p in builder.PROFILES:
            if p[2]!='lcm':continue
            _,api=builder.build(p)
            self.assertEqual(api['51']['class_type'],'SamplerLCM')
            self.assertEqual(api['51']['inputs'],{'s_noise':1.0,'s_noise_end':1.0,'noise_clip_std':0.0})

    def test_no_automatic_lora_or_enhancer(self):
        for p in builder.PROFILES:
            g,_=builder.build(p)
            self.assertFalse(any('Lora' in n['type'] or 'Fizgig' in n['type'] or 'TextGenerate' in n['type'] for n in g['nodes']))

    def test_reference_roles_and_count(self):
        for p in builder.PROFILES:
            g,api=builder.build(p)
            count=0 if p[1]==0 else 3 if p[1]==3 else 4
            self.assertEqual(api['40']['inputs']['minimum_references'],count)
            self.assertEqual(sum(n['type']=='H3ImageOptionalReference' for n in g['nodes']),count)
            for i in range(1,count+1):self.assertEqual(api['40']['inputs']['reference_'+str(i)],[str(i),0])

    def test_replacement_follows_scene(self):
        for p in builder.PROFILES:
            if p[1]!=3:continue
            _,api=builder.build(p)
            self.assertTrue(api['40']['inputs']['follow_image_1_aspect'])
            self.assertEqual(api['40']['inputs']['aspect_source_slot'],0)

    def test_two_people_new_composition(self):
        for p in builder.PROFILES:
            if p[1]!=4:continue
            _,api=builder.build(p)
            self.assertFalse(api['40']['inputs']['follow_image_1_aspect'])
            self.assertEqual(api['40']['inputs']['aspect_ratio'],'3:2')

    def test_pruna_same_latent_single_sample(self):
        g,api=builder.build(builder.PROFILES[-1])
        self.assertEqual(sum(n['type']=='H3ImageTimedSampler' for n in g['nodes']),1)
        self.assertEqual(api['45']['inputs']['samples'],api['53']['inputs']['samples'])
        self.assertEqual(api['40']['inputs']['vae'],['32',0])
        self.assertEqual(api['45']['inputs']['vae'],['32',0])
        self.assertEqual(api['53']['inputs']['vae'],['52',0])
        self.assertEqual(api['52']['inputs']['vae_name'],builder.PRUNA_VAE)

    def test_vae_ab_keeps_full_res_outputs_separate_from_preview(self):
        _,api=builder.build(builder.PROFILES[-1])
        self.assertEqual(api['46']['inputs']['images'],['45',0])
        self.assertEqual(api['54']['inputs']['images'],['53',0])
        self.assertEqual(api['58']['inputs']['image1'],['56',0])
        self.assertEqual(api['58']['inputs']['image2'],['57',0])
        self.assertEqual(api['56']['inputs']['largest_size'],1024)
        self.assertEqual(api['57']['inputs']['largest_size'],1024)

    def test_no_shared_original_output_prefix(self):
        for p in builder.PROFILES:
            _,api=builder.build(p)
            for n in api.values():
                if n['class_type']=='SaveImage':self.assertTrue(n['inputs']['filename_prefix'].startswith('H3_Image/Community_Tests/'))

    def test_topological_execution(self):
        for p in builder.PROFILES:
            g,_=builder.build(p);nodes={n['id']:n for n in g['nodes']}
            for _,s,_,d,_,_ in g['links']:self.assertLess(nodes[s]['order'],nodes[d]['order'])

    def test_missing_link_rejected(self):
        g,api=builder.build(builder.PROFILES[0]);g['links'].pop()
        with self.assertRaises((AssertionError,KeyError)):builder.validate(g,api)

    def test_cycle_rejected(self):
        g,_=builder.build(builder.PROFILES[0])
        # Make an artificial self-edge in a standalone graph for ordering validation.
        tiny={'nodes':[{'id':1,'inputs':[{'name':'x','link':1}], 'outputs':[{'name':'x','type':'X','links':[1]}]}],
              'links':[[1,1,0,1,0,'X']]}
        with self.assertRaises(ValueError):builder.reindex(tiny)

    def test_source_reconstruction_flag(self):
        for p in builder.PROFILES:
            g,_=builder.build(p)
            self.assertIn('NOT an extracted',g['extra']['h3_image']['source'])

    def test_pruna_download_is_not_in_startup(self):
        self.assertNotIn('PrunaAI', (PACKAGE/'runtime/backend.py').read_text())
        self.assertEqual(len(json.loads((PACKAGE/'sources.json').read_text())['models']),3)

    def test_different_vae_only_in_ab(self):
        for p in builder.PROFILES:
            g,_=builder.build(p)
            self.assertEqual(sum(n['type']=='VAELoader' for n in g['nodes']),2 if p[4] else 1)

    def test_metadata_scan_record(self):
        d=json.loads((PACKAGE/'community/source_image_metadata.json').read_text())
        self.assertEqual(d['format'],'JPEG')
        self.assertEqual(d['size'],[1536,657])
        self.assertTrue(all(v==-1 for v in d['markers'].values()))
        self.assertEqual(d['trailing_bytes_after_jpeg_eoi'],0)


class OptionalDecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'root'
        self.body=b'small test data, not model weights'
        self.row=dict(manage.MODEL,size_bytes=len(self.body),sha256=hashlib.sha256(self.body).hexdigest())
    def tearDown(self):self.temp.cleanup()
    def downloader(self,**kwargs):
        self.kwargs=kwargs;p=Path(kwargs['local_dir'])/kwargs['filename'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(self.body);return str(p)
    def test_checksum_good(self):
        p=Path(self.temp.name)/'a';p.write_bytes(self.body);self.assertTrue(manage.verify(p,self.row))
    def test_checksum_bad(self):
        p=Path(self.temp.name)/'a';p.write_bytes(b'x'*len(self.body));self.assertFalse(manage.verify(p,self.row))
    def test_download_exact_revision_and_file(self):
        p=manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
        self.assertTrue(p.is_symlink());self.assertEqual(p.read_bytes(),self.body)
        self.assertEqual(self.kwargs['revision'],self.row['revision']);self.assertEqual(self.kwargs['filename'],self.row['file'])
    def test_existing_match_no_network(self):
        p=manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
        def no(**kw):raise AssertionError('Network should not run')
        self.assertEqual(manage.download_vae(self.root,no,reuse=[],model=self.row),p)
    def test_existing_mismatch_not_overwritten(self):
        p=self.root/'models/vae'/self.row['name'];p.parent.mkdir(parents=True);p.write_bytes(b'do not replace')
        with self.assertRaises(RuntimeError):manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
        self.assertEqual(p.read_bytes(),b'do not replace')
    def test_reuses_old_workspace_without_modification(self):
        old=Path(self.temp.name)/'old.safetensors';old.write_bytes(self.body)
        p=manage.download_vae(self.root,lambda **kw: self.fail('network'),reuse=[old],model=self.row)
        self.assertEqual(p.resolve(),old.resolve());self.assertEqual(old.read_bytes(),self.body)
    def test_unexpected_download_path_rejected(self):
        with self.assertRaises(RuntimeError):manage.download_vae(self.root,lambda **kw: '/tmp/not-the-model',reuse=[],model=self.row)
    def test_low_disk_rejected(self):
        with patch.object(manage.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10,10,0)):
            with self.assertRaises(RuntimeError):manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
    def test_cached_mismatch_not_overwritten(self):
        source=self.root/'community/downloads/pruna_vae'/self.row['file'];source.parent.mkdir(parents=True);source.write_bytes(b'bad')
        with self.assertRaises(RuntimeError):manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
        self.assertEqual(source.read_bytes(),b'bad')
    def test_symlinked_model_dir_escape_rejected(self):
        outside=Path(self.temp.name)/'elsewhere';outside.mkdir()
        (self.root/'models').mkdir(parents=True);(self.root/'models/vae').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(RuntimeError):manage.download_vae(self.root,self.downloader,reuse=[],model=self.row)
    def test_live_schema_complete(self):
        info={name:{'input':{'required':{}}} for name in manage.REQUIRED_HELPERS}
        for name,values in manage.CORE_INPUTS.items():info[name]={'input':{'required':{n:['ANY'] for n in values}}}
        info['KSamplerSelect']['input']['required']['sampler_name']=[['euler','er_sde']]
        info['BasicScheduler']['input']['required']['scheduler']=[['simple']]
        self.assertEqual(manage.validate_live_info(info),[])
        del info['SamplerLCM']['input']['required']['s_noise_end']
        self.assertTrue(any('s_noise_end' in e for e in manage.validate_live_info(info)))
    def test_live_missing_nodes(self):self.assertTrue(any('Missing node:' in e for e in manage.validate_live_info({})))
    def test_published_asset_lock(self):
        self.assertEqual(manage.MODEL['size_bytes'],3146996440)
        self.assertEqual(manage.MODEL['sha256'],'583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe')


class CommunityNativeBuildTests(unittest.TestCase):
    @unittest.skipUnless(Path('/opt/ComfyUI/comfy').is_dir() and importlib.util.find_spec('torch') is not None,
                         'Native core/dependencies only exist inside the built workspace image')
    def test_real_cpu_sampler_constructors(self):
        result=subprocess.run([sys.executable,str(PACKAGE/'community/native_smoke.py')],
                              text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout[-16000:])


if __name__=='__main__':unittest.main()

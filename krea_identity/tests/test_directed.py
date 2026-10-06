"""CPU behavior/graph tests only. These do not measure Krea likeness or realism."""
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

N=load('krea_directed_nodes',HERE/'node/__init__.py')
C=load('krea_directed_check',HERE/'check.py')
G=load('krea_multi_graph_builder_tests',HERE/'make_multi_workflows.py')


def image(h=96,w=128,value=.5,dtype=torch.float32):
    return torch.full((1,h,w,3),value,dtype=dtype)


def refs(n=2):
    return tuple({'image':image(96,64,i/max(n,1)), 'label':f'original {i+1}', 'use_for':f'purpose {i+1}'} for i in range(n))


def directed(scene=None,reference_group='A',references=None,**kwargs):
    values=dict(scene=image() if scene is None else scene,references=refs() if references is None else references,
                target_description='man on left in gray suit',replacement_description='man from my original references',
                clothing='Keep scene clothing',extra_instruction='Keep the seated child unchanged.',megapixels=.25,
                reference_group=reference_group)
    values.update(kwargs)
    return N.KreaIdentityDirectedEdit().prepare(**values)


def couple(**kwargs):
    values=dict(scene=image(),references_a=refs(),references_b=refs(),target_a='man on left',replacement_a='man from A',
                clothing_a='Keep scene clothing',target_b='woman on right',replacement_b='woman from B',
                clothing_b='Use labeled wardrobe references',extra_instruction='Keep the child in the middle.',megapixels=.25)
    values.update(kwargs)
    return N.KreaIdentityCoupleEdit().prepare(**values)


def region(scene=None,**kwargs):
    values=dict(image=image(128,256) if scene is None else scene,left=.05,top=.10,width=.4,height=.8,
                context_padding=.12,feather_pixels=8)
    values.update(kwargs)
    return N.KreaIdentityRegion().prepare(**values)


def graph(number):
    return json.loads(next((HERE/'workflows').glob(f'*_{number}_*')).read_text())


class ReferenceLists(unittest.TestCase):
    def test_one_original_is_not_beautified_or_copied(self):
        photo=image()
        result=N.KreaIdentityReference().add(photo,'Face','Identity only',True)[0]
        self.assertEqual(result[0]['image'].data_ptr(),photo.data_ptr())
    def test_appending_does_not_mutate_the_original_bank(self):
        a=refs(1)
        b=N.KreaIdentityReference().add(image(),'Body','Body only',True,a)[0]
        self.assertEqual(len(a),1);self.assertEqual(len(b),2)
    def test_six_references_supported(self):
        self.assertEqual(len(N.checked_references(refs(6))),6)
    def test_seven_references_rejected(self):
        with self.assertRaisesRegex(ValueError,'PER person'):N.checked_references(refs(7))
    def test_empty_bank_stops_before_sampling(self):
        with self.assertRaises(ValueError):directed(references=())
    def test_disabled_reference_does_not_enter_bank(self):
        result=N.KreaIdentityReference().add(image(),'Face','unused',False,refs())[0]
        self.assertEqual(len(result),2)
    def test_disabled_first_node_allows_later_reference(self):
        a=N.KreaIdentityReference().add(image(),'Face','unused',False)[0]
        self.assertEqual(a,())
        b=N.KreaIdentityReference().add(image(),'Body','Physique',True,a)[0]
        self.assertEqual(len(b),1)
    def test_non_boolean_include_rejected(self):
        with self.assertRaises(ValueError):N.KreaIdentityReference().add(image(),'Face','Identity',1)
    def test_duplicate_labels_rejected_case_insensitively(self):
        with self.assertRaisesRegex(ValueError,'unique'):N.KreaIdentityReference().add(image(),'ORIGINAL 1','hair only',True,refs(1))
    def test_two_people_may_reuse_labels(self):
        self.assertEqual(len(couple()),6)
    def test_missing_label_rejected(self):
        with self.assertRaises(ValueError):N.KreaIdentityReference().add(image(),' ','identity',True)
    def test_missing_purpose_rejected(self):
        with self.assertRaises(ValueError):N.KreaIdentityReference().add(image(),'face',' ',True)
    def test_overlong_purpose_rejected(self):
        with self.assertRaises(ValueError):N.KreaIdentityReference().add(image(),'face','x'*601,True)
    def test_unicode_labels_preserved_in_prompt(self):
        r=N.KreaIdentityReference().add(image(),'Profil\u00e9','facial geometry',True)[0]
        self.assertIn('Profil\u00e9',directed(references=r)[2])
    def test_unknown_reference_structure_rejected(self):
        with self.assertRaises(ValueError):N.checked_references([{'foo':1}])
    def test_nan_reference_rejected(self):
        im=image();im[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):N.KreaIdentityReference().add(im,'face','identity',True)


class SheetGeometry(unittest.TestCase):
    def test_all_panel_counts_render(self):
        for n in range(1,7):
            with self.subTest(n=n):
                sheet,manifest=N.labeled_sheet(refs(n),'A',512,512)
                self.assertEqual(sheet.shape,(1,512,512,3))
                self.assertEqual(len(manifest.splitlines()),n)
                self.assertTrue(torch.isfinite(sheet).all())
    def test_board_is_bounded_independently_of_source_resolution(self):
        sheet,_=N.labeled_sheet(refs(6),'B')
        self.assertEqual(sheet.shape,(1,1536,1536,3))
    def test_primary_identity_gets_half_the_board_with_many_refs(self):
        r=list(refs(6));r[0]['image']=image(128,64,1)
        sheet,_=N.labeled_sheet(r,'A',512,512)
        self.assertEqual(sheet[0,256,128,0],1)
    def test_one_reference_entire_image_is_letterboxed_not_cropped(self):
        r=({'image':image(32,128,1),'label':'wide','use_for':'body'},)
        sheet,_=N.labeled_sheet(r,'A',256,256)
        self.assertAlmostEqual(float(sheet[0,128,0,0]),1,places=6)
        self.assertAlmostEqual(float(sheet[0,128,-1,0]),1,places=6)
        self.assertEqual(sheet[0,40,128,0],.5)
    def test_labels_are_outside_reference_pixels(self):
        sheet,_=N.labeled_sheet(refs(1),'A',256,256)
        self.assertTrue((sheet[0,10,10] != 0).any())
        self.assertEqual(sheet[0,128,128,0],0)
    def test_wrong_group_rejected(self):
        with self.assertRaises(ValueError):N.labeled_sheet(refs(),'C')
    def test_man_woman_board_namespaces_remain_distinct(self):
        result=couple()
        self.assertEqual(result[1].shape,(1,1024,2048,3))
        for panel in ('A1:','A2:','B1:','B2:'):
            self.assertIn(panel,result[2])
    def test_different_photo_dtypes_supported(self):
        a=list(refs());a[1]['image']=a[1]['image'].half()
        sheet,_=N.labeled_sheet(a,'A',256,256)
        self.assertEqual(sheet.dtype,torch.float32)


class DirectedInstructions(unittest.TestCase):
    def test_user_identifies_exact_target_and_replacement(self):
        result=directed()
        self.assertIn('[man on left in gray suit]',result[2])
        self.assertIn('[man from my original references]',result[2])
    def test_multiple_people_are_not_reduced_to_one_person(self):
        text=directed()[2]
        self.assertIn('person count',text);self.assertIn('unselected people',text)
        self.assertNotIn('photograph of one person',text)
    def test_extra_instructions_survive(self):
        self.assertIn('Keep the seated child unchanged.',directed()[2])
    def test_empty_target_rejected(self):
        with self.assertRaises(ValueError):directed(target_description=' ')
    def test_empty_replacement_rejected(self):
        with self.assertRaises(ValueError):directed(replacement_description=' ')
    def test_same_two_targets_rejected(self):
        with self.assertRaisesRegex(ValueError,'DIFFERENT'):couple(target_a='Man on left',target_b='man on left')
    def test_clothing_choices_independent(self):
        prompt=couple()[2]
        a=prompt.split('REPLACEMENT A:')[1].split('REPLACEMENT B:')[0]
        b=prompt.split('REPLACEMENT B:')[1]
        self.assertIn('original scene clothing',a)
        self.assertIn('clothing only from panels',b)
    def test_blank_extra_instruction_valid(self):
        self.assertEqual(len(directed(extra_instruction='')),6)
    def test_unknown_clothing_rejected(self):
        with self.assertRaises(ValueError):directed(clothing='guess')
    def test_local_woman_pass_uses_B_labels(self):
        prompt=directed(reference_group='B')[2]
        self.assertIn('REPLACEMENT B:',prompt)
        self.assertIn('B1:',prompt);self.assertNotIn('A1:',prompt)
    def test_source_is_image_one_refs_are_image_two(self):
        self.assertIn('Image 1 is the scene',directed()[2])
        self.assertIn('Image 2 contains labeled reference',directed()[2])
    def test_explicit_no_identity_mixing(self):
        self.assertIn('Never use A panels for target B',couple()[2])
    def test_all_six_purposes_reach_prompt(self):
        text=directed(references=refs(6))[2]
        for n in range(1,7):self.assertIn(f'purpose {n}',text)


class CanvasAndAspect(unittest.TestCase):
    def test_landscape(self):
        result=directed(scene=image(108,192))
        self.assertGreater(result[3],result[4])
    def test_portrait(self):
        result=directed(scene=image(192,108))
        self.assertLess(result[3],result[4])
    def test_square(self):
        result=directed(scene=image(96,96));self.assertEqual(result[3],result[4])
    def test_couple_uses_scene_not_reference_aspect(self):
        result=couple(scene=image(80,200))
        self.assertGreater(result[3],result[4])
    def test_two_megapixel_limit(self):
        result=couple(megapixels=2)
        self.assertLessEqual(result[3]*result[4],2_000_000)
    def test_canvas_grid_and_unframe_across_aspects(self):
        for w,h in ((192,108),(108,192),(200,200),(311,79),(513,377)):
            with self.subTest(w=w,h=h):
                out,ow,oh,g=N.framed_canvas(image(h,w),.25)
                self.assertEqual(ow%16,0);self.assertEqual(oh%16,0)
                restored=N.unframe(out,g,True)
                self.assertEqual(restored.shape,(1,h,w,3))
                torch.testing.assert_close(restored,image(h,w),atol=1e-6,rtol=1e-6)
    def test_full_source_edge_is_included(self):
        im=image(97,163,0);im[:,:3]=1;im[:,-3:]=1;im[:,:,:3]=1;im[:,:,-3:]=1
        canvas,w,h,g=N.framed_canvas(im,.25)
        result=N.unframe(canvas,g,True)
        self.assertGreater(float(result[0,0,80,0]),.9)
        self.assertGreater(float(result[0,-1,80,0]),.9)
        self.assertGreater(float(result[0,48,0,0]),.9)
        self.assertGreater(float(result[0,48,-1,0]),.9)
    def test_bad_canvas_output_size_rejected(self):
        _,_,_,g=N.framed_canvas(image(),.25)
        with self.assertRaises(ValueError):N.unframe(image(),g)
    def test_forged_canvas_bounds_rejected(self):
        ready,_,_,g=N.framed_canvas(image(),.25);g['width']=9999
        with self.assertRaises(ValueError):N.unframe(ready,g)


class ProtectedRegions(unittest.TestCase):
    def test_context_extends_beyond_edit_region(self):
        crop,g,mask,_=region()
        self.assertGreater(crop.shape[2],int((mask[0]>0).any(0).sum()))
    def test_rectangle_preview_does_not_modify_input_crop(self):
        im=image(128,256,.6);before=im.clone()
        crop,g,mask,preview=region(im)
        self.assertTrue(torch.equal(im,before))
        self.assertAlmostEqual(float(crop.mean()),.6,places=5)
        self.assertLess(float(preview[0,0,0,0]),float(im[0,0,0,0]))
    def test_empty_painted_mask_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'empty'):region(edit_mask=torch.zeros(1,128,256))
    def test_mismatched_mask_rejected_not_resized(self):
        with self.assertRaisesRegex(ValueError,'dimensions'):region(edit_mask=torch.ones(1,64,64))
    def test_painted_mask_overrides_rectangle(self):
        mask=torch.zeros(1,128,256);mask[:,40:100,180:230]=1
        _,_,effective,_=region(edit_mask=mask,feather_pixels=0)
        self.assertTrue(torch.equal(mask,effective))
    def test_second_mask_cannot_touch_first_including_soft_edges(self):
        _,_,first,_=region()
        _,_,second,_=region(left=.25,width=.7,protect_mask=first)
        self.assertFalse(bool(((first>0)&(second>0)).any()))
    def test_composite_preserves_outside_exactly(self):
        im=torch.rand(1,128,256,3);before=im.clone()
        crop,g,mask,_=region(im)
        _,w,h,cg=N.framed_canvas(crop,.25)
        out=N.KreaIdentityRegionComposite().composite(im,image(h,w,1),g,cg)[0]
        self.assertTrue(torch.equal(im,before))
        self.assertTrue(torch.equal(out[mask==0],im[mask==0]))
        self.assertGreater(float((out-im).abs().sum()),0)
    def test_two_pass_simulation_retains_first_edit_exactly(self):
        im=torch.rand(1,128,256,3)
        crop_a,ga,ma,_=region(im)
        _,wa,ha,cga=N.framed_canvas(crop_a,.25)
        ca=N.KreaIdentityRegionComposite().composite(im,image(ha,wa,.1),ga,cga)[0]
        crop_b,gb,mb,_=region(ca,left=.35,width=.6,protect_mask=ma)
        _,wb,hb,cgb=N.framed_canvas(crop_b,.25)
        cb=N.KreaIdentityRegionComposite().composite(ca,image(hb,wb,.9),gb,cgb)[0]
        self.assertTrue(torch.equal(cb[ma>0],ca[ma>0]))
        outside=(ma==0)&(mb==0)
        self.assertTrue(torch.equal(cb[outside],im[outside]))
        self.assertEqual(cb.shape,im.shape)
    def test_all_protected_region_rejected(self):
        with self.assertRaisesRegex(ValueError,'protected'):region(protect_mask=torch.ones(1,128,256))
    def test_negative_mask_rejected(self):
        with self.assertRaises(ValueError):region(edit_mask=torch.full((1,128,256),-.1))
    def test_nan_mask_rejected(self):
        with self.assertRaises(ValueError):region(edit_mask=torch.full((1,128,256),float('nan')))
    def test_invalid_fraction_rejected(self):
        for args in ({'left':-1},{'top':float('nan')},{'width':0},{'height':2}):
            with self.subTest(args=args),self.assertRaises(ValueError):region(**args)
    def test_invalid_feather_rejected(self):
        with self.assertRaises(ValueError):region(feather_pixels=1.5)
    def test_full_frame_mask_has_finite_feather(self):
        crop,g,mask,_=region(edit_mask=torch.ones(1,128,256))
        self.assertTrue(torch.isfinite(mask).all());self.assertGreater(mask.max(),0)
    def test_geometry_from_different_crop_rejected(self):
        im=image(128,256);crop,g,_,_=region(im)
        ready,w,h,cg=N.framed_canvas(crop,.25);cg['original_width']+=1
        with self.assertRaises(ValueError):N.KreaIdentityRegionComposite().composite(im,ready,g,cg)
    def test_original_scene_dimension_mismatch_rejected(self):
        crop,g,_,_=region();ready,w,h,cg=N.framed_canvas(crop,.25)
        with self.assertRaises(ValueError):N.KreaIdentityRegionComposite().composite(image(256,256),ready,g,cg)
    def test_zero_feather_retains_hard_mask(self):
        _,_,mask,_=region(feather_pixels=0)
        self.assertEqual(set(mask.unique().tolist()),{0.,1.})
    def test_low_precision_protected_pixels_unchanged(self):
        im=torch.rand(1,128,256,3).half();crop,g,mask,_=region(im)
        ready,w,h,cg=N.framed_canvas(crop,.25)
        out=N.KreaIdentityRegionComposite().composite(im,torch.ones_like(ready),g,cg)[0]
        self.assertEqual(out.dtype,im.dtype)
        self.assertTrue(torch.equal(out[mask==0],im[mask==0]))
    def test_tiny_region_rejected(self):
        with self.assertRaises(ValueError):region(width=.01,height=.01)


class NewGraphContracts(unittest.TestCase):
    def test_three_added_graphs_pass(self):
        for number in ('05','06','07'):C.validate_graph(graph(number))
    def test_all_local_widget_orders_match_class_schema(self):
        for number in ('05','06','07'):
            for n in graph(number)['nodes']:
                if n['type'] not in N.NODE_CLASS_MAPPINGS:continue
                cls=N.NODE_CLASS_MAPPINGS[n['type']]
                expected=[]
                for group in ('required','optional'):
                    for name,definition in cls.INPUT_TYPES().get(group,{}).items():
                        t=definition[0]
                        if isinstance(t,list) or t in ('STRING','INT','FLOAT','BOOLEAN'):expected.append(name)
                self.assertEqual(list(n['widgets_values_named']),expected,n['type'])
    def test_all_local_kwargs_match_function_signatures(self):
        for number in ('05','06','07'):
            for n in graph(number)['nodes']:
                if n['type'] not in N.NODE_CLASS_MAPPINGS:continue
                cls=N.NODE_CLASS_MAPPINGS[n['type']]
                signature=inspect.signature(getattr(cls,cls.FUNCTION))
                names=set(n['widgets_values_named'])|{i['name'] for i in n['inputs']}
                self.assertTrue(names.issubset(signature.parameters),n['type'])
    def test_optional_reference_sockets_are_not_converted_widgets(self):
        for number in ('05','06','07'):
            for n in graph(number)['nodes']:
                if n['type']=='KreaIdentityReference':
                    inp=next(i for i in n['inputs'] if i['name']=='references')
                    self.assertNotIn('widget',inp)
    def test_missing_protection_fails_validation(self):
        g=graph('07');n=next(n for n in g['nodes'] if n['type']=='KreaIdentityRegion' and any(i['name']=='protect_mask' and i['link'] for i in n['inputs']))
        inp=next(i for i in n['inputs'] if i['name']=='protect_mask');inp['link']=None
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_shifted_seed_widgets_fail_validation(self):
        g=graph('06');n=next(n for n in g['nodes'] if n['type']=='KSampler');n['widgets_values'].pop(1)
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_rebalancer_is_optional_in_every_pass(self):
        for number in ('05','06','07'):
            for n in graph(number)['nodes']:
                if n['type']=='KreaIdentityRebalanceControl':self.assertFalse(n['widgets_values_named']['enabled'])
    def test_woman_pass_gets_its_own_group(self):
        g=graph('07');s=g['extra']['krea_identity']['stages'][1]
        prep=next(n for n in g['nodes'] if n['id']==s['prepare'])
        self.assertEqual(prep['widgets_values_named']['reference_group'],'B')
    def test_only_one_model_load_in_two_pass_graph(self):
        g=graph('07')
        self.assertEqual(sum(n['type']=='UNETLoader' for n in g['nodes']),1)
        self.assertEqual(sum(n['type']=='KSampler' for n in g['nodes']),2)
    def test_new_generator_is_reproducible(self):
        expected={p.name:p.read_bytes() for p in (HERE/'workflows').glob('*_v1_1.json')}
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp);(target/'workflows').mkdir()
            with patch.object(G,'HERE',target),patch.object(G.base,'HERE',target):G.main()
            for name,data in expected.items():self.assertEqual((target/'workflows'/name).read_bytes(),data)
    @unittest.skipUnless((HERE.parent/'Dockerfile').is_file(), 'repository-only Docker recipe assertion; actual registry gate runs separately in Docker')
    def test_build_still_requires_real_comfy_registry(self):
        docker=(HERE.parent/'Dockerfile').read_text()
        self.assertIn('check.py --build-smoke',docker)
        text=(HERE/'check.py').read_text()
        self.assertIn('for graph in graphs:validate_schema(graph,registry)',text)
    def test_no_additional_upstream_node_dependency(self):
        deps=json.loads((HERE/'config/custom_nodes.json').read_text())
        self.assertEqual({x['name'] for x in deps},{'comfyui-krea2edit','Rebalance-Pack'})


if __name__=='__main__':unittest.main()

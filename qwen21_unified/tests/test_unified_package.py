import importlib.util,json,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'qwen21_unified/scripts'
sys.path.insert(0,str(SCRIPTS))
from filter_requirements import filter_file
from generate_dockerfile import generate
from runtime_prepare import install_for

PHOTO_DOCKERFILE='''# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y git curl tini
RUN git clone https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI
COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/
RUN python /opt/qwen21_photo_edit/pin_torch.py /tmp/pytorch-constraints.txt
RUN set -Eeuo pipefail \\
 && python -m pip install --constraint /tmp/pytorch-constraints.txt -r /opt/ComfyUI/requirements.txt \\
 && python -m pip install --constraint /tmp/pytorch-constraints.txt scipy huggingface-hub pillow \\
 && python -m pip check \\
 && python /opt/qwen21_photo_edit/preflight.py --comfy-home /opt/ComfyUI --sam
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen21_photo_edit/start.sh"]
'''

class Tests(unittest.TestCase):
    def test_removes_only_media_bundle(self):
        with tempfile.TemporaryDirectory() as t:
            src=Path(t)/'requirements.txt';dst=Path(t)/'filtered.txt'
            src.write_text('comfyui-frontend-package==1.55.14\ncomfyui-workflow-templates==0.11.78\ntorch\ntransformers>=4.50.3\nscipy\nsafetensors\n')
            lines=filter_file(src,dst)
            self.assertEqual(len(lines),1)
            result=dst.read_text()
            self.assertNotIn('comfyui-workflow-templates',result)
            for keep in ['comfyui-frontend-package','torch','transformers','scipy','safetensors']:
                self.assertIn(keep,result)
    def test_rejects_unrecognized_dependency_table(self):
        with tempfile.TemporaryDirectory() as t:
            src=Path(t)/'requirements.txt';src.write_text('torch\ntransformers\ncomfyui-frontend-package\n')
            with self.assertRaises(RuntimeError):filter_file(src,Path(t)/'filtered.txt')
    def test_photo_boot_and_comfy_stack_preserved(self):
        d=generate(PHOTO_DOCKERFILE)
        self.assertIn('qwen21_photo_edit/preflight.py',d)
        self.assertIn('COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/',d)
        self.assertIn('COPY qwen21_native/ /opt/qwen21_native/',d)
        self.assertIn('COPY qwen21_unified/ /opt/qwen21_unified/',d)
        self.assertIn('filter_requirements.py',d)
        self.assertIn('--no-cache-dir',d)
        self.assertIn('build_install.py',d)
        self.assertIn('entry.sh',d)
        self.assertIn('FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime',d)
        self.assertNotIn('-r /opt/ComfyUI/requirements.txt',d)
        self.assertEqual(d.count('ENTRYPOINT ['),1)
    def test_fail_closed_when_photo_docker_changed(self):
        with self.assertRaises(ValueError):generate(PHOTO_DOCKERFILE.replace('/opt/ComfyUI/requirements.txt','/opt/new_requirements.txt'))
        with self.assertRaises(ValueError):generate(PHOTO_DOCKERFILE.replace('/opt/qwen21_photo_edit/start.sh','/bin/false'))
    def test_installer_native_and_torso_visible_and_preserved(self):
        import runtime_prepare
        original=runtime_prepare.NATIVE
        runtime_prepare.NATIVE=ROOT/'qwen21_native'
        try:
            with tempfile.TemporaryDirectory() as t:
                comfy=Path(t)/'ComfyUI'; user=Path(t)/'user'
                qwen=comfy/'comfy_extras/nodes_qwen.py';qwen.parent.mkdir(parents=True)
                qwen.write_text('class TextEncodeQwenImage21: pass')
                aus=comfy/'custom_nodes/ComfyUI-AusBoss/nodes/node_inpaint_crop_stitch.py'
                aus.parent.mkdir(parents=True);aus.write_text('AUSBOSS_NODES_StitchInpaint')
                pre=user/'default/workflows/Existing'/ 'user_modified.json'
                pre.parent.mkdir(parents=True);pre.write_text('{"my":"edit"}')
                self.assertEqual(install_for(comfy,user),38)
                native=list((user/'default/workflows/Qwen21 Native v1.1.0').glob('*.json'))
                torso=list((user/'default/workflows/Qwen21 Torso Lock v1.0.0').glob('*.json'))
                self.assertEqual(len(native),34)
                self.assertEqual(len(torso),4)
                self.assertEqual(pre.read_text(),'{"my":"edit"}')
                self.assertTrue((comfy/'custom_nodes/qwen21_native_tools/__init__.py').exists())
                self.assertTrue((comfy/'custom_nodes/qwen21_torso_tools/__init__.py').exists())
                self.assertEqual(install_for(comfy,user),38)
        finally:runtime_prepare.NATIVE=original
    def test_model_assets_not_baked(self):
        count=0
        for p in (ROOT/'qwen21_native').rglob('*'):
            if p.suffix=='.safetensors':count+=1
        self.assertEqual(count,0)
    def test_unified_template_contract(self):
        template=json.loads((ROOT/'qwen21_unified/config/runpod-template.json').read_text())
        self.assertIn(':unified-v2.1.3',template['imageName'])
        self.assertIn('8188/http',template['ports'])
        self.assertEqual(template['dockerEntrypoint'],[])
        self.assertEqual(template['env']['QWEN_PHOTO_MODEL_PRECISION'],'bf16')
        self.assertEqual(template['env']['QWEN_PHOTO_DOWNLOAD_SAM'],'1')
        self.assertGreaterEqual(template['containerDiskInGb'],150)


class TestQwen21ComfyV3ComboCompatibility(unittest.TestCase):
    """Regression: ComfyUI V3 io.Combo inputs are COMBO widgets in /object_info."""

    @staticmethod
    def validator(relative):
        import importlib.util
        import sys
        from unittest.mock import patch
        path=ROOT/relative
        # Both packages have a build_workflows.py module; import one at a time
        # without leaking an unrelated module's static node schema to the other.
        spec=importlib.util.spec_from_file_location('schema_regression_validator', path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'build_workflows': None}):
            # A None sys.modules entry blocks import, so import the file explicitly
            # and temporarily supply the right one under its expected name.
            schema_spec=importlib.util.spec_from_file_location('build_workflows',path.parent/'build_workflows.py')
            schema_module=importlib.util.module_from_spec(schema_spec)
            schema_spec.loader.exec_module(schema_module)
            sys.modules['build_workflows']=schema_module
            spec.loader.exec_module(module)
        return module

    @staticmethod
    def inputs_and_info(style='v3'):
        api={
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'qwen_image_2.1_bf16.safetensors','weight_dtype':'default'}},
            '2':{'class_type':'QwenImage21Cache','inputs':{'model':['1',0],'device':'auto','dtype':'default'}},
        }
        if style=='v3':
            device=['COMBO',{'options':['auto','gpu','cpu','off'],'default':'auto'}]
            dtype=['COMBO',{'options':['default','int8','int4'],'default':'default'}]
        else:
            device=[['auto','gpu','cpu','off'],{'default':'auto'}]
            dtype=[['default','int8','int4'],{'default':'default'}]
        info={
            'UNETLoader':{
                'input':{'required':{
                    'unet_name':[['qwen_image_2.1_bf16.safetensors'],{}],
                    'weight_dtype':[['default','fp8_e4m3fn'],{}],
                }},'output':['MODEL'],
            },
            'QwenImage21Cache':{'input':{'required':{'model':['MODEL',{}],'device':device,'dtype':dtype}},'output':['MODEL']},
        }
        return api,info

    def test_native_v3_combo_device_dtype_pass(self):
        module=self.validator('qwen21_native/scripts/validate_workflows.py')
        api,info=self.inputs_and_info()
        module.check_schema({},api,info)

    def test_torso_v3_combo_device_dtype_pass(self):
        module=self.validator('qwen21_native/addons/torso_lock/scripts/validate_workflows.py')
        api,info=self.inputs_and_info()
        module.check_schema({},api,info)

    def test_legacy_dropdown_unchanged(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info('legacy')
            module.check_schema({},api,info)

    def test_v3_combo_rejects_invalid_device(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            api['2']['inputs']['device']='invented'
            with self.assertRaisesRegex(ValueError,'not in runtime options'):
                module.check_schema({},api,info)

    def test_v3_loadimage_without_options_is_valid(self):
        """On ComfyUI 0.39, a file upload COMBO can advertise no options.

        The UI/API graph can still contain a placeholder file name; the file
        will be uploaded interactively after the ComfyUI server starts.
        """
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            api['3']={'class_type':'LoadImage','inputs':{'image':'scene_not_yet_uploaded.png'}}
            info['LoadImage']={'input':{'required':{
                'image':['COMBO',{'image_upload':True,'image_folder':'input',
                                  'remote':{'route':'/internal/files/input'}}]
            }},'output':['IMAGE','MASK']}
            module.check_schema({},api,info)

    def test_v3_loadimage_with_empty_options_is_valid(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            api['3']={'class_type':'LoadImage','inputs':{'image':'pending.png'}}
            info['LoadImage']={'input':{'required':{
                'image':['COMBO',{'options':[],'image_upload':True}]
            }},'output':['IMAGE','MASK']}
            module.check_schema({},api,info)
            # Explicit strict-file validation must not claim a missing file exists.
            with self.assertRaisesRegex(ValueError,'not in runtime options'):
                module.check_schema({},api,info,check_files=True)

    def test_v3_file_selector_not_yet_available_passes_normal_check(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            # Files/model weights can appear later (RunPod runtime download).
            info['UNETLoader']['input']['required']['unet_name']=['COMBO',{'options':[]}]
            module.check_schema({},api,info)
            with self.assertRaisesRegex(ValueError,'not in runtime options'):
                module.check_schema({},api,info,check_files=True)

    def test_v3_file_selector_requires_nonempty_string(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            info['UNETLoader']['input']['required']['unet_name']=['COMBO',{}]
            for invalid in ('',42,None):
                api['1']['inputs']['unet_name']=invalid
                with self.assertRaisesRegex(ValueError,'invalid file selector value'):
                    module.check_schema({},api,info)

    def test_v3_rejects_malformed_file_combo_options(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            info['UNETLoader']['input']['required']['unet_name']=['COMBO',{'options':{'bad':'schema'}}]
            with self.assertRaisesRegex(ValueError,'malformed COMBO options'):
                module.check_schema({},api,info)

    def test_v3_combo_rejects_missing_options(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            info['QwenImage21Cache']['input']['required']['dtype']=['COMBO',{}]
            with self.assertRaisesRegex(ValueError,'missing COMBO options'):
                module.check_schema({},api,info)

    def test_v3_combo_respects_force_input(self):
        for path in ('qwen21_native/scripts/validate_workflows.py',
                     'qwen21_native/addons/torso_lock/scripts/validate_workflows.py'):
            module=self.validator(path)
            api,info=self.inputs_and_info()
            info['QwenImage21Cache']['input']['required']['device'][1]['forceInput']=True
            with self.assertRaisesRegex(ValueError,'UI widget order differs'):
                module.check_schema({},api,info)

if __name__=='__main__':unittest.main()

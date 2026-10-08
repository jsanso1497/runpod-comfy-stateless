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
        self.assertIn(':unified-v2.1.0',template['imageName'])
        self.assertIn('8188/http',template['ports'])
        self.assertEqual(template['dockerEntrypoint'],[])
        self.assertEqual(template['env']['QWEN_PHOTO_MODEL_PRECISION'],'bf16')
        self.assertEqual(template['env']['QWEN_PHOTO_DOWNLOAD_SAM'],'1')
        self.assertGreaterEqual(template['containerDiskInGb'],150)

if __name__=='__main__':unittest.main()

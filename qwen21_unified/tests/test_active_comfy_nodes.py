"""Offline tests for active ComfyUI vs build-time ComfyUI mismatch."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

FILE=Path(__file__).resolve().parents[1]/'scripts/runtime_prepare.py'
spec=importlib.util.spec_from_file_location('runtime_prepare_active_test', FILE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ActiveComfyTests(unittest.TestCase):
    def test_bridges_pinned_ausboss_without_changing_active_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            opt=root/'opt/ComfyUI/custom_nodes/ComfyUI-AusBoss'
            (opt/'nodes').mkdir(parents=True)
            (opt/'nodes/node_inpaint_crop_stitch.py').write_text('AUSBOSS_NODES_StitchInpaint')
            active=root/'workspace/ComfyUI'
            (active/'custom_nodes').mkdir(parents=True)
            marker=mod.ensure_ausboss_for_active_comfy(active,opt)/mod.REQUIRED_AUSBOSS
            self.assertTrue(marker.is_file())
            self.assertTrue((active/'custom_nodes/ComfyUI-AusBoss').is_symlink())
            self.assertEqual(mod.ensure_ausboss_for_active_comfy(active,opt),active/'custom_nodes/ComfyUI-AusBoss')
    def test_entire_native_torso_install_in_active_workspace(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stage=root/'opt/ComfyUI/custom_nodes/ComfyUI-AusBoss'
            marker=stage/mod.REQUIRED_AUSBOSS
            marker.parent.mkdir(parents=True)
            marker.write_text('AUSBOSS_NODES_StitchInpaint')
            comfy=root/'workspace/ComfyUI'
            qwen=comfy/'comfy_extras/nodes_qwen.py'
            qwen.parent.mkdir(parents=True)
            qwen.write_text('class TextEncodeQwenImage21: pass')
            user=root/'workspace/ComfyUI/user'
            with patch.object(mod,'NATIVE',FILE.parents[2]/'qwen21_native'), patch.object(mod,'AUSBOSS_STAGE',stage):
                self.assertEqual(mod.install_for(comfy,user),38)
            self.assertTrue((comfy/'custom_nodes/ComfyUI-AusBoss').is_symlink())
            self.assertTrue((comfy/'custom_nodes/qwen21_native_tools/__init__.py').is_file())
            self.assertTrue((comfy/'custom_nodes/qwen21_torso_tools/__init__.py').is_file())
            self.assertEqual(len(list((user/'default/workflows/Qwen21 Native v1.1.0').glob('*.json'))),34)
            self.assertEqual(len(list((user/'default/workflows/Qwen21 Torso Lock v1.0.0').glob('*.json'))),4)

    def test_existing_incompatible_plugin_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            active=root/'ComfyUI'
            bad=active/'custom_nodes/ComfyUI-AusBoss'
            bad.mkdir(parents=True)
            (bad/'user_edit.txt').write_text('do not overwrite')
            with self.assertRaisesRegex(RuntimeError,'incomplete/incompatible'):
                mod.ensure_ausboss_for_active_comfy(active,root/'no-stage')
            self.assertEqual((bad/'user_edit.txt').read_text(),'do not overwrite')
    def test_missing_staged_package_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Pinned AusBoss checkout'):
                mod.ensure_ausboss_for_active_comfy(root/'active',root/'staging-not-present')
            self.assertFalse((root/'active/custom_nodes/ComfyUI-AusBoss').exists())
    def test_no_copy_when_active_already_has_ausboss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            active=root/'workspace/ComfyUI'
            marker=active/'custom_nodes/ComfyUI-AusBoss'/mod.REQUIRED_AUSBOSS
            marker.parent.mkdir(parents=True)
            marker.write_text('AUSBOSS_NODES_StitchInpaint')
            self.assertEqual(mod.ensure_ausboss_for_active_comfy(active,root/'missing'),marker.parents[1])
            self.assertFalse(marker.parents[1].is_symlink())

if __name__=='__main__': unittest.main()

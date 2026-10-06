import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch

MODULE = Path(__file__).resolve().parents[1] / 'service.py'
spec = importlib.util.spec_from_file_location('file_manager_service', MODULE)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class FileManagerTests(unittest.TestCase):
    def test_empty_password_refused(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'FILEBROWSER_PASSWORD'):
            service.password_from_environment()
    def test_short_password_refused(self):
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'short'}), self.assertRaises(ValueError):
            service.password_from_environment()
    def test_unresolved_secret_refused(self):
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'{{ RUNPOD_SECRET_filebrowser_password }}'}), self.assertRaisesRegex(ValueError, 'placeholder'):
            service.password_from_environment()
    def test_password_not_stripped(self):
        value = ' a strong secret with spaces '
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':value}):
            self.assertEqual(service.password_from_environment(), value)
    def test_multiline_refused(self):
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'a long password\nsecond line'}), self.assertRaises(ValueError):
            service.password_from_environment()
    def test_nul_refused(self):
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'a long secret word'}):
            with patch.object(service.os, 'environ', {'FILEBROWSER_PASSWORD':'a long password\0z'}), self.assertRaises(ValueError):
                service.password_from_environment()
    def test_default_port(self): self.assertEqual(service.checked_port('8888'), 8888)
    def test_reserved_ports_refused(self):
        for p in (8188,11434,0,65536,'not-a-port'):
            with self.subTest(port=p), self.assertRaises(ValueError):service.checked_port(p)
    def test_private_state_not_in_workspace(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'Test-Only-Pass-123!'}):
            root=Path(t)
            with self.assertRaisesRegex(ValueError, 'outside'):service.write_config(root, root/'config',8888)
    def test_hashed_auth_config(self):
        from jupyter_server.auth import passwd_check
        secret='Test-Only-Pass-123!'
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':secret}):
            root=Path(t)/'workspace';state=Path(t)/'private'
            config=service.write_config(root,state,8888)
            text=config.read_text();data=json.loads(text)
            self.assertNotIn(secret,text)
            self.assertTrue(passwd_check(data['PasswordIdentityProvider']['hashed_password'], secret))
            self.assertTrue(data['PasswordIdentityProvider']['password_required'])
            self.assertFalse(data['PasswordIdentityProvider']['allow_password_change'])
            self.assertEqual(data['IdentityProvider']['token'],'')
            self.assertEqual(data['ServerApp']['port_retries'],0)
            self.assertFalse(data['ServerApp']['disable_check_xsrf'])
            self.assertEqual(data['ServerApp']['allow_origin'],'')
            self.assertTrue(data['ServerApp']['trust_xheaders'])
            self.assertEqual(stat.S_IMODE(config.stat().st_mode),0o600)
            self.assertEqual(stat.S_IMODE(state.stat().st_mode),0o700)
    def test_env_redaction(self):
        with patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'secret','HF_TOKEN':'secret','CIVITAI_TOKEN':'secret','RUNPOD_API_KEY':'secret','PYTHONPATH':'/somewhere','PATH':'/usr/bin'},clear=True):
            env=service.safe_environment(Path('/run/test'))
            for k in ('FILEBROWSER_PASSWORD','HF_TOKEN','CIVITAI_TOKEN','RUNPOD_API_KEY','PYTHONPATH'):
                self.assertNotIn(k,env)
            self.assertEqual(env['PATH'],'/usr/bin')
    def test_no_auth_ready_rejected(self):
        with patch.object(service, 'get_status', side_effect=[200,200]):self.assertFalse(service.ready(8888))
    def test_auth_ready_accepted(self):
        with patch.object(service, 'get_status', side_effect=[200,403]):self.assertTrue(service.ready(8888))
    def test_missing_app_refused(self):
        with self.assertRaisesRegex(ValueError,'application'):service.supervise(['/no-such-script'])


    def test_supervisor_missing_password_starts_nothing(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(service.subprocess,'Popen') as start:
            with self.assertRaises(ValueError):service.supervise(['/bin/true'])
            start.assert_not_called()
    def test_supervisor_preserves_model_tokens_but_not_browser_password(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'Test-Only-Pass-123!','HF_TOKEN':'keep-for-models','FILEBROWSER_PORT':'8888'},clear=True), patch.object(service,'STATE_ROOT',Path(t)), patch.object(service.signal,'signal'), patch.object(service,'ready',return_value=True), patch.object(service,'stop_process') as stop:
            browser=MagicMock();browser.poll.return_value=None
            app=MagicMock();app.poll.return_value=0
            with patch.object(service.subprocess,'Popen',side_effect=[browser,app]) as start:
                self.assertEqual(service.supervise(['/bin/true']),0)
                env=start.call_args_list[1].kwargs['env']
                self.assertNotIn('FILEBROWSER_PASSWORD',env)
                self.assertEqual(env['HF_TOKEN'],'keep-for-models')
                self.assertEqual(stop.call_count,2)
    def test_browser_failure_prevents_application_start(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'Test-Only-Pass-123!','FILEBROWSER_PORT':'8888'},clear=True), patch.object(service,'STATE_ROOT',Path(t)), patch.object(service.signal,'signal'), patch.object(service,'stop_process'):
            browser=MagicMock();browser.poll.return_value=1
            with patch.object(service.subprocess,'Popen',return_value=browser) as start:
                with self.assertRaisesRegex(RuntimeError,'model downloads were not started'):service.supervise(['/bin/true'])
                self.assertEqual(start.call_count,1)
    def test_application_failure_stops_browser(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'Test-Only-Pass-123!','FILEBROWSER_PORT':'8888'},clear=True), patch.object(service,'STATE_ROOT',Path(t)), patch.object(service.signal,'signal'), patch.object(service,'ready',return_value=True), patch.object(service,'stop_process') as stop:
            browser=MagicMock();browser.poll.return_value=None
            app=MagicMock();app.poll.return_value=2
            with patch.object(service.subprocess,'Popen',side_effect=[browser,app]):
                self.assertEqual(service.supervise(['/bin/true']),2)
                stop.assert_any_call(browser)
    def test_runtime_browser_failure_stops_application(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ, {'FILEBROWSER_PASSWORD':'Test-Only-Pass-123!','FILEBROWSER_PORT':'8888'},clear=True), patch.object(service,'STATE_ROOT',Path(t)), patch.object(service.signal,'signal'), patch.object(service,'ready',return_value=True), patch.object(service,'stop_process') as stop:
            browser=MagicMock();browser.poll.side_effect=[None,1]
            app=MagicMock();app.poll.return_value=None
            with patch.object(service.subprocess,'Popen',side_effect=[browser,app]):
                self.assertEqual(service.supervise(['/bin/true']),1)
                stop.assert_any_call(app)


if __name__=='__main__':unittest.main()

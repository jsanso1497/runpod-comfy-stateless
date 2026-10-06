"""Real loopback HTTP with a fake Ollama service; no model downloads or inference."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

service=load(ROOT/'ollama_service.py','protocol_service')
runtime=load(ROOT/'node/local_runtime.py','protocol_runtime')
MODEL='test/local-vision:instruct'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.state={'installed':False,'running':False,'posts':[]}
        state=self.state
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,obj,ndjson=False):
                data=(b''.join(json.dumps(x).encode()+b'\n' for x in obj) if ndjson else json.dumps(obj).encode())
                self.send_response(200);self.send_header('Content-Type','application/x-ndjson' if ndjson else 'application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_GET(self):
                if self.path=='/api/tags': self.send({'models':[{'name':MODEL}] if state['installed'] else []})
                elif self.path=='/api/ps':self.send({'models':[{'name':MODEL}] if state['running'] else []})
                else:self.send({'version':'test-fixture'})
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                state['posts'].append((self.path,data))
                if self.path=='/api/pull':
                    state['installed']=True
                    self.send([{'status':'pulling','completed':91,'total':100},{'status':'success'}],ndjson=True)
                elif self.path=='/api/show':self.send({'capabilities':['vision']})
                elif self.path=='/api/generate':
                    if data.get('keep_alive')==0:state['running']=False
                    self.send({'done':True})
                else:self.send({})
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)

    def test_actual_streamed_http_pull_and_cached_second_start(self):
        with patch.object(service.oc,'BASE',self.base),service.oc.session() as s:
            service.pull_model(s,MODEL,emit=lambda *a,**k:None)
            service.pull_model(s,MODEL,emit=lambda *a,**k:None)
        self.assertEqual(sum(path=='/api/pull' for path,_ in self.state['posts']),1)

    def test_actual_model_query_and_gpu_handoff_protocol(self):
        self.state['installed']=True;state=self.state
        def original(prompt_provider='local',api_base='http://127.0.0.1:11434/v1',local_model_slug=MODEL):
            state['running']=True
            return 'synthetic prompt'
        with patch.object(runtime,'BASE',self.base),patch.object(runtime,'release_comfy'):
            result=runtime.guard_refpack(original)()
            runtime.require_idle()
        self.assertEqual(result,'synthetic prompt');self.assertFalse(state['running'])
        self.assertIn(('/api/generate',{'model':MODEL,'stream':False,'keep_alive':0}),state['posts'])

    def test_real_http_empty_tag_list_blocks_the_writer(self):
        with patch.object(runtime,'BASE',self.base),runtime.session() as s:
            with self.assertRaisesRegex(RuntimeError,'not ready'):runtime.ensure_installed(s,MODEL)
        self.assertEqual(self.state['posts'],[])


if __name__=='__main__':unittest.main()

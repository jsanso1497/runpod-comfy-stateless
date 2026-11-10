"""Run in the built image on CPU. No model weights or GPU inference are involved."""
import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import urllib.request

PACKAGE = Path(__file__).resolve().parent
COMFY = Path('/opt/ComfyUI')


def configure_cpu_imports():
    """Select CPU in this test process before importing native ComfyUI modules.

    main.py opts in to CLI parsing itself. Direct imports do not: ComfyUI's
    default is to parse an empty argument list. Setting sys.argv alone therefore
    leaves args.cpu false and can initialize CUDA on a CPU-only CI runner.
    This helper is test-only; it never changes the workspace GPU launcher.
    """
    if 'comfy.model_management' in sys.modules:
        raise RuntimeError('CPU smoke bootstrap must run before comfy.model_management is imported.')
    sys.path.insert(0, str(COMFY))
    sys.argv = ['native-smoke', '--cpu', '--cache-none', '--disable-pinned-memory']
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    if not args.cpu:
        raise RuntimeError('CPU smoke arguments were already cached without --cpu. Run in a fresh Python process.')
    return args


def main():
    command = [sys.executable, str(COMFY/'main.py'), '--cpu', '--listen', '127.0.0.1', '--port', '8189',
               '--enable-manager', '--cache-none', '--disable-pinned-memory', '--disable-auto-launch']
    report = {'gpu_generation_tested': False, 'model_weights_downloaded': False}
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp)/'comfy-cpu.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(command,cwd=str(COMFY),stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        raise RuntimeError('CPU ComfyUI stopped:\n'+log_path.read_text()[-8000:])
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8189/object_info',timeout=3) as r:info=json.load(r)
                        break
                    except OSError:time.sleep(1)
                else:raise RuntimeError('CPU startup timeout:\n'+log_path.read_text()[-8000:])
                required=['MiniMaxH3ImageConditioning','H3ImageManualConditioning','H3ImageOptionalReference',
                          'H3ImageSixStepSchedule','H3ImageReport','H3ImageTimedSampler','H3ImageTimedDecode','H3ImageCompare','UNETLoader','CLIPLoader','VAELoader',
                          'SamplerCustomAdvanced','BasicGuider','RandomNoise','VAEDecode','SaveImage']
                missing=[n for n in required if n not in info]
                if missing:raise RuntimeError('Missing native/helper nodes: '+str(missing))
                for graph in (PACKAGE/'api_templates').glob('*.json'):
                    for entry in json.loads(graph.read_text()).values():
                        schema=info[entry['class_type']]['input']
                        allowed=set(schema.get('required',{}))|set(schema.get('optional',{}))
                        invalid=set(entry['inputs'])-allowed
                        if invalid:raise RuntimeError('%s has unknown inputs %s'%(entry['class_type'],invalid))
                report['all_workflow_input_names_match_native_schemas']=True
                report['manager_enabled_cpu_server_started']=True
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    # Exercise the real native conditioner with fake encoder/VAE objects, not fake geometry functions.
    configure_cpu_imports()
    import torch
    from comfy import model_management
    device = model_management.get_torch_device()
    if device.type != 'cpu':
        raise RuntimeError('Native CPU smoke selected a non-CPU device: ' + str(device))
    report.update(in_process_cpu_mode=True, native_test_device=str(device))
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning
    spec=importlib.util.spec_from_file_location('image_workspace',PACKAGE/'nodes/H3ImageWorkspace/__init__.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    class Clip:
        def tokenize(self,prompt,**kwargs):self.prompt=prompt;self.args=kwargs;return {'test':'tokens'}
        def encode_from_tokens_scheduled(self,tokens):return [[torch.zeros(1,1,1),{}]]
    class VAE:
        def encode(self,image):return torch.zeros(1,24,1,image.shape[1]//16,image.shape[2]//16)
    clip=Clip();vae=VAE();node=helpers.H3ImageManualConditioning()
    for count in (0,1,4):
        refs={'reference_%s'%(i+1):{'image':torch.zeros(1,32,32,3),'file':'test%s.png'%i} for i in range(count)}
        prompt='  Literal manual prompt {not expanded}.  '
        cond,latent,text=node.encode(clip,vae,prompt,'1:1',4.,False,count,**refs)
        assert clip.prompt==prompt
        assert tuple(latent['samples'].shape)==(1,24,1,128,128)
        assert json.loads(text)['reference_count']==count
        assert len(cond[0][1].get('minimax_refs',[]))==count
        for row in json.loads(text)['references']:
            assert row['vision_size'] == [1024,1024]
            assert row['vae_size'] == [1024,1024]
    from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling
    class Config:
        unet_config={'pdd_sigmas':helpers.SIGMAS}
        sampling_settings={}
    actual=MiniMaxH3ImageModelSampling(Config())
    class Model:
        def get_model_object(self,name):return actual
    sampler,sigmas=helpers.H3ImageSixStepSchedule().prepare(Model())
    assert len(sigmas)==7
    report.update(native_conditioner_counts_tested=[0,1,4],literal_prompt_preserved=True,
                  real_native_sampling_grid_matches_uploaded_schedule=True)
    (PACKAGE/'native-smoke-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Real FFmpeg/tensor tests without CUDA, model weights or a ComfyUI import."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile

PACKAGE=Path(__file__).resolve().parent


def main():
    import torch
    import torch.nn.functional as F
    spec=importlib.util.spec_from_file_location('media_cpu_package',PACKAGE/'nodes/studio/__init__.py',submodule_search_locations=[str(PACKAGE/'nodes/studio')])
    studio=importlib.util.module_from_spec(spec);sys.modules[spec.name]=studio;spec.loader.exec_module(studio)
    video=sys.modules[spec.name+'.video_input'];media=sys.modules[spec.name+'.media_nodes']
    # Isolate compositing/routing from Comfy imports. Actual upstream resizing/conditioning has a separate CI gate.
    studio.resize_image=lambda image,w,h: image if tuple(image.shape[1:3])==(h,w) else F.interpolate(image.movedim(-1,1),size=(h,w),mode='bilinear',align_corners=False).movedim(1,-1)
    with tempfile.TemporaryDirectory(prefix='h3-video-test-') as folder:
        root=Path(folder);clip=root/'fixture.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x96:rate=30',
                        '-t','4.5','-c:v','libx264','-pix_fmt','yuv420p','-y',str(clip)],check=True,timeout=30)
        frames,report=video.decode_section(clip,.5,3.1,.5)
        assert frames.shape==(73,96,160,3),frames.shape
        assert report['fps']==24 and report['reference_frames']==73 and report['audio']=='ignored'
        assert not torch.equal(frames[0],frames[-1]),'The decoded reference did not retain multiple moments.'
        source=torch.rand(1,96,160,3)
        mask=torch.zeros(1,96,160);mask[:,20:70,40:120]=1.;mask[:,19,40:120]=.5
        target={'image':source,'mask':mask,'name':'target.png'}
        identity={'image':source*.5,'name':'identity.png'}
        p,w,h=media.H3StudioMediaPlan().prepare('Replace person','2:3',.25,.25,
            'Quality - 50 steps, no Turbo',.25,True,
            video={'frames':frames,'summary':report},reference_1=target,reference_2=identity)
        assert p['conditioning_length']==73 and p['video_count']==1 and p['count']==2
        generated=torch.ones(1,h,w,3)*.2
        result=media.H3StudioMediaFinish().finish(p,generated)[0]
        assert result.shape==source.shape
        outside=(mask==0).unsqueeze(-1).expand_as(source)
        assert torch.equal(result[outside],source[outside]),'Unmasked pixels changed.'
        assert torch.allclose(result[:,25:60,50:100],torch.ones_like(result[:,25:60,50:100])*.2)
        try:
            media.H3StudioMediaPlan().prepare('Replace person','2:3',.25,.25,
                'Quality - 50 steps, no Turbo',.25,True,reference_1={'image':source,'name':'unmasked'},reference_2=identity)
        except ValueError as error:
            assert 'mask' in str(error).lower()
        else:raise AssertionError('Protected replacement accepted an empty mask.')
        output={'real_ffmpeg_decode':True,'normalized_fps':24,'selected_frames':73,
                'moving_video_content_retained':True,'target_mask_required_when_protected':True,
                'unmasked_source_pixels_exactly_preserved':True,'original_canvas_preserved':True,
                'reference_duration_not_truncated_to_5':True,'torch':torch.__version__,
                'resize_in_this_test':'CPU interpolation fixture; upstream Comfy resizing is checked in CI',
                'model_weights_loaded':False,'gpu_inference':False}
        (PACKAGE/'media-cpu-report.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(output,indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Deterministic H3 motion-control graphs and API exports. No model download."""
from __future__ import annotations
from graphlib import TopologicalSorter
import copy
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'minimax_h3_ref2va_bf16.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_bf16.safetensors'
VAE = 'minimax_h3_video_vae_fp16.safetensors'
PATCH = 'minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors'
PROMPT = '''<Picture 1> and <Picture 2> depict the same replacement character. <Picture 1> defines the primary facial identity; <Picture 2> provides a supporting view and body proportions. Use the photographs for identity, not their pose, framing or background.

Replace the main performer in <Video 1> with that character. Follow the supplied pose-control sequence frame by frame: keep the timing, direction, trajectory, body orientation, limb positions, gestures and changes of pose. Do not invent a different performance, pause or camera move. The source video supplies the scene, clothing, lighting, framing, camera movement and interaction with the environment. Retain those attributes while using the replacement character's recognizable identity.

One continuous photorealistic shot with coherent anatomy and scene-matched lighting. Preserve natural facial proportions; do not blend the source performer's face with the replacement. Do not add people, cuts, captions or unrelated actions. No spoken dialogue or music. The control frames are structural guidance, not a visible skeleton to draw in the output.'''


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.api = {}

    def add(self, ident, kind, title, pos, inputs=(), outputs=(), widgets=None, size=(350, 170), color=None):
        named = dict(widgets or {})
        node = {'id': ident, 'type': kind, 'title': title, 'pos': list(pos), 'size': list(size),
                'flags': {}, 'order': len(self.nodes), 'mode': 0,
                'inputs': [{'name': k, 'type': t, 'link': None} for k, t in inputs],
                'outputs': [{'name': k, 'type': t, 'links': []} for k, t in outputs],
                'properties': {'Node name for S&R': kind},
                'widgets_values': list(named.values()), 'widgets_values_named': named}
        if not kind.startswith('WBH3'):
            node['properties'].update(cnr_id='comfy-core', ver='b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c')
        for p in node['inputs']:
            if p['name'] in named:
                p['widget'] = {'name': p['name']}
        if color:
            node['color'], node['bgcolor'] = color
        self.nodes.append(node)
        if kind not in ('MarkdownNote', 'Note'):
            self.api[str(ident)] = {'class_type': kind, 'inputs': {k:v for k,v in named.items() if k not in ('upload', 'control_after_generate')}, '_meta': {'title': title}}
        return node

    def link(self, src, out, dst, inp):
        nodes = {n['id']:n for n in self.nodes}
        a,b = nodes[src], nodes[dst]
        oi = next(i for i,p in enumerate(a['outputs']) if p['name']==out)
        ii = next(i for i,p in enumerate(b['inputs']) if p['name']==inp)
        typ = a['outputs'][oi]['type']
        assert typ == b['inputs'][ii]['type'], (src, dst, typ, b['inputs'][ii]['type'])
        lid = len(self.links)+1
        self.links.append([lid, src, oi, dst, ii, typ])
        a['outputs'][oi]['links'].append(lid)
        assert b['inputs'][ii]['link'] is None
        b['inputs'][ii]['link'] = lid
        self.api[str(dst)]['inputs'][inp] = [str(src), oi]

    def finish(self, preview=False):
        nodes, links, api = copy.deepcopy((self.nodes,self.links,self.api))
        if preview:
            required = {34, 41, 42, 43}
            changed=True
            while changed:
                prev=set(required)
                required.update(l[1] for l in links if l[3] in required)
                changed=prev!=required
            nodes=[n for n in nodes if n['id'] in required or n['type']=='MarkdownNote']
            links=[l for l in links if l[1] in required and l[3] in required]
            api={k:v for k,v in api.items() if int(k) in required}
            valid_links={l[0] for l in links}
            for n in nodes:
                for port in n['outputs']:
                    port['links']=[l for l in port['links'] if l in valid_links]
                if n['id']==90:
                    txt='PREVIEW ONLY: upload a source video, then Run. This graph loads only the small pose/detector models. No H3 diffusion, encoder, ControlNet or character references are needed. Review SOURCE | POSE CONTROL before opening the render graph.\n\nIf the skeleton follows the wrong person, set first_person in Select Performer (-1=largest at first frame; 0,1,... left to right). Avoid shot cuts and heavy occlusion.'
                    n['widgets_values']=[txt]; n['widgets_values_named']={'text':txt}
        deps={n['id']:set() for n in nodes}
        for l in links:deps[l[3]].add(l[1])
        order=list(TopologicalSorter(deps).static_order())
        for n in nodes:n['order']=order.index(n['id'])
        graph={'id':str(uuid.uuid5(uuid.NAMESPACE_URL, 'workbench/h3-motion-2026-10-22/'+str(preview))),
               'revision':0, 'last_node_id':max(n['id'] for n in nodes),
               'last_link_id':max([l[0] for l in links],default=0),
               'nodes':nodes, 'links':links, 'groups':[], 'config':{},
               'extra':{'ds':{'scale':.48,'offset':[150,200]}, 'h3_motion_release':'2026-10-22.1',
                        'gpu_tested':False, 'primary_source':'MiniMax-H3-Fun-Controlnet-Union-2.0'}, 'version':.4}
        return graph,api


def build():
    g=Graph()
    g.add(1,'LoadImage','1. CHARACTER A: primary face',(0,0),outputs=[('IMAGE','IMAGE'),('MASK','MASK')],widgets={'image':'','upload':'image'},size=(360,410))
    g.add(2,'LoadImage','2. CHARACTER B: same person / supporting view',(0,460),outputs=[('IMAGE','IMAGE'),('MASK','MASK')],widgets={'image':'','upload':'image'},size=(360,410))
    g.add(3,'LoadVideo','3. SOURCE VIDEO: original performance',(0,930),inputs=[('file','COMBO')],outputs=[('VIDEO','VIDEO')],widgets={'file':''},size=(360,360))
    g.add(4,'WBH3MotionVideo','4. SOURCE TIMING + AUTO ASPECT',(420,930),inputs=[('video','VIDEO')],outputs=[('source_frames','IMAGE'),('timing_report','STRING')],widgets={'start_seconds':0.0,'window_seconds':5.25,'megapixels':1.03},size=(380,220))
    g.add(5,'CheckpointLoaderSimple','SDPose full-body FP16',(420,0),outputs=[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],widgets={'ckpt_name':'sdpose_wholebody_fp16.safetensors'})
    g.add(6,'UNETLoader','Person detector FP16',(420,240),outputs=[('MODEL','MODEL')],widgets={'unet_name':'rt_detr_v4-x-hgnet_fp16.safetensors','weight_dtype':'default'})
    g.add(7,'RTDETR_detect','Detect performers in every frame',(850,240),inputs=[('model','MODEL'),('image','IMAGE')],outputs=[('bboxes','BOUNDING_BOX')],widgets={'threshold':.35,'class_name':'person','max_detections':10})
    g.add(8,'WBH3MotionTrackPerson','5. SELECT PERFORMER: -1 = largest',(850,500),inputs=[('bboxes','BOUNDING_BOX'),('source_frames','IMAGE')],outputs=[('selected_person_boxes','BOUNDING_BOX'),('tracking_report','STRING')],widgets={'first_person':-1,'max_detection_gap':6},size=(390,210))
    g.add(9,'SDPoseKeypointExtractor','Extract pose: every source frame',(1280,0),inputs=[('model','MODEL'),('vae','VAE'),('image','IMAGE'),('bboxes','BOUNDING_BOX')],outputs=[('keypoints','POSE_KEYPOINT')],widgets={'batch_size':1},size=(360,190))
    g.add(10,'SDPoseDrawKeypoints','BODY + HANDS; facial landmarks OFF',(1280,270),inputs=[('keypoints','POSE_KEYPOINT')],outputs=[('IMAGE','IMAGE')],widgets={'draw_body':True,'draw_hands':True,'draw_face':False,'draw_feet':False,'stick_width':4,'face_point_size':3,'score_threshold':.3,'draw_head':True},size=(360,330))
    g.add(11,'WBH3MotionReady','6. MOTION CHECK / RENDER',(1700,0),inputs=[('source_frames','IMAGE'),('pose_frames','IMAGE'),('keypoints','POSE_KEYPOINT')],outputs=[('source_frames','IMAGE'),('control_frames','IMAGE'),('width','INT'),('height','INT'),('length','INT'),('h3_model','COMBO'),('h3_encoder','COMBO'),('h3_vae','COMBO'),('controlnet_model','COMBO')],widgets={'mode':'Render video'},size=(400,350))
    g.add(12,'UNETLoader','Existing H3 Ref2VA BF16',(2180,0),inputs=[('unet_name','COMBO')],outputs=[('MODEL','MODEL')],widgets={'unet_name':MODEL,'weight_dtype':'default'})
    g.add(13,'CLIPLoader','Existing H3 encoder BF16: NOT Ollama',(2180,220),inputs=[('clip_name','COMBO')],outputs=[('CLIP','CLIP')],widgets={'clip_name':ENCODER,'type':'minimax','device':'default'})
    g.add(14,'VAELoader','Existing H3 video VAE',(2180,440),inputs=[('vae_name','COMBO')],outputs=[('VAE','VAE')],widgets={'vae_name':VAE})
    g.add(15,'ModelPatchLoader','NEW: Fun ControlNet Union 2.0 BF16',(2180,660),inputs=[('name','COMBO')],outputs=[('MODEL_PATCH','MODEL_PATCH')],widgets={'name':PATCH},size=(440,130))
    g.add(16,'WBH3MotionCheckPatch','Verify all 10 control injection layers',(2660,660),inputs=[('model_patch','MODEL_PATCH')],outputs=[('MODEL_PATCH','MODEL_PATCH')],size=(360,100))
    g.add(17,'MiniMaxH3ReferenceToVideo','7. DIRECT PROMPT + CHARACTER REFERENCES',(2660,0),inputs=[('clip','CLIP'),('vae','VAE'),('audio_vae','VAE'),('width','INT'),('height','INT'),('length','INT'),('ref_images.ref_image_0','IMAGE'),('ref_images.ref_image_1','IMAGE'),('ref_images.ref_image_2','IMAGE'),('ref_videos.ref_video_0','IMAGE'),('ref_videos.ref_video_1','IMAGE')],outputs=[('positive','CONDITIONING'),('LATENT','LATENT')],widgets={'prompt':PROMPT,'width':1344,'height':768,'length':124,'ref_image_size':'max'},size=(650,570))
    g.add(18,'MiniMaxH3SigmaShift','Native H3 shifts',(2660,850),inputs=[('model','MODEL')],outputs=[('MODEL','MODEL')],widgets={'shift_video':12.,'shift_audio':3.})
    g.add(19,'MiniMaxH3FunControlNetApply','8. MOTION CONTROL STRENGTH',(3360,650),inputs=[('model','MODEL'),('model_patch','MODEL_PATCH'),('vae','VAE'),('control_video','IMAGE'),('mask','MASK'),('source_video','IMAGE')],outputs=[('MODEL','MODEL')],widgets={'strength':1.0,'start_percent':0.0,'end_percent':1.0},size=(420,240))
    g.add(20,'BasicGuider','Native guidance = 1',(3360,0),inputs=[('model','MODEL'),('conditioning','CONDITIONING')],outputs=[('GUIDER','GUIDER')])
    g.add(21,'BasicScheduler','40 steps; no Turbo',(3360,250),inputs=[('model','MODEL')],outputs=[('SIGMAS','SIGMAS')],widgets={'scheduler':'beta','steps':40,'denoise':1.0})
    g.add(22,'RandomNoise','Fixed seed for comparisons',(3830,0),outputs=[('NOISE','NOISE')],widgets={'noise_seed':42,'control_after_generate':'fixed'})
    g.add(23,'KSamplerSelect','Full-step sampler',(3830,240),outputs=[('SAMPLER','SAMPLER')],widgets={'sampler_name':'res_multistep'})
    g.add(24,'SamplerCustomAdvanced','H3: pose-controlled sampling',(4280,0),inputs=[('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')],outputs=[('output','LATENT'),('denoised_output','LATENT')],size=(380,220))
    g.add(25,'VAEDecode','Decode generated frames',(4280,320),inputs=[('samples','LATENT'),('vae','VAE')],outputs=[('IMAGE','IMAGE')])
    g.add(26,'WBH3LosslessMaster','LOSSLESS RGB16 MASTER',(4740,0),inputs=[('images','IMAGE')],outputs=[('images','IMAGE'),('master_file','STRING')],widgets={'fps':24.,'filename_prefix':'video/H3_Motion_Control/native/H3_motion'},size=(430,180))
    g.add(27,'CreateVideo','Silent video at source-time 24 fps',(4740,260),inputs=[('images','IMAGE'),('audio','AUDIO')],outputs=[('VIDEO','VIDEO')],widgets={'fps':24.,'bit_depth':8,'color_space':'sRGB','codec':'none'})
    g.add(28,'SaveVideo','9. RESULT: native motion-controlled MP4',(5200,0),inputs=[('video','VIDEO')],outputs=[('video','VIDEO')],widgets={'filename_prefix':'video/H3_Motion_Control/H3_motion','format':'mp4','format.codec':'auto'},size=(620,500))
    g.add(30,'WBH3MotionCompare','SOURCE | POSE: inspect this before rendering',(1280,720),inputs=[('source','IMAGE'),('other','IMAGE')],outputs=[('IMAGE','IMAGE')],widgets={'label':'POSE CONTROL'})
    g.add(31,'CreateVideo','Pose check: 24 fps',(1700,720),inputs=[('images','IMAGE'),('audio','AUDIO')],outputs=[('VIDEO','VIDEO')],widgets={'fps':24.,'bit_depth':8,'color_space':'sRGB','codec':'none'})
    g.add(34,'SaveVideo','CONTROL CHECK: same source frames and timing',(2180,1150),inputs=[('video','VIDEO')],outputs=[('video','VIDEO')],widgets={'filename_prefix':'video/H3_Motion_Control/checks/Source_vs_Pose','format':'mp4','format.codec':'auto'},size=(680,470))
    g.add(35,'WBH3MotionCompare','SOURCE | GENERATED: motion comparison',(4280,620),inputs=[('source','IMAGE'),('other','IMAGE')],outputs=[('IMAGE','IMAGE')],widgets={'label':'GENERATED'})
    g.add(36,'CreateVideo','Output comparison: 24 fps',(4740,620),inputs=[('images','IMAGE'),('audio','AUDIO')],outputs=[('VIDEO','VIDEO')],widgets={'fps':24.,'bit_depth':8,'color_space':'sRGB','codec':'none'})
    g.add(37,'SaveVideo','OUTPUT CHECK: synchronized side by side',(5200,620),inputs=[('video','VIDEO')],outputs=[('video','VIDEO')],widgets={'filename_prefix':'video/H3_Motion_Control/checks/Source_vs_Generated','format':'mp4','format.codec':'auto'},size=(620,450))
    g.add(41,'PreviewAny','Exact source timing and auto dimensions',(0,1400),inputs=[('source','STRING')],outputs=[('STRING','STRING')],size=(570,400))
    g.add(42,'PreviewAny','Performer-tracking report',(850,1000),inputs=[('source','STRING')],outputs=[('STRING','STRING')],size=(400,310))
    g.add(43,'DrawBBoxes','Selected performer overlay',(850,1400),inputs=[('image','IMAGE'),('bboxes','BOUNDING_BOX')],outputs=[('out_image','IMAGE')],size=(380,100))
    g.add(90,'MarkdownNote','START HERE: corrected motion control',(0,-500),widgets={'text':'Load two photos of the SAME replacement character and the source MP4. Set source start/window, then run. Source aspect is automatic. This graph uses native Fun ControlNet Union 2.0 with per-frame SDPose body/hands, not just a video reference.\n\nNo Ollama or Reference Pack. No Turbo. Existing full BF16 H3 weights. First run the separate PREVIEW graph, or switch node 11 to Preview motion only. All large H3 loaders depend on that gate. Inspect Source_vs_Pose: broken skeletons or the wrong performer cannot be repaired by a stronger prompt.\n\nSource/ref/control/output all share one exact 24-fps frame count. Default 5.25-second window produces 124 frames when available (5.167 seconds); a 15-second source yields 345 frames (14.375 seconds). A source/window of at least 15.084 seconds can supply 362 frames (15.083 seconds). No missing tail is invented.'},size=(810,390))
    g.add(91,'MarkdownNote','QUALITY / LIMITS',(900,-500),widgets={'text':'Default control strength 1.0 for all denoising steps, 40-step res_multistep/beta, guidance 1 via BasicGuider. If the pose preview is correct but output still drifts, compare the SAME seed at strength 1.15, then 1.3. Higher strength is not automatically better and can distort identity.\n\nHands and body are controlled; dense face landmarks are OFF to reduce source-face transfer. The original video remains a scene/camera/clothing reference. This is NOT a pixel-locked background replacement or exact optical-flow retiming. Source identity, occlusions, fast cuts and invisible limbs can still cause errors.\n\nFull BF16 plus ControlNet on an A100 80 GB needs offloading and remains unbenchmarked. Preprocessing models are unloaded before H3. Keep the 1 MP native canvas and short default clip first. Check generated face AND motion. Rebuild the H3 image and prepare HMOTION models before loading this JSON.'},size=(810,390))
    links=[(3,'VIDEO',4,'video'),(6,'MODEL',7,'model'),(4,'source_frames',7,'image'),(7,'bboxes',8,'bboxes'),(4,'source_frames',8,'source_frames'),(5,'MODEL',9,'model'),(5,'VAE',9,'vae'),(4,'source_frames',9,'image'),(8,'selected_person_boxes',9,'bboxes'),(9,'keypoints',10,'keypoints'),(4,'source_frames',11,'source_frames'),(10,'IMAGE',11,'pose_frames'),(9,'keypoints',11,'keypoints'),(11,'h3_model',12,'unet_name'),(11,'h3_encoder',13,'clip_name'),(11,'h3_vae',14,'vae_name'),(11,'controlnet_model',15,'name'),(15,'MODEL_PATCH',16,'model_patch'),(13,'CLIP',17,'clip'),(14,'VAE',17,'vae'),(11,'width',17,'width'),(11,'height',17,'height'),(11,'length',17,'length'),(1,'IMAGE',17,'ref_images.ref_image_0'),(2,'IMAGE',17,'ref_images.ref_image_1'),(11,'source_frames',17,'ref_videos.ref_video_0'),(12,'MODEL',18,'model'),(18,'MODEL',19,'model'),(16,'MODEL_PATCH',19,'model_patch'),(14,'VAE',19,'vae'),(11,'control_frames',19,'control_video'),(19,'MODEL',20,'model'),(17,'positive',20,'conditioning'),(19,'MODEL',21,'model'),(22,'NOISE',24,'noise'),(20,'GUIDER',24,'guider'),(23,'SAMPLER',24,'sampler'),(21,'SIGMAS',24,'sigmas'),(17,'LATENT',24,'latent_image'),(24,'output',25,'samples'),(14,'VAE',25,'vae'),(25,'IMAGE',26,'images'),(26,'images',27,'images'),(27,'VIDEO',28,'video'),(4,'source_frames',30,'source'),(10,'IMAGE',30,'other'),(30,'IMAGE',31,'images'),(31,'VIDEO',34,'video'),(4,'source_frames',35,'source'),(25,'IMAGE',35,'other'),(35,'IMAGE',36,'images'),(36,'VIDEO',37,'video'),(4,'timing_report',41,'source'),(8,'tracking_report',42,'source'),(4,'source_frames',43,'image'),(8,'selected_person_boxes',43,'bboxes')]
    for link in links:g.link(*link)
    return g


def write(root=ROOT):
    g=build()
    files=[]
    for preview,stem in [(True,'00_H3_Motion_Preview'),(False,'01_H3_Character_Swap_Pose_Control')]:
        graph,api=g.finish(preview)
        for folder,ext,value in [('workflows/h3/HMOTION_Pose_Control','.json',graph),('automation/HMOTION','.api.json',api)]:
            path=Path(root)/folder/(stem+ext)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2)+'\n')
            files.append(str(path))
    return files

if __name__=='__main__':
    print('\n'.join(write()))

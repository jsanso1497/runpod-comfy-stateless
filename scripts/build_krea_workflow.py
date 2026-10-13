#!/usr/bin/env python3
"""Rebuild the editable K01 graph and matching API export. No ComfyUI import needed."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UI='workflows/krea/K01/Krea_Edit_Multi_Reference_HQ.json'
API='automation/K01/Krea_Edit_Multi_Reference_HQ.api.json'

def build():
 graph={'last_node_id':15,'last_link_id':0,'nodes':[],'links':[],'groups':[], 'config':{},'extra':{'ds':{'scale':0.65,'offset':[100,100]}},'version':0.4}
 api={}
 def node(id,kind,title,position,size,inputs,outputs,values,api_values=None):
  ports=[{'name':n,'type':t,'link':None} for n,t in inputs]
  n=dict(id=id,type=kind,title=title,pos=position,size=size,flags={},order=id-1,mode=0,inputs=ports,
   outputs=[{'name':n,'type':t,'links':[]} for n,t in outputs],properties={'Node name for S&R':kind},widgets_values=values)
  graph['nodes'].append(n)
  if kind!='Note':api[str(id)]={'class_type':kind,'inputs':api_values or {},'_meta':{'title':title}}
  return n
 node(1,'LoadImage','1. SOURCE photo to edit',[30,80],[350,350],[],[('IMAGE','IMAGE'),('MASK','MASK')],['','image'],{'image':''})
 for i in range(4):
  role=['Face and identity of the replacement subject','Body proportions of the same subject','Additional view of the same subject','Clothing or object details only'][i]
  node(2+i,'WBKreaReference',f'Optional reference {i+1}: enable after upload',[440,80+i*240],[340,210],[],[('reference','WB_KREA_REF')],[False,'none',role],{'enabled':False,'image':'none','role':role})
 node(6,'WBKreaPrepare','2. Source + reference roles',[840,80],[340,250],[('source','IMAGE')]+[(f'reference_{i}','WB_KREA_REF') for i in range(1,5)],[('editing_inputs','WB_KREA_PACKAGE'),('reference_sheet_preview','IMAGE')],[1.9],{'megapixels':1.9})
 instruction='Edit the source photo using the enabled references for their assigned roles. Describe the specific change here.'
 node(7,'WBKreaPrompt','3. Edit instructions + Qwen 3.8 27B',[1250,80],[470,700],[('editing_inputs','WB_KREA_PACKAGE')],[('final_edit_instruction','STRING')],[instruction,True,'qwen3.8:27b-bf16','',768],{'instruction':instruction,'enhance':True,'ollama_model':'qwen3.8:27b-bf16','trigger_words':'','max_tokens':768})
 node(8,'WBKreaModels','4. Krea checkpoint + sampling preset',[1790,80],[400,330],[('ready_prompt','STRING')],[('model','MODEL'),('krea_encoder','CLIP'),('vae','VAE'),('steps','INT'),('cfg','FLOAT')],['krea2_raw_bf16.safetensors','qwen3vl_4b_bf16.safetensors','qwen_image_vae.safetensors','Auto',20,3.0],{'checkpoint':'krea2_raw_bf16.safetensors','text_encoder':'qwen3vl_4b_bf16.safetensors','vae_name':'qwen_image_vae.safetensors','sampling_preset':'Auto','custom_steps':20,'custom_cfg':3.0})
 node(9,'WBKreaLoRAs','5. Required edit adapter + your LoRAs',[1790,470],[400,300],[('model','MODEL')],[('MODEL','MODEL')],['krea2_identity_edit_v1_2.safetensors',1.0,'none',1.0,'none',1.0],{'edit_adapter':'krea2_identity_edit_v1_2.safetensors','edit_strength':1.0,'lora_1':'none','strength_1':1.0,'lora_2':'none','strength_2':1.0})
 node(10,'WBKreaCondition','6. Image-grounded FIT conditioning',[2260,80],[390,300],[('model','MODEL'),('clip','CLIP'),('vae','VAE'),('editing_inputs','WB_KREA_PACKAGE'),('prompt','STRING')],[('edit_model','MODEL'),('positive','CONDITIONING'),('grounded_negative','CONDITIONING'),('target_latent','LATENT')],[1.0,4.0,768],{'scene_strength':1.0,'reference_strength':4.0,'grounding_px':768})
 sampler=node(11,'KSampler','7. Generate the edited photograph',[2720,80],[330,360],[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT'),('steps','INT'),('cfg','FLOAT')],[('LATENT','LATENT')],[12345,'randomize',20,3.0,'euler','simple',1.0],{'seed':12345,'sampler_name':'euler','scheduler':'simple','denoise':1.0})
 sampler['inputs'][4]['widget']={'name':'steps'};sampler['inputs'][5]['widget']={'name':'cfg'}
 node(12,'VAEDecode','8. Decode',[2720,510],[330,120],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')],[])
 node(13,'SaveImage','9. Final image',[3130,80],[620,720],[('images','IMAGE')],[],['Krea_Edit'],{'filename_prefix':'Krea_Edit'})
 node(14,'PreviewImage','Reference sheet sent to Krea',[840,400],[340,430],[('images','IMAGE')],[],[])
 node(15,'Note','READ FIRST',[30,500],[350,550],[],[],['KREA 2 EDIT HQ\n\nUpload the source. Enable only the references you need and describe their roles.\n\n0 references: source-only edit.\n1 reference: direct second-image conditioning.\n2-4 references: labeled sheet in the second input. This is NOT native five-image conditioning.\n\nQwen enhancement is ON and sees source + each individual reference. Its BF16 model unloads before Krea generation. The final prompt appears after execution.\n\nCheckpoint: only compatible Krea 2 weights. Auto sets Raw 20 steps / CFG 3 or Turbo 8 / CFG 1. Custom checkpoints require an explicit sampling preset.\n\nKeep the edit adapter enabled. Two extra LoRAs are optional. Put exact trigger words in the enhancer trigger_words field.\n\nOutput follows source aspect ratio, rounded to multiples of 16, capped at 1.9 MP by default; no automatic upscaling. This is full-frame editing, not a masked pixel-preserving compositor.\n\nRequires the krea workspace build. Source/static-tested, not GPU-certified.'])
 byid={n['id']:n for n in graph['nodes']}
 def link(src,slot,dest,name):
  origin=byid[src]['outputs'][slot];ports=byid[dest]['inputs'];idx=next(i for i,p in enumerate(ports) if p['name']==name)
  lid=len(graph['links'])+1
  graph['links'].append([lid,src,slot,dest,idx,origin['type']]);origin['links'].append(lid);ports[idx]['link']=lid
  api[str(dest)]['inputs'][name]=[str(src),slot]
 link(1,0,6,'source')
 for i in range(4):link(2+i,0,6,f'reference_{i+1}')
 link(6,0,7,'editing_inputs');link(6,1,14,'images');link(7,0,8,'ready_prompt')
 link(8,0,9,'model');link(9,0,10,'model');link(8,1,10,'clip');link(8,2,10,'vae');link(6,0,10,'editing_inputs');link(7,0,10,'prompt')
 for slot,name in enumerate(['model','positive','negative','latent_image']):link(10,slot,11,name)
 link(8,3,11,'steps');link(8,4,11,'cfg');link(11,0,12,'samples');link(8,2,12,'vae');link(12,0,13,'images')
 graph['last_link_id']=len(graph['links'])
 return graph,api

UI_STANDARD='workflows/krea/K01/Krea_Standard_No_Identity_LoRA_HQ.json'
API_STANDARD='automation/K01/Krea_Standard_No_Identity_LoRA_HQ.api.json'

def build_standard():
 graph,api=build()
 byid={n['id']:n for n in graph['nodes']}
 instruction='Describe the target photograph you want, using the enabled references only for their assigned appearance details.'
 for id,kind,title in [(7,'WBKreaStandardPrompt','3. Target photograph + visual prompt enhancer'),
                        (9,'WBKreaOptionalLoRAs','5. Optional LoRAs; no Identity Edit adapter'),
                        (10,'WBKreaStandardCondition','6. Native text-only Krea conditioning')]:
  byid[id]['type']=kind;byid[id]['title']=title;byid[id]['properties']['Node name for S&R']=kind
  api[str(id)]['class_type']=kind;api[str(id)]['_meta']['title']=title
 byid[7]['widgets_values'][0]=instruction;api['7']['inputs']['instruction']=instruction
 byid[9]['widgets_values']=['none',1.0,'none',1.0]
 api['9']['inputs']={'model':['8',0],'lora_1':'none','strength_1':1.0,'lora_2':'none','strength_2':1.0}
 byid[10]['inputs']=[port for port in byid[10]['inputs'] if port['name']!='vae']
 byid[10]['widgets_values']=[]
 byid[10]['outputs'][0]['name']='model';byid[10]['outputs'][2]['name']='negative'
 api['10']['inputs']={name:value for name,value in api['10']['inputs'].items() if name in ('model','clip','editing_inputs','prompt')}
 byid[13]['widgets_values']=['Krea_Standard_No_Identity'];api['13']['inputs']['filename_prefix']='Krea_Standard_No_Identity'
 byid[14]['title']='References for prompt analysis only; not sent to Krea'
 byid[15]['widgets_values']=['KREA STANDARD: NO IDENTITY EDIT LORA\n\nThis is text-to-image generation, not source-preserving editing. The prompt enhancer sees your source and references and writes a complete target-image description. Krea itself receives only text. Source dimensions set the output aspect ratio.\n\nTwo optional Krea-compatible LoRAs are available, without the Identity Edit adapter.\n\nTurn enhancement OFF only when providing a complete standalone generation prompt. For source edits and direct reference conditioning, use the Identity Edit workflow instead.\n\nNo GPU validation has been performed.']
 graph['links']=[]
 for node in graph['nodes']:
  for port in node.get('inputs',[]):port['link']=None
  for port in node.get('outputs',[]):port['links']=[]
 for dest,node in api.items():
  for name,value in node['inputs'].items():
   if not (isinstance(value,list) and len(value)==2 and isinstance(value[1],int)):continue
   src,slot=int(value[0]),value[1];ports=byid[int(dest)]['inputs'];idx=next(i for i,p in enumerate(ports) if p['name']==name)
   origin=byid[src]['outputs'][slot];lid=len(graph['links'])+1
   graph['links'].append([lid,src,slot,int(dest),idx,origin['type']]);origin['links'].append(lid);ports[idx]['link']=lid
 graph['last_link_id']=len(graph['links'])
 return graph,api

def main():
 for paths,builder in [((UI,API),build),((UI_STANDARD,API_STANDARD),build_standard)]:
  for path,value in zip(paths,builder()):
   p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
   p.write_text(json.dumps(value,indent=2)+'\n')
   print(path)
if __name__=='__main__':main()

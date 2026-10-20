"""Cabinet recognition requires both masked appearance and RGB context support.

Context is another semantic observation, not a geometric/noise/overmerge gate.
All original masked-region scores and the original decisions remain auditable.
"""
from pathlib import Path
import json,hashlib,copy,shutil,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
from safetensors.torch import load_file

R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019/semantic_route')
S=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019/frontend')
O=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019/overmerge/web/data.json')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def context_box(bbox,shape):
    # One object width/height of context on each side, with an 80 px minimum.
    # A tiny fragment therefore cannot provide only a magnified blurry sliver.
    x0,y0,x1,y1=bbox;px=max(80,x1-x0);py=max(80,y1-y0)
    h,w=shape[:2]
    return [max(0,x0-px),max(0,y0-py),min(w,x1+px),min(h,y1+py)]

def square_image(rgb):
    h,w=rgb.shape[:2];side=max(h,w);out=np.full((side,side,3),127,np.uint8)
    out[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=rgb
    return out

def main():
    started=time.time();original=R/'semantic_mask_only.json'
    if not original.exists():shutil.copyfile(R/'semantic.json',original)
    orig_sha=sha(original);s=read(original);cfg=s['config'];assert 'cabinet_context' not in cfg
    f=read(O);manifest=read(S/'input_manifest.json');jobs={j['key']:j for j in manifest['jobs']};source_inputs={}
    for j in f['jobs']:source_inputs.setdefault(j['sources'][2],jobs[j['key']])
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pc=open_clip.get_pretrained_cfg(cfg['model_name'],cfg['pretrained_tag'])
    model,_,preprocess=open_clip.create_model_and_transforms(cfg['model_name'],pretrained=None,device='cuda',precision='fp16',image_mean=pc['mean'],image_std=pc['std'])
    state=load_file(s['model']['weights'],device='cpu');model.load_state_dict(state,strict=True);del state;model.eval()
    tokenizer=open_clip.get_tokenizer(cfg['model_name'])
    with torch.inference_mode():text=F.normalize(model.encode_text(tokenizer(s['prompts']).cuda()).float(),dim=-1)
    terms=s['terms'];cabinet_indices=[i for i,t in enumerate(terms) if t in cfg['targets']['cabinet']];other_indices=[i for i in range(len(terms)) if i not in cabinet_indices]
    # Freeze the semantic condition before evaluating any context images.
    context_rule=dict(required_for='cabinet only',padding_each_side='max(80 native pixels, bbox width or height)',margin_threshold=0.03,target_terms=cfg['targets']['cabinet'],competing_terms=[terms[i] for i in other_indices],scope='same original ROI RGB input; clipped at input boundary',semantic_observations_required='masked region AND RGB context',no_area_or_noise_cutoff=True)
    save(R/'cabinet_context_rule.json',context_rule)
    candidates=[r for r in s['candidates'].values() if r['hit'] and r['target_family']=='cabinet'];batch=[];metadata=[]
    for r in candidates:
        job=source_inputs[r['source_index']];p=Path(job['input_file']);assert sha(p)==r['RGB_sha256'];rgb=np.asarray(Image.open(p).convert('RGB'));box=context_box(r['bbox'],rgb.shape);x0,y0,x1,y1=box;im=square_image(rgb[y0:y1,x0:x1]);batch.append(preprocess(Image.fromarray(im)));metadata.append(dict(bbox=box,RGB_sha256=r['RGB_sha256'],context_image_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    scores=np.empty((0,len(terms)),np.float32)
    if batch:
        with torch.inference_mode():features=F.normalize(model.encode_image(torch.stack(batch).to(device='cuda',dtype=model.visual.conv1.weight.dtype)).float(),dim=-1);scores=(features@text.T).cpu().numpy()
    accepted=0
    for r,sc,meta in zip(candidates,scores,metadata):
        ci=max(cabinet_indices,key=lambda i:sc[i]);oi=max(other_indices,key=lambda i:sc[i]);margin=float(sc[ci]-sc[oi]);passed=margin>=context_rule['margin_threshold'];accepted+=int(passed)
        r['mask_only_hit']=True;r['cabinet_context']=dict(meta,scores=[float(v) for v in sc],target_term=terms[ci],target_score=float(sc[ci]),control_term=terms[oi],control_score=float(sc[oi]),margin=margin,passed=bool(passed))
        r['hit']=bool(passed);r['decision_reason']='MASK_AND_CONTEXT_CABINET' if passed else 'CABINET_CONTEXT_NOT_SUPPORTED'
        print(r['candidate_id'],r['area'],'masked',round(r['margin'],5),'context',terms[ci],round(float(sc[ci]),5),'vs',terms[oi],round(float(sc[oi]),5),'delta',round(margin,5),'accepted',passed,flush=True)
    s['config']['cabinet_context']=context_rule;s['schema']='semantic-sam-route-v2-context-cabinet';s['stats']['mask_only_hit_cabinet']=len(candidates);s['stats']['hit_cabinet']=accepted;s['stats']['not_hit']+=len(candidates)-accepted;s['stats']['cabinet_context_not_supported']=len(candidates)-accepted;s['source_sha256'][str(original)]=orig_sha
    s['cabinet_refinement']=dict(seconds=round(time.time()-started,2),classification_order='SAM partitions -> image/text semantic recognition (masked region, plus context for cabinet) -> semantic direct route or ordinary pipeline',window_decisions_unchanged=True)
    assert sha(original)==orig_sha
    save(R/'semantic.json',s);print('COMPLETE',json.dumps(s['stats']),flush=True)

if __name__=='__main__':main()

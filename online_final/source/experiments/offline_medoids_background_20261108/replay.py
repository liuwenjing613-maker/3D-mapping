"""Apply the frozen main semantic rule to the 975 accepted medoid representatives."""
import sys,json,hashlib,time,collections,importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
WEB=Path('/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
SEM=Path('/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019/v2/semantic')
sys.path.insert(0,str(WEB));from appdata import scene_data,mask_array
from common import SCENES,BASE
sys.path.insert(0,str(SEM));from rules import summarize_mask,resolve_labels
from run import preview_seed,palette_for,sha

INPUT=Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108')
OUT=Path('/data/chenkejun/CVPR/results/offline_medoids_background_20261108')
VERSION='medoids-v3-background-main-v1-20261108'

def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8');tmp.replace(p)

class Predictor:
 def __init__(self,cfg,reference):
  import torch
  from transformers import SegformerConfig,SegformerImageProcessor,SegformerForSemanticSegmentation
  self.torch=torch;torch.set_num_threads(4);torch.manual_seed(0)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available()
  root=Path(cfg['model_directory']);config=SegformerConfig.from_pretrained(str(root),local_files_only=True)
  self.processor=SegformerImageProcessor.from_pretrained(str(root),local_files_only=True)
  self.model=SegformerForSemanticSegmentation(config)
  state=torch.load(str(root/'pytorch_model.bin'),map_location='cpu',weights_only=True)
  self.model.load_state_dict(state,strict=True);del state
  self.model=self.model.float().eval().cuda()
  assert torch.__version__==reference['torch_version']
  assert importlib.metadata.version('transformers')==reference['transformers_version']
  assert self.processor.to_dict()==reference['processor']
  assert resolve_labels(config.id2label,cfg)[0]=={int(k):v for k,v in reference['id2label'].items()}
  self.checked=False
  self.meta=dict(torch_version=torch.__version__,transformers_version=importlib.metadata.version('transformers'),device=torch.cuda.get_device_name(),precision='float32; TF32 disabled; batch size 1')
 def predict(self,rgb):
  import torch.nn.functional as F
  torch=self.torch;h,w=rgb.shape[:2];inputs=self.processor(images=Image.fromarray(rgb),return_tensors='pt')
  assert tuple(inputs['pixel_values'].shape)==(1,3,640,640)
  with torch.inference_mode():
   logits=self.model(pixel_values=inputs['pixel_values'].cuda()).logits.float()
   up=F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
   prob,ids=up.softmax(dim=1).max(dim=1)
   if not self.checked:
    official=self.processor.post_process_semantic_segmentation(type('O',(),{'logits':logits})(),target_sizes=[(h,w)])[0]
    assert torch.equal(ids[0],official);self.checked=True
   pred=ids[0].cpu().numpy().astype(np.uint8);conf=prob[0].cpu().numpy().astype(np.float32)
  return pred,conf

def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 old={s:json.loads((BASE/s/'v2/semantic/classification.json').read_text()) for s in SCENES}
 cfg=old['room0']['config'];labels,excluded_ids=resolve_labels(old['room0']['model']['id2label'],cfg)
 assert cfg['pixel_confidence']==.7 and cfg['mask_fraction']==.9
 protected={};cache_locks={};summaries=[];parity=[];predictor=None
 for scene in SCENES:
  for key in ['model_revision','weight_sha256','pixel_confidence','mask_fraction','exclude_classes','postprocess','preprocess','precision']:
   assert old[scene]['config'][key]==cfg[key],(scene,key)
  p=BASE/scene/'v2/semantic/classification.json';protected[str(p)]=sha(p)
 for p,h in old['room0']['code_hashes'].items():assert sha(p)==h;protected[p]=h
 root=Path(cfg['model_directory']);downloads=json.loads((root/'download_manifest.json').read_text())
 assert downloads['revision']==cfg['model_revision'] and downloads['model_id']==cfg['model_id']
 for name,entry in downloads['files'].items():
  p=root/name;assert sha(p)==entry['sha256'];protected[str(p)]=entry['sha256']
 assert sha(root/'pytorch_model.bin')==cfg['weight_sha256']
 protected[str(root/'download_manifest.json')]=sha(root/'download_manifest.json')
 medoids={s:json.loads((INPUT/(s+'.json')).read_text()) for s in SCENES}
 for s in SCENES:protected[str(INPUT/(s+'.json'))]=sha(INPUT/(s+'.json'))
 save(OUT/'input_freeze.json',dict(version=VERSION,baseline='main auto-v3-swap / partial-e1-swap-20261021',input_stage='medoids分簇剔除重复',input_version=medoids['room0']['version'],config=cfg,protected=protected,GT_used=False,tracking=False,map_write=False))
 colors=palette_for(labels)
 for scene in SCENES:
  d=scene_data(scene);protected.update(d['locks']);dest=OUT/scene
  (dest/'previews').mkdir(parents=True,exist_ok=True);(dest/'semantic_cache').mkdir(exist_ok=True)
  targets=[]
  for gid,g in medoids[scene]['groups'].items():
   for c in g['clusters']:
    targets.append(dict(group_id=gid,cluster=c['number'],mask_id=c['representative'],original_primary=g['original_primary'],cluster_members=c['members']))
  assert len(targets)==medoids[scene]['summary']['kept'] and len({t['mask_id'] for t in targets})==len(targets)
  by_frame=collections.defaultdict(list)
  for t in targets:by_frame[int(d['raw'][t['mask_id']]['frame'])].append(t)
  old_frames={f['frame']:f for f in old[scene]['frames']};old_records={r['mask_id']:r for r in old[scene]['records']}
  records=[];frames=[];hits=new=0
  for number,(frame,items) in enumerate(sorted(by_frame.items()),1):
   rgb_path=Path(d['rawcfg']['scene_root'])/d['rawcfg']['rgb_pattern'].format(frame=frame)
   rgb_hash=sha(rgb_path);protected[str(rgb_path)]=rgb_hash;rgb=np.asarray(Image.open(rgb_path).convert('RGB'));shape=rgb.shape[:2]
   if frame in old_frames:
    f=old_frames[frame];cache=BASE/scene/'v2/semantic'/f['cache']
    assert f['RGB_sha256']==rgb_hash and f['shape']==list(shape) and sha(cache)==f['cache_sha256']
    meta=json.loads(cache.with_suffix('.json').read_text());assert meta['RGB_sha256']==rgb_hash and meta['npz_sha256']==f['cache_sha256']
    with np.load(cache,allow_pickle=False) as z:pred=z['labels'];conf=z['confidence']
    cache_locks[str(cache)]=f['cache_sha256'];hits+=1;origin='original_main_cache'
   else:
    cache=dest/'semantic_cache'/f'f{frame:06d}.npz';meta_path=cache.with_suffix('.json')
    if cache.exists() and meta_path.exists():
     meta=json.loads(meta_path.read_text());assert meta['RGB_sha256']==rgb_hash and meta['model_revision']==cfg['model_revision'] and sha(cache)==meta['npz_sha256']
     with np.load(cache,allow_pickle=False) as z:pred=z['labels'];conf=z['confidence']
    else:
     if predictor is None:
      predictor=Predictor(cfg,old['room0']['model'])
      # Re-infer one frozen frame to check exact class and confidence-gate agreement.
      ref=old['room0']['frames'][0];ref_rgb=np.asarray(Image.open(ref['RGB_path']).convert('RGB'))
      rp,rc=predictor.predict(ref_rgb)
      with np.load(BASE/'room0/v2/semantic'/ref['cache']) as z:
       assert np.array_equal(rp,z['labels']) and np.array_equal(rc>=.7,z['confidence']>=.7)
       save(OUT/'inference_parity.json',dict(frame=ref['frame'],scene='room0',labels_equal=True,confidence_gate_equal=True,max_confidence_difference=float(np.max(np.abs(rc-z['confidence']))),runtime=predictor.meta))
     pred,conf=predictor.predict(rgb);np.savez_compressed(cache,labels=pred,confidence=conf)
     save(meta_path,dict(RGB_sha256=rgb_hash,model_revision=cfg['model_revision'],shape=list(shape),npz_sha256=sha(cache)))
    new+=1;origin='same_model_new_frame'
   assert pred.shape==conf.shape==shape and pred.dtype==np.uint8 and conf.dtype==np.float32
   cache_hash=sha(cache)
   frames.append(dict(frame=frame,RGB_path=str(rgb_path),RGB_sha256=rgb_hash,cache=str(cache),cache_sha256=cache_hash,origin=origin))
   for t in items:
    mid=t['mask_id'];raw=d['raw'][mid];mask=mask_array(d,mid)
    p=d['seedpaths'][mid];protected[str(p)]=sha(p);assert protected[str(p)]==raw['mask_sha256']
    info=summarize_mask(pred,conf,mask,labels,cfg)
    images,box=preview_seed(rgb,mask,pred,conf,colors,excluded_ids,cfg,dest,mid)
    rec=dict(**t,scene=scene,frame=frame,mask_sha256=raw['mask_sha256'],support=raw['support_count'],images=images,preview_crop_box=box,frame_cache=str(cache),cache_sha256=cache_hash,RGB_sha256=rgb_hash,operational_decision='exclude' if info['exclude'] else 'retain',**info)
    if mid in old_records and old_records[mid]['mask_sha256']==raw['mask_sha256']:
     original=old_records[mid]
     for key in info:assert info[key]==original[key],(scene,mid,key)
     parity.append(dict(scene=scene,mask_id=mid,status='EXACT'))
    rec['categories']=sorted([v for v in info['categories'] if v['pixels']],key=lambda x:(-x['pixels'],x['id']))
    records.append(rec)
   if number%10==0 or number==len(by_frame):print('FRAME',scene,number,'/',len(by_frame),'cached',hits,'new',new,flush=True)
  records.sort(key=lambda r:(r['group_id'],r['cluster'],r['mask_id']))
  stats=dict(scene=scene,input=len(records),excluded=sum(r['exclude'] for r in records),retained=sum(not r['exclude'] for r in records),confident_retained=sum(r['decision']=='retain' for r in records),review_retained=sum(r['decision']=='review' for r in records),unique_frames=len(frames),cached_frames=hits,new_frames=new)
  result=dict(version=VERSION,input_version=medoids[scene]['version'],config=cfg,model=old[scene]['model'],records=records,frames=frames,summary=stats,GT_used=False,stage='background semantic screening only; no tracking, actual repair or evaluation')
  save(dest/'classification.json',result)
  save(dest/'retained_candidates.json',dict(version=VERSION,scene=scene,candidates=[{k:r[k] for k in ['group_id','cluster','mask_id','frame','mask_sha256','support']} for r in records if not r['exclude']],stage='after background semantics only; not final repair seeds'))
  summaries.append(stats);print('SCENE',json.dumps(stats),flush=True)
 assert all(sha(p)==h for p,h in protected.items()) and all(sha(p)==h for p,h in cache_locks.items())
 save(OUT/'summary.json',summaries)
 save(OUT/'audit.json',dict(status='REPLAY_PASS',version=VERSION,input_version=medoids['room0']['version'],input=sum(s['input'] for s in summaries),retained=sum(s['retained'] for s in summaries),excluded=sum(s['excluded'] for s in summaries),source_locks=protected,cache_locks=cache_locks,exact_original_record_matches=parity,code_sha256=sha(__file__),GT_used=False,input_unchanged=True,stage='offline background semantic screening replay only',elapsed_seconds=time.time()-started))
 print('COMPLETE',sum(s['input'] for s in summaries),sum(s['retained'] for s in summaries),sum(s['excluded'] for s in summaries),'exact original matches',len(parity),flush=True)

if __name__=='__main__':main()

from pathlib import Path
import os,sys,json,time,traceback,hashlib,importlib.util,collections,fcntl,subprocess
import numpy as np
from PIL import Image
scene=sys.argv[1];base=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110/online')/scene/'100/B';src=base/'candidates/v1';out=base/'candidates/background070';out.mkdir(parents=True,exist_ok=True);start=time.time()
def write(p,x):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(status,**kw):write(out/'status.json',dict(status=status,scene=scene,stage='BACKGROUND070_ONLY',elapsed_seconds=time.time()-start,tracking_or_repair=False,**kw))
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
from gpu_memory_scheduler import acquire as memory_acquire
def acquire():
 lease=memory_acquire(8000,'background:'+str(base),out);lease.attach(os.getpid());return lease.data['gpu'],lease.uid,lease

locks=[]
try:
 med=json.loads((base/'candidates/medoids/result.json').read_text());data=json.loads((src/'grouping.json').read_text());assert med['status']=='PASS' and med['source_grouping_sha256']==sha(src/'grouping.json');seeds={x['id']:x for x in data['seeds']}
 ref=json.loads((Path('/data/chenkejun/CVPR/results/offline_medoids_background070_20261108')/scene/'classification.json').read_text());cfg=ref['config'];assert cfg['pixel_confidence']==cfg['mask_fraction']==.7
 spec=importlib.util.spec_from_file_location('frozen_semantics','/home/chenkejun/CVPR/experiments/offline_medoids_background_20261108/replay.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);labels,_=mod.resolve_labels(ref['model']['id2label'],cfg)
 config=json.loads((base/'mapping/config.json').read_text());byframe=collections.defaultdict(list)
 for g in med['groups']:
  for cl in g['clusters']:
   s=seeds[cl['representative']];fid=int(s['frame']);assert config['frame_selection']['start']<=fid<config['frame_selection']['stop_exclusive'];byframe[fid].append(dict(group_id=g['group_id'],cluster=cl['number'],mask_id=s['id'],cluster_members=cl['members'],original_primary=g['original_primary']))
 cacheby={int(x['frame']):x for x in ref['frames']}; pred=None;results=[];nreuse=nnew=0;gpu=None
 for n,(fid,items) in enumerate(sorted(byframe.items()),1):
  rgb=Path(config['source']['scene_root'])/config['source']['rgb_pattern'].format(frame=fid);rgbsha=sha(rgb);old=cacheby.get(fid)
  if old and old['RGB_sha256']==rgbsha and Path(old['cache']).is_file() and sha(Path(old['cache']))==old['cache_sha256']:
   cache=Path(old['cache']);nreuse+=1
   with np.load(cache) as z:p=z['labels'];c=z['confidence']
   origin='verified_existing_pixel_cache'
  else:
   if pred is None:
    idx,uid,locks=acquire();os.environ['CUDA_VISIBLE_DEVICES']=uid;gpu={'index':idx,'uuid':uid};import torch;torch.cuda.set_per_process_memory_fraction(6000*1048576/torch.cuda.get_device_properties(0).total_memory);pred=mod.Predictor(cfg,ref['model'])
    a=ref['frames'][0];v,w=pred.predict(np.asarray(Image.open(a['RGB_path']).convert('RGB')))
    with np.load(a['cache']) as z:assert np.array_equal(v,z['labels']) and np.array_equal(w>=.7,z['confidence']>=.7),'semantic model parity failed'
   p,c=pred.predict(np.asarray(Image.open(rgb).convert('RGB')));cache=out/f'f{fid:06d}.npz';np.savez_compressed(cache,labels=p,confidence=c);nnew+=1;origin='new_inference_same_frozen_model'
  chash=sha(cache)
  for item in items:
   s=seeds[item['mask_id']];maskpath=src/s['mask'];assert sha(maskpath)==s['mask_sha256'];mask=np.asarray(Image.open(maskpath))>0;info=mod.summarize_mask(p,c,mask,labels,cfg)
   results.append({**item,'frame':fid,'mask_sha256':s['mask_sha256'],'support':s['support_count'],'cache':str(cache),'cache_sha256':chash,'cache_origin':origin,'rgb_sha256':rgbsha,'decision':info['decision'],'exclude':info['exclude'],'top_class':info['top_class'],'high_excluded_fraction':info['high_excluded_fraction']})
  state('RUNNING',frames=n,total_frames=len(byframe),reused=nreuse,new=nnew,gpu=gpu)
 kept=[r for r in results if not r['exclude']];result={'status':'PASS','stage':'MEDOIDS_BACKGROUND070_ONLY','scene':scene,'checkpoint':300,'records':results,'retained_candidates':kept,'summary':{'input':len(results),'excluded':len(results)-len(kept),'retained':len(kept),'frames':len(byframe),'reused_frames':nreuse,'inferred_frames':nnew},'source_medoids_sha256':sha(base/'candidates/medoids/result.json'),'gpu':gpu,'GT_used':False,'tracking_or_repair':False,'elapsed_seconds':time.time()-start};write(out/'result.json',result);state('PASS',summary=result['summary'],gpu=gpu)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
finally:
 if locks:locks.close()

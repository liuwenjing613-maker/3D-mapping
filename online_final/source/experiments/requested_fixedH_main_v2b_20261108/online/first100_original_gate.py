from pathlib import Path
from dataclasses import replace
import sys,os,json,time,hashlib,traceback,collections
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
root=Path(sys.argv[1]);out=root/'gate_sam3';out.mkdir(parents=True,exist_ok=True);start=time.time()
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019/e1_repair');import bootstrap;import run_batch as b
from instance_repair.cli import load_script
pure=load_script('_replica_online_pure_gate',Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106/main/scene0050_00/track_pure_gate.py'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def state(s,**kw):write(out/'status.json',dict(status=s,elapsed_seconds=time.time()-start,stage='ORIGINAL_MASK_GATE_ONLY',**kw))
try:
 track=json.loads((root/'tracking_sam3/driver_status.json').read_text());assert track['status']=='PASS';sel=json.loads((root/'candidates/fixedH/result.json').read_text());rows=sorted(sel['selected_records'],key=lambda r:r['mask_id']);n=int(root.parts[-2]);assert len(rows)==track['input_seeds']
 B=(root/'mapping').resolve();assert B.is_dir();mc=json.loads((B/'mapping_complete.json').read_text());assert mc['status']=='PASS' and mc['frame_count']==n and not mc['GT_used'] and not mc['future_inputs_used'];ids=list(range(0,n*5,5));maps={};oldmax=0
 for line in (B/'association/associations.jsonl').read_text().splitlines():
  x=json.loads(line);maps[(int(x['frame_id']),int(x['mask_local_id']))]=int(x['instance_id']);oldmax=max(oldmax,int(x['instance_id']))
 class Scene:
  def __init__(self):
   with np.load(B/'baseline/final/instance_surface.npz') as z:self.xyz=z['xyz_m'].copy();self.labels=z['instance_id'].copy()
   self.records={fid:None for fid in ids};self.tree=cKDTree(self.xyz.astype(np.float64));self.source=b.r.ReplicaFrameSource(B/'config.json');prior=json.loads((B/'prior_commit.json').read_text());self.prior_count=int(prior['processed_frames']);self.prior_layers={x['frame_id']:x['layer_files'] for x in json.loads(Path(prior['journal_manifest']).read_text())['frames']};assert sorted(self.prior_layers)==ids[:self.prior_count]
  def family_view(self,fid,frame,families):
   with np.load(B/'baseline/frame_support'/f'f{fid:06d}.npz') as z:present=bool(np.isin(z['candidate_persistent_id'],families).any())
   if fid in self.prior_layers:
    layers=[]
    for row in self.prior_layers[fid]:
     p=Path(row['path']);assert sha(p)==row['sha256'];layers.append(np.asarray(Image.open(p),np.int32))
   else:
    table=np.zeros(int(frame.mask_local.max())+1,np.int32)
    for k in np.unique(frame.mask_local):
     if int(k)>0:table[int(k)]=maps[(fid,int(k))]
    layers=[table[frame.mask_local]]
   return present,np.logical_or.reduce([(mapped>0)&~np.isin(mapped,families) for mapped in layers])
 scene=Scene();oldmax=max(oldmax,int(scene.labels.max()));nextid=oldmax+1;objects={};cases={};canonical={};tracking={};records=[]
 for tid,r in enumerate(rows,1):
  mid=r['mask_id'];fid=int(r['frame']);src=root/'candidates/v1/masks'/f'{mid}.png';assert sha(src)==r['mask_sha256'];mask=np.asarray(Image.open(src))>0;fr=scene.source.load_frame(fid);assert mask.shape==fr.mask_local.shape
  assert fid not in scene.prior_layers,('seed from prior frames requires historical identity mapping',mid,fid);vals,counts=np.unique(fr.mask_local[mask],return_counts=True);hist=collections.Counter()
  for local,count in zip(vals,counts):
   if int(local)>0:hist[maps[(fid,int(local))]]+=int(count)
  assert hist,('no family for original gate',mid);family=min(hist,key=lambda v:(-hist[v],v))
  if r['id_action']=='inherit':pid=int(r['target_id'])
  else:pid=nextid;nextid+=1
  p,t,_=b.r.project_frame_regions(replace(fr,mask_local=np.where(mask,tid,0).astype(np.uint16)),scene.tree,pixel_stride=2,max_distance_m=.015,workers=2);points=np.unique(p[t==tid]);projection_stride=2
  if not len(points):
   p,t,_=b.r.project_frame_regions(replace(fr,mask_local=np.where(mask,tid,0).astype(np.uint16)),scene.tree,pixel_stride=1,max_distance_m=.015,workers=2);points=np.unique(p[t==tid]);projection_stride=1
  assert len(points)>0,('seed has no support even at full pixel resolution',mid)
  sp=out/'seed_support'/f'{mid}.npz';sp.parent.mkdir(exist_ok=True);np.savez_compressed(sp,surface_point_index=p,track_id=t);samples=points[np.linspace(0,len(points)-1,min(len(points),2048),dtype=int)];adapt=out/'labels'/mid;adapt.mkdir(parents=True,exist_ok=True);frames=[]
  for f in ids:
   source=root/'tracking_sam3'/mid/'masks'/f'{f:06d}.png';bitmap=np.asarray(Image.open(source))>0;assert bitmap.shape==mask.shape
   if f==fid:assert np.array_equal(bitmap,mask)
   dst=adapt/f'{f:06d}.png';Image.fromarray(np.where(bitmap,tid,0).astype(np.uint16)).save(dst,compress_level=1);frames.append({'frame':f,'label_file':str(dst),'label_sha256':sha(dst),'pixels':int(bitmap.sum())})
  cases[mid]={'ids':[tid],'config':{'seed_frame':fid,'old_family_ids':[family]},'case':{'case_uid':mid}};canonical[tid]=tid;objects[tid]={'track_id':tid,'case_uid':mid,'canonical_track_id':tid,'points':points,'samples':samples,'bounds':(scene.xyz[points].min(0)-.05,scene.xyz[points].max(0)+.05),'group_id':r['relation_group'],'mask_id':mid,'old_family_id':family,'persistent_id':pid};tracking[mid]={x['frame']:{'label_file':x['label_file']} for x in frames}
  records.append({**r,'track_id':tid,'persistent_id':pid,'old_family_for_gate':family,'old_family_histogram':dict(hist),'support_file':str(sp),'support_sha256':sha(sp),'projected_points':len(points),'seed_projection_stride':projection_stride,'labels':frames})
  if tid%5==0:state('RUNNING',progress='ADAPT_TRACK_LABELS',completed=tid,total=len(rows))
 assert len({x['persistent_id'] for x in records})==len(records) and nextid<65535
 policy=json.loads(Path('/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019/e1_repair/gate_policy.json').read_text());pad=policy['seed_surface_bbox_padding_m']
 for obj in objects.values():obj['bounds']=(scene.xyz[obj['points']].min(0)-pad,scene.xyz[obj['points']].max(0)+pad)
 def progress(i,total,seconds):state('RUNNING',progress='EVALUATE_ORIGINAL_MASK_GATE',frames=i,total=total)
 write(out/'input_records.json',{'status':'PASS','objects':records,'instance_base':max(oldmax,nextid)+1,'GT_used':False,'selection_sha256':sha(root/'candidates/fixedH/result.json')})
 result=pure.evaluate(b,scene,cases,objects,canonical,policy,tracking,out=out,progress=progress);assert result['status']=='PASS' and len(result['frames'])==n;result.update(model='SAM3',GT_used=False,repair_executed=False,gate_policy_sha256=sha(Path('/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019/e1_repair/gate_policy.json')));write(out/'frame_decisions.json',result);state('PASS',frames=n,objects=len(rows),repair_executed=False,GT_used=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise

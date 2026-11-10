from pathlib import Path
import os,sys,json,time,traceback,hashlib
import numpy as np
from scipy.spatial import cKDTree
scene=sys.argv[1];R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/scene;run=R/'400/B';out=R/'400/no_repair_replay';out.mkdir(parents=True,exist_ok=True);status=out/'status.json';start=time.time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def save(p,**x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(q,**x);q.replace(p)
def state(s,**kw):dump(status,dict(status=s,elapsed_seconds=time.time()-start,scene=scene,GT_used=False,**kw))
try:
 while True:
  p=run/'mapping_driver_status.json';d=read(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('online 400 mapping failed '+str(d))
  if d['status']=='PASS':break
  state('WAITING_MAPPING');time.sleep(20)
 mapping=run/'mapping';source=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')/scene/'400/A/mapping';assert read(source/'mapping_complete.json')['status']=='PASS' and not read(source/'mapping_complete.json')['repair_applied'];assert read(mapping/'mapping_complete.json')['frame_count']==400
 oldenv=os.environ;os.environ['CVPR_SCENE']=scene;os.environ['CVPR_CHECKPOINT']='400';os.environ['CVPR_TRACK_GROUP']='B';sys.path.insert(0,'/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online');import aligned_surface as al
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019/e1_repair');import bootstrap;import run_batch as b
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/prefix100');from one_vote import effective_keys
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/policies/online');import scene_completion.pipeline;assert 'final-abc-20261101-online-v2' in str(Path(scene_completion.pipeline.__file__).resolve())
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106');import dbscan_guarded as db
 from scene_completion.pipeline import write_scene_complete,finalize
 from revisable_instance_map.surface_evidence import reduce_surface_votes
 from revisable_instance_map.offline_surface_assignment import AssignmentSettings
 from scene_completion.core import complete
 import open3d as o3d
 with np.load(mapping/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'].copy();rgb=z['rgb'].copy()
 cloud=o3d.io.read_point_cloud(str(mapping/'geometry/surface.ply'));np.testing.assert_array_equal(np.asarray(cloud.points).astype(np.float32),xyz);norm=np.asarray(cloud.normals).astype(np.float32);tree=cKDTree(xyz.astype(float));source_frames=b.r.ReplicaFrameSource(source/'config.json');ids=list(range(0,2000,5));assert list(source_frames.frame_ids)==ids
 lookup={};assoc=source/'association/associations.jsonl'
 for line in assoc.read_text().splitlines():
  x=json.loads(line);lookup[(int(x['frame_id']),int(x['mask_local_id']))]=int(x['instance_id'])
 base=max(lookup.values())+1;keys=[];transactions=out/'repair/partial/transactions';transactions.mkdir(parents=True,exist_ok=True)
 for i,fid in enumerate(ids,1):
  # The two branches must observe the same original (unrepaired) input mask and depth.
  original=source/'inputs/masks'/f'frame{fid:06d}.png';other=mapping/'inputs/masks'/f'frame{fid:06d}.png';assert sha(original)==sha(other),('different raw masks',fid)
  fr=source_frames.load_frame(fid);points,local,_=b.r.project_frame_regions(fr,tree,pixel_stride=2,max_distance_m=.015,workers=2);table=np.zeros(int(fr.mask_local.max())+1,np.int64)
  for lid in np.unique(local):table[lid]=lookup[(fid,int(lid))]
  raw=np.unique(points.astype(np.int64)*base+table[local]);effective=effective_keys(points,table[local],base,'abstain');np.testing.assert_array_equal(db.strict(raw,base),effective);keys.append(effective);save(transactions/f'f{fid:06d}.npz',whole_to_partial_candidate_keys=raw,whole_to_partial_frame_keys=effective)
  if i%20==0:state('RUNNING',stage='REPLAY_NO_REPAIR_OBSERVATIONS_ON_ONLINE_GEOMETRY',frames=i,total=400)
 kk,votes=np.unique(np.concatenate(keys),return_counts=True);del keys;ev=reduce_surface_votes(kk,votes,len(xyz),base,2,.67);native=np.where(ev['state']==2,ev['top1_instance_id'],-1).astype(np.int32);pair={'surface_point_index':(kk//base).astype(np.int32),'instance_id':(kk%base).astype(np.int32),'frame_votes':votes.astype(np.int32)};state('RUNNING',stage='ASSIGN_NO_REPAIR_ON_FIXED_GEOMETRY')
 variants,_,_=al.assign_surface(xyz,norm,rgb,ev,pair,native,AssignmentSettings());lab=variants['holes_geodesic'];save(out/'baseline/final/instance_surface.npz',xyz_m=xyz,rgb=rgb,instance_id=lab);save(out/'baseline/native/instance_surface.npz',xyz_m=xyz,rgb=rgb,instance_id=native);save(out/'baseline/surface_evidence.npz',xyz_m=xyz,rgb=rgb,**ev);save(out/'baseline/surface_instance_frame_votes.npz',**pair)
 manifest=out/'scene_complete.json';write_scene_complete(manifest,scene=scene,workflow='no_repair',frame_ids=ids,inputs={'surface':out/'baseline/final/instance_surface.npz','native':out/'baseline/native/instance_surface.npz','evidence':out/'baseline/surface_evidence.npz','normal_geometry':mapping/'geometry/surface.ply'},final_processing_complete=True,final_repair_complete=False);receipt=finalize(manifest,out/'final_B');final=db.load(receipt['final_surface']);ctl,_,_=complete(xyz,norm,lab,ev);np.testing.assert_array_equal(ctl,final['instance_id']);del ctl
 provenance=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/full_depth_rescreen_postprocessed/postprocess')/scene/'before/provenance.json';pixel=Path(read(provenance)['source_pixel_support']);assert pixel.exists();state('RUNNING',stage='BASELINE_DBSCAN_GUARDED');work=out/'dbscan';work.mkdir(exist_ok=True);lut,dec=db.build_candidates(xyz,final['instance_id'],pair,pixel,work,repair_branch=out,allow_repair_history=False)
 if len(lut)<base:lut=np.r_[lut,np.arange(len(lut),base,dtype=np.int32)]
 if dec['families']:_,_,mp=db.replay_votes(out,pair,base,lut,work);new=reduce_surface_votes(db.encode(mp,base),mp['frame_votes'],len(xyz),base,2,.67)
 else:mp=pair;new=ev;dump(work/'frame_replay_audit.json',{'status':'PASS','no_identity_changes':True})
 labels=lab.copy();ix=labels>0;labels[ix]=lut[labels[ix]];was=ev['state']==2;now=new['state']==2;labels[was&~now]=-1;labels[now]=new['top1_instance_id'][now]
 save(work/'pre_B.npz',xyz_m=xyz,rgb=rgb,instance_id=labels);save(work/'native.npz',xyz_m=xyz,rgb=rgb,instance_id=np.where(now,new['top1_instance_id'],-1).astype(np.int32));save(work/'evidence.npz',xyz_m=xyz,rgb=rgb,**new)
 write_scene_complete(work/'scene_complete.json',scene=scene,workflow='no_repair',frame_ids=ids,inputs={'surface':work/'pre_B.npz','native':work/'native.npz','evidence':work/'evidence.npz','normal_geometry':mapping/'geometry/surface.ply'},final_processing_complete=True,final_repair_complete=False);post=finalize(work/'scene_complete.json',work/'final_B');labels=db.load(post['final_surface'])['instance_id'];valid=np.isfinite(norm).all(axis=1)&(np.linalg.norm(norm,axis=1)>.5)&np.isfinite(xyz).all(axis=1);labels=db.fill_gray(xyz,norm,valid,labels,new,mp,work);save(out/'instance_surface.npz',xyz_m=xyz,rgb=rgb,instance_id=labels)
 state('PASS',stage='NO_REPAIR_FINAL_B_DBSCAN_ON_ONLINE_GEOMETRY',surface=str(out/'instance_surface.npz'),source_associations=str(assoc),source_associations_sha256=sha(assoc),source_pixel_support=str(pixel),source_pixel_support_sha256=sha(pixel),same_original_masks_verified=400,geometry_source=str(mapping/'geometry/surface.ply'),families=dec['families'])
except BaseException:state('FAILED',traceback=traceback.format_exc());raise

from pathlib import Path
import os,sys,json,time,traceback,hashlib
import numpy as np
R=Path('/data/chenkejun/CVPR/results/scannet0518_threshold_transfer_20261110/scannet400');root=R/'main/scene0518_00/400/B';out=R/'baseline_postprocess';branch=Path('/home/chenkejun/CVPR/experiments/scannet0518_threshold_transfer_20261110/scannet_repair_runtime/scannet400/main/scene0518_00');start=time.time()
def read(p):return json.loads(Path(p).read_text())
def save(p,**x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(q,**x);q.replace(p)
def dump(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def state(s,**k):dump(out/'status.json',dict(status=s,elapsed_seconds=time.time()-start,**k))
try:
 while True:
  p=out/'original_evidence/status.json';d=read(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('baseline evidence failed '+str(d))
  if d['status']=='PASS':break
  state('WAITING_ORIGINAL_EVIDENCE');time.sleep(15)
 evidence=read(out/'original_evidence/complete.json');assert evidence['status']=='PASS' and evidence['all_saved_observation_support_exact'];pixel=out/'original_evidence/source_pixel_support.npz'
 sys.path.insert(0,str(branch));import bootstrap_continue as boot
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet0518_threshold_transfer_20261110/scannet400_final_policy');import scene_completion.pipeline
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106');import dbscan_guarded as db
 sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet_ovimap_strict_dbscan_20261104/policies/main');from scene_completion.pipeline import finalize,write_scene_complete
 from revisable_instance_map.surface_evidence import reduce_surface_votes
 from scene_completion.core import complete
 scene=boot.r.Scene('scene0518_00');xyz=scene.xyz;rgb=scene.rgb;norm=scene.normals;basepath=root/'mapping/baseline';src=db.load(basepath/'final/instance_surface.npz');pair=db.load(basepath/'surface_instance_frame_votes.npz');ev=db.load(basepath/'surface_evidence.npz');np.testing.assert_array_equal(src['xyz_m'],xyz);np.testing.assert_array_equal(src['rgb'],rgb)
 state('RUNNING',stage='BASELINE_FINAL_B');work=out/'dbscan';work.mkdir(exist_ok=True);write_scene_complete(work/'scene_complete.json',scene='scene0518_00',workflow='no_repair',frame_ids=list(range(0,2000,5)),inputs=dict(surface=basepath/'final/instance_surface.npz',native=basepath/'native/instance_surface.npz',evidence=basepath/'surface_evidence.npz',normal_geometry=root/'mapping/geometry/surface.ply'),final_processing_complete=True,final_repair_complete=False);receipt=finalize(work/'scene_complete.json',work/'final_B_before_merge');final=db.load(receipt['final_surface']);base=max(int(pair['instance_id'].max()),int(src['instance_id'].max()),int(final['instance_id'].max()))+1
 old=reduce_surface_votes(db.encode(pair,base),pair['frame_votes'],len(xyz),base,2,.67)
 for key in boot.r.FIELDS:np.testing.assert_array_equal(old[key],ev[key])
 ctl,_,_=complete(xyz,norm,src['instance_id'],old);np.testing.assert_array_equal(ctl,final['instance_id']);state('RUNNING',stage='BASELINE_DBSCAN_GUARDED_MERGE')
 lut,decisions=db.build_candidates(xyz,final['instance_id'],pair,pixel,work,repair_branch=root,allow_repair_history=False)
 if len(lut)<base:lut=np.r_[lut,np.arange(len(lut),base,dtype=np.int32)]
 if decisions['families']:
  kk,vv,mp=db.replay_votes(root,pair,base,lut,work,original_only=True);merged=reduce_surface_votes(kk,vv,len(xyz),base,2,.67)
 else:merged=old;mp=pair;dump(work/'frame_replay_audit.json',{'status':'PASS','no_identity_changes':True})
 lab=src['instance_id'].copy();assigned=lab>0;lab[assigned]=lut[lab[assigned]];was=old['state']==2;now=merged['state']==2;lab[was&~now]=-1;lab[now]=merged['top1_instance_id'][now]
 save(work/'pre_B.npz',xyz_m=xyz,rgb=rgb,instance_id=lab);save(work/'native.npz',xyz_m=xyz,rgb=rgb,instance_id=np.where(now,merged['top1_instance_id'],-1).astype(np.int32));save(work/'evidence.npz',xyz_m=xyz,rgb=rgb,**merged);write_scene_complete(work/'merged_scene_complete.json',scene='scene0518_00',workflow='no_repair',frame_ids=list(range(0,2000,5)),inputs=dict(surface=work/'pre_B.npz',native=work/'native.npz',evidence=work/'evidence.npz',normal_geometry=root/'mapping/geometry/surface.ply'),final_processing_complete=True,final_repair_complete=False);receipt=finalize(work/'merged_scene_complete.json',work/'final_B_after_merge');labels=db.load(receipt['final_surface'])['instance_id'];state('RUNNING',stage='BASELINE_DBSCAN_GUARDED_FILL');valid=np.isfinite(norm).all(axis=1)&(np.linalg.norm(norm,axis=1)>.5)&np.isfinite(xyz).all(axis=1);labels=db.fill_gray(xyz,norm,valid,labels,merged,mp,work);save(out/'instance_surface.npz',xyz_m=xyz,rgb=rgb,instance_id=labels);digest=hashlib.sha256((out/'instance_surface.npz').read_bytes()).hexdigest();state('PASS',stage='BASELINE_FINAL_B_AND_GUARDED_DBSCAN_COMPLETE',surface=str(out/'instance_surface.npz'),sha256=digest,families=decisions['families'],GT_used_for_decision=False,evaluation_complete=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise

"""All annotated room1 identities; GT-free simultaneous multi-view expansion."""
from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010/room0')
import run_history_core as h
from adaptive_core import choose_core
from observation_ops import visible_samples,mask_support,choose_view_representatives,connected_expansion,resolve_claims
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/room1/e1_repair/e1');O=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/room1/e1_repair');A=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/room1/e1_repair/modified_repair/room1')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):h.f.atomic_json(p,x)
def foreground(pid,fid):
 mask=np.zeros((680,1200),bool)
 for tid in groups[pid]:
  if tid not in dec[fid]['accepted_track_ids']:continue
  row=info[obj[tid]]['frames'][fid];path=Path(row['label_file']);assert sha(path)==row['label_sha256']
  mask|=binary_erosion(np.asarray(Image.open(path))==tid,iterations=1)
 return mask
def expand_identity(pid):
 core=np.flatnonzero(coreowner==pid);fids=sorted(fid for fid,d in dec.items() if any(t in d['accepted_track_ids'] for t in groups[pid]));centers=[];positive=[]
 for fid in fids:
  fr=scene.source.load_frame(fid);mask=foreground(pid,fid);p,_=h.project(fr,mask.astype(np.uint16),scene);p=np.unique(p);vis,_,_=visible_samples(scene.xyz[p],fr.camera_to_world,fr.camera,fr.depth_m,.02);positive.append(p[vis]);centers.append(fr.camera_to_world[:3,3])
 pool=np.unique(np.concatenate([core,*positive]));support=np.zeros((len(fids),len(pool)),bool);visible=np.zeros_like(support)
 for i,fid in enumerate(fids):
  fr=scene.source.load_frame(fid);visible[i],support[i]=mask_support(scene.xyz[pool],fr.camera_to_world,fr.camera,fr.depth_m,foreground(pid,fid),.02)
 reps=choose_view_representatives(np.asarray(centers),.05);ns=support.sum(0);nv=visible.sum(0);views=support[reps].sum(0);ratio=np.divide(ns,nv,out=np.zeros(len(pool)),where=nv>0);blocked=((coreowner[pool]>0)&(coreowner[pool]!=pid))|(coreowner[pool]==-2);eligible=(ns>=3)&(views>=2)&(ratio>=.8)&~blocked;isc=np.isin(pool,core)
 chosen,rounds=connected_expansion(scene.xyz[pool],isc,eligible,.02) if len(pool) else (isc,[])
 h.f.atomic_npz(R/'regions'/f'id{pid}.npz',surface_point_index=pool[chosen],candidate_surface_point_index=pool,core=isc,eligible=eligible,selected=chosen,support_count=ns,visible_count=nv,view_count=views,positive_ratio=ratio,blocked=blocked,frame_ids=np.array(fids),support_matrix=support,visibility_matrix=visible,camera_representative_indices=reps)
 result={'persistent_id':pid,'track_ids':groups[pid],'core_points':len(core),'accepted_unique_frames':len(fids),'candidate_points':len(pool),'proposed_points':int(chosen.sum()),'growth_rounds':rounds,'blocked_foreign_core':int(blocked.sum())};print('IDENTITY',json.dumps(result),flush=True);return result
def main():
 global scene,groups,dec,info,obj,coreowner
 R.mkdir(parents=True,exist_ok=True);assert not (R/'region.npz').exists()
 policy={'method':'room1_all_annotated_observation_expansion_v1','control':str(A),'scope':'all v2 retained primary masks and their frozen persistent identities','same_identity_aliases':'erode each accepted member mask then union, count each frame once','core':'largest erosion radius 10 down to 1 giving max(20,ceil(0.2*seed projected unique points)); preserve existing core claims; conflicts ambiguous','depth_tolerance_m':.02,'mask_erosion_pixels':1,'minimum_support_frames':3,'minimum_supporting_camera_representatives':2,'camera_separation_m':.05,'minimum_visible_positive_fraction':.8,'growth_radius_m':.02,'support':'depth-visible TSDF reverse projection into accepted mask; frozen forward vote projection unchanged','cross_target_growth':'simultaneous claims, different IDs ambiguous, remove ambiguous bridges then reconnect to core','history':'retained original sampled source observations only; same visible point relabel to unique confirmed region identity, unrestricted historical IDs','E1':'expanded history only','SAM2_and_mask_gate':'same V2 gate applied to new SAM2.1 tracks','final_commit':'only points with aggregate vote counts changed vs control','GT_read_by_mapper':False,'GT_used_for_thresholds':False,'threshold_sweep':False,'architecture_frozen':True}
 dump(R/'policy.json',policy);scene=h.r.Scene('room1');cases,assoc,obj,info,canonical,fixed=h.catalog(O);groups={}
 for tid,can in canonical.items():groups.setdefault(int(fixed[can]),[]).append(int(tid))
 groups={k:sorted(v) for k,v in groups.items()};assert sum(map(len,groups.values()))==len(read(O/'seed_id_map.json')['objects']) and len(groups)==len(read(O/'seed_id_map.json')['objects'])
 dec={x['frame']:x for x in read(O/'validated/full_track/frame_decisions.json')['frames']};priorz=dict(np.load(A/'core_claims.npz'));prior=np.full(len(scene.xyz),-1,np.int32);prior[priorz['surface_point_index']]=priorz['claimed_persistent_id'];coreclaims=[];coreids=[];objects=[]
 for case in read(O/'seed_manifest.json')['seeds']:
  if case['status']!='READY':continue
  uid=case['case_uid'];fid=case['source_choice']['frame'];fr=scene.source.load_frame(fid);lab=np.asarray(Image.open(O/'cases'/uid/'prepared_seed.png'))
  for ob in case['objects']:
   tid=ob['track_id'];pid=int(fixed[canonical[tid]]);project=lambda mask:h.project(fr,mask.astype(np.uint16),scene)[0];interior,profile=choose_core(lab==tid,project);p=np.unique(project(interior));coreclaims.append(p);coreids.append(pid);objects.append({'ROI':case['source_choice']['ROI'],'uid':uid,'frame':fid,'mask':ob['native_mask_id'],'track_id':tid,'persistent_id':pid,'profile':profile,'seed_points':ob['projected_seed_points'],'new_core_points':len(p)});print('CORE',tid,pid,profile['radius'],len(p),flush=True)
 for pid in groups:coreclaims.append(np.flatnonzero(prior==pid));coreids.append(pid)
 coreowner=resolve_claims(coreclaims,coreids,len(scene.xyz));coreowner[prior==-2]=-2
 h.f.atomic_npz(R/'adaptive_cores.npz',owner=coreowner);dump(R/'objects.json',objects)
 with ProcessPoolExecutor(max_workers=6) as pool:reports=list(pool.map(expand_identity,sorted(groups)))
 proposals=[np.load(R/'regions'/f'id{pid}.npz')['surface_point_index'] for pid in sorted(groups)];owner=resolve_claims(proposals,sorted(groups),len(scene.xyz));owner[coreowner==-2]=-2;initial_amb=owner==-2;finalclaims=[]
 for row in reports:
  pid=row['persistent_id'];z=dict(np.load(R/'regions'/f'id{pid}.npz'));p=z['candidate_surface_point_index'];eligible=z['eligible']&~initial_amb[p];chosen,rounds=connected_expansion(scene.xyz[p],z['core'],eligible,.02) if len(p) else (z['core'],[]);assert not initial_amb[p[chosen]].any();finalclaims.append(p[chosen]);row['final_points']=int(chosen.sum());row['lost_to_cross_target_ambiguity_or_disconnection']=row['proposed_points']-row['final_points'];z['selected']=chosen;z['surface_point_index']=p[chosen];h.f.atomic_npz(R/'regions'/f'id{pid}.npz',**z)
 finalowner=resolve_claims(finalclaims,[r['persistent_id'] for r in reports],len(scene.xyz));finalowner[initial_amb]=-2;assert np.all(finalowner[coreowner>0]==coreowner[coreowner>0]);points=np.flatnonzero(finalowner!=-1)
 h.f.atomic_npz(R/'region.npz',surface_point_index=points,owner=finalowner,core_owner=coreowner,prior_owner=prior);dump(R/'region_report.json',{'targets':reports,'objects':objects,'region_points':len(points),'unique_owned_points':int((finalowner>0).sum()),'ambiguous_points':int((finalowner==-2).sum()),'core_ambiguous_points':int((coreowner==-2).sum()),'GT_read':False})
 sources={str(p):sha(p) for p in [O/'validated/full_track/frame_decisions.json',O/'global_association.json',O/'seed_manifest.json',A/'core_claims.npz']};dump(R/'region_freeze.json',{'status':'REGION_FROZEN_BEFORE_GT','region_sha256':sha(R/'region.npz'),'policy_sha256':sha(R/'policy.json'),'source_sha256':sources,'code_sha256':{str(C/x):sha(C/x) for x in ['expand_room1.py','observation_ops.py','adaptive_core.py']}});print('ALL_REGIONS_FROZEN',len(points),flush=True)
if __name__=='__main__':main()

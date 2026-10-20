"""Frozen E1: expanded trusted regions and historical observation reassociation."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib,time,traceback,gc
import numpy as np
from PIL import Image
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010/room0')
import run_history_core as h
from observation_ops import visible_samples,revise_all_history,keyset,strict_frame
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office3/e1_repair/e1');O=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office3/e1_repair');A=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office3/e1_repair/modified_repair/office3');B=O/'original_repair/office3/full_track';NAMES=['E1_expanded_history']
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:dict(np.load(p))
def dump(p,v):h.f.atomic_json(p,v)
def emit(stage,**kw):v={'stage':stage,**kw,'unix':time.time()};dump(R/'progress.json',v);print(json.dumps(v),flush=True)

def main():
 started=time.time();freeze=read(R/'region_freeze.json');assert sha(R/'region.npz')==freeze['region_sha256'];assert sha(R/'policy.json')==freeze['policy_sha256'];assert read(R/'tests_complete.json')['status']=='PASS'
 for p,d in freeze['code_sha256'].items():assert sha(p)==d,p
 codehash={str(C/p):sha(C/p) for p in ['observation_ops.py','expand_office3.py','run_observation_repair.py','test_observation_ops.py']}
 baselinefreeze=read(A.parent/'predictions_freeze.json')
 for p,d in baselinefreeze['output_sha256'].items():assert sha(p)==d,p
 for name in NAMES:(R/name/'office3').mkdir(parents=True,exist_ok=False)
 scene=h.r.Scene('office3');cases,assoc,obj,info,canonical,fixed=h.catalog(O);base=assoc['instance_base'];dec={x['frame']:x for x in h.read(O/'validated/full_track/frame_decisions.json')['frames']}
 final0=load(A/'final/instance_surface.npz');native0=load(A/'native/instance_surface.npz');e0=load(A/'native/surface_evidence.npz');pair0=load(A/'native/surface_instance_frame_votes.npz');bk,bv=h.encode(pair0,base);np.testing.assert_array_equal(scene.xyz,final0['xyz_m']);claims=load(A/'core_claims.npz');prior=np.full(len(scene.xyz),-1,np.int32);prior[claims['surface_point_index']]=claims['claimed_persistent_id'];regiondata=load(R/'region.npz');region=regiondata['surface_point_index'];owner=regiondata['owner'];np.testing.assert_array_equal(prior,regiondata['prior_owner']);expanded=owner!=-1
 allowed=expanded|(prior!=-1);apoints=np.flatnonzero(allowed);all_before=[];all_after=[[]];removed=[[]];added=[[]];receipts=[[]];sources=[];projected=0;skipped=0
 for j,fid in enumerate(sorted(scene.records)):
  # New seeds may have frames with no accepted tracks. Reconstruct their exact
  # unchanged original candidate set instead of assuming a transaction exists.
  bpath=B/'transactions'/f'f{fid:06d}.npz'
  if bpath.exists():btx=load(bpath)
  else:
   record=scene.records[fid];z=h.load(record['support_file'],record['support_sha256']);pp=z['surface_point_index'];ll=z['mask_local_id'];tt=np.zeros(int(ll.max())+1,np.int64)
   for mid0 in np.unique(ll):tt[mid0]=scene.lookup[(fid,int(mid0))]
   kk=keyset(pp,tt[ll],base);btx={'new_candidate_keys':kk,'new_frame_keys':strict_frame(kk,base)}
   assert not dec[fid]['accepted_track_ids']
  path=A/'transactions'/f'f{fid:06d}.npz';atx=load(path) if path.exists() else btx;candidate0=atx['new_candidate_keys'];before=atx['new_frame_keys'];np.testing.assert_array_equal(strict_frame(candidate0,base),before)
  affected=np.any(expanded[candidate0//base]);after=[before];candidates=[candidate0]
  if affected:
   frame=scene.source.load_frame(fid);record=scene.records[fid];original=h.load(record['support_file'],record['support_sha256']);table=np.zeros(int(frame.mask_local.max())+1,np.int64)
   for m in np.unique(frame.mask_local):
    if m:table[m]=scene.lookup[(fid,int(m))]
   p,mid,u,v=h.sourcepixels(frame,scene);np.testing.assert_array_equal(keyset(p,mid,len(table)),keyset(original['surface_point_index'],original['mask_local_id'],len(table)))
   residual,rp,rl,np_,nl=h.reconstruct_current(frame,table,scene,dec[fid],obj,info,canonical,fixed);available=residual[v,u]==mid;old=table[mid];idsA=old.copy();q=available&(prior[p]>0)&(old!=prior[p]);idsA[q]=prior[p[q]]
   newkey=keyset(np_,np.asarray([fixed[int(t)] for t in nl]),base);rawB=np.union1d(keyset(rp,table[rl],base),newkey);np.testing.assert_array_equal(rawB,btx['new_candidate_keys']);reconstructedA=np.union1d(keyset(p[available],idsA[available],base),newkey);np.testing.assert_array_equal(reconstructedA,candidate0)
   vis=np.zeros(len(scene.xyz),bool);vis[apoints],_,_=visible_samples(scene.xyz[apoints],frame.camera_to_world,frame.camera,frame.depth_m,.02);idsE,changed=revise_all_history(p,old,available,prior,owner,vis)
   # Fresh projection of a revised original-observation image, independently checked against sampled-pixel provenance.
   histimg=table[residual].astype(np.uint16);histimg[v[available],u[available]]=idsE[available];hp,hl=h.project(frame,histimg,scene);histkey=keyset(hp,hl,base);np.testing.assert_array_equal(histkey,keyset(p[available],idsE[available],base))
   candidates[0]=np.union1d(histkey,newkey);after[0]=strict_frame(candidates[0],base)
   # Unresolved competing identities abstain; one reliable source cannot receive two votes.
   for k in range(1):
    assert len(after[k])==len(np.unique(after[k]//base));np.testing.assert_array_equal(candidates[k][~allowed[candidates[k]//base]],candidate0[~allowed[candidate0//base]])
   assert np.all(np.isin(newkey,candidates[0]))
   source_row={'frame':fid,'reassociated_old_source_pixels':int(changed.sum()),'source_old_ID_counts':{str(int(i)):int((old[changed]==i).sum()) for i in np.unique(old[changed])},'target_ID_counts':{str(int(i)):int((idsE[changed]==i).sum()) for i in np.unique(idsE[changed])}};sources.append(source_row);projected+=1
   if changed.any():
    h.f.atomic_npz(R/'source_observations'/f'f{fid:06d}.npz',reassociated_pixel_uv=np.column_stack((u[changed],v[changed])).astype(np.int32),original_mask_local_id=mid[changed].astype(np.int32),old_persistent_id=old[changed].astype(np.int32),new_persistent_id=idsE[changed].astype(np.int32),surface_point_index=p[changed].astype(np.int32))
  else:skipped+=1
  all_before.append(before[allowed[before//base]])
  for k,name in enumerate(NAMES):
   rem=np.setdiff1d(before,after[k]);add=np.setdiff1d(after[k],before);scope=np.unique(np.r_[rem//base,add//base]);assert np.all(allowed[scope]);np.testing.assert_array_equal(before[~allowed[before//base]],after[k][~allowed[after[k]//base]]);all_after[k].append(after[k][allowed[after[k]//base]])
   if len(rem) or len(add):
    removed[k].append(rem);added[k].append(add);receipts[k].append({'frame':fid,'removed':len(rem),'added':len(add),'scope_points':len(scope)})
    h.f.atomic_npz(R/name/'office3/transactions'/f'f{fid:06d}.npz',old_candidate_keys=candidate0,new_candidate_keys=candidates[k],old_frame_keys=before,new_frame_keys=after[k],removed_effective_keys=rem,added_effective_keys=add)
  if (j+1)%25==0:emit('replay',frames=j+1,fresh_projection_frames=projected,changed_frames=[len(x) for x in receipts])
 beforekeys,beforecounts=np.unique(np.concatenate(all_before),return_counts=True);sel=allowed[bk//base];np.testing.assert_array_equal(beforekeys,bk[sel]);np.testing.assert_array_equal(beforecounts,bv[sel]);del all_before
 for k,name in enumerate(NAMES):
  emit('reduce',variant=name);rem=np.concatenate(removed[k]) if removed[k] else np.array([],np.int64);add=np.concatenate(added[k]) if added[k] else np.array([],np.int64);nk,nvotes=h.batch_delta(bk,bv,rem,add);backk,backv=h.batch_delta(nk,nvotes,add,rem);np.testing.assert_array_equal(backk,bk);np.testing.assert_array_equal(backv,bv);fullk,fullv=np.unique(np.concatenate(all_after[k]),return_counts=True);take=allowed[nk//base];np.testing.assert_array_equal(fullk,nk[take]);np.testing.assert_array_equal(fullv,nvotes[take]);all_after[k]=[]
  common,ni,bi=np.intersect1d(nk,bk,return_indices=True);scope=np.unique(np.r_[np.setxor1d(nk,bk)//base,common[nvotes[ni]!=bv[bi]]//base]).astype(np.int32);monitor=np.zeros(len(scene.xyz),bool);monitor[scope]=True;assert np.all(allowed[scope]);e=h.r.reduce_surface_votes(nk,nvotes,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio']);pt=h.pairtable(nk,nvotes,base);native=np.where(e['state']==h.r.CONFIRMED,e['top1_instance_id'],-1).astype(np.int32)
  for field in h.r.FIELDS:np.testing.assert_array_equal(e[field][~monitor],e0[field][~monitor])
  variants,_,post=h.r.assign_surface(scene.xyz,scene.normals,scene.rgb,e,pt,native,scene.settings,progress=lambda stage,values:emit('postprocess',variant=name,substage=stage,values=values));final=final0['instance_id'].copy();final[scope]=variants['holes_geodesic'][scope];np.testing.assert_array_equal(final[~monitor],final0['instance_id'][~monitor]);folder=R/name/'office3'
  for stage,lab in [('native',native),('final',final)]:h.f.atomic_npz(folder/stage/'instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=lab,native_instance_ids=np.union1d(final0['native_instance_ids'],np.unique(lab[lab>0])).astype(np.int32))
  h.f.atomic_npz(folder/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**e);h.f.atomic_npz(folder/'native/surface_instance_frame_votes.npz',**pt);h.f.atomic_npz(folder/'allowed_surface_ids.npz',surface_point_index=scope)
  complete={'status':'PASS','variant':name,'GT_read_by_mapper':False,'mapping_frames':400,'fresh_projection_frames':projected,'unchanged_frames_skipped':skipped,'expanded_region_points':len(region),'effective_vote_changed_points':len(scope),'final_changed_points':int((final!=final0['instance_id']).sum()),'before':h.summary(e0,final0['instance_id']),'after':h.summary(e,final),'target_regions':{str(int(pid)):{'before':h.region_summary(e0,final0['instance_id'],np.flatnonzero(owner==pid),int(pid)),'after':h.region_summary(e,final,np.flatnonzero(owner==pid),int(pid))} for pid in np.unique(owner[owner>0])},'native_state_transition':np.bincount(e0['state'].astype(np.int64)*4+e['state'],minlength=16).reshape(4,4).tolist(),'checks':{'original_projection_matches_frozen_support':True,'baseline_source_reconstruction_exact':True,'source_pixel_oracle_equals_fresh_projection':True,'one_vote_per_frame_point':True,'shared_same_ID_support_preserved':True,'incremental_equals_full_400_frame_replay':True,'exact_rollback':True,'outside_changed_scope_bit_identical':True},'frame_receipts':receipts[k],'postprocessing_statistics':post};dump(folder/'complete.json',complete);emit('variant_complete',variant=name,changed=complete['final_changed_points']);del e,pt,nk,nvotes,variants;gc.collect()
 dump(R/'source_observations.json',{'frames':sources,'GT_read':False});scene.verify_unchanged()
 for p,d in codehash.items():assert sha(p)==d,p
 for p,d in h.protected.items():assert sha(p)==d,p
 for p,d in baselinefreeze['output_sha256'].items():assert sha(p)==d,p
 for name in NAMES:
  folder=R/name;dump(folder/'predictions_freeze.json',{'status':'PREDICTIONS_FROZEN_BEFORE_GT','GT_read_by_mapper':False,'GT_used_for_thresholds':False,'policy_sha256':sha(R/'policy.json'),'region_sha256':sha(R/'region.npz'),'code_sha256':codehash,'output_sha256':{str(p):sha(p) for p in folder.rglob('*.npz')},'baseline_modified':False,'control':str(A)})
 dump(R/'mapping_complete.json',{'status':'PASS','variants':NAMES,'seconds':time.time()-started,'baseline_modified':False});emit('mapping_complete',seconds=time.time()-started)
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'failure.json',{'traceback':traceback.format_exc()});raise

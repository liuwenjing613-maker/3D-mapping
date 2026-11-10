from bootstrap import *
from dataclasses import replace
import time,gc,traceback,resource
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import run_history_core as h
import observation_ops as obs
from history_core_ops import core_claims,strict_frame
STAGES=('direct','core_all','whole_to_partial')
PARENT=dict(zip(STAGES,('baseline','direct','core_all')))
def cat(chunks):return np.concatenate(chunks) if chunks else np.empty(0,np.int64)
def aggregate(chunks):return np.unique(cat(chunks),return_counts=True)
def run_gate():
 b=load_script('_online_batch',C/'run_batch.py')
 exec(compile((C/'gate_online_body.py').read_text(),str(C/'gate_online_body.py'),'exec'),b.__dict__)
 scene,cases,objects,canonical,policy=b.inputs()
 association=b.freeze_global_association(scene,cases,objects,canonical)
 tracking=b.validate_tracking(cases)
 counts=b.gate_online(scene,cases,objects,canonical,policy,association,tracking)
 return counts
def build_core():
 scene=r.Scene('room0');cases,association,obj,info,canonical,fixed=h.catalog(R)
 core_arrays=[];target_ids=[];audit=[]
 for case in cases:
  uid=case['case_uid'];cfg=info[uid]['config'];fr=scene.source.load_frame(cfg['seed_frame'])
  label=np.asarray(Image.open(info[uid]['seedpath']));corelabel=np.zeros(label.shape,np.uint16)
  for item in case['objects']:
   tid=item['track_id'];corelabel[binary_erosion(label==tid,iterations=10)]=tid
  cp,cl=h.project(fr,corelabel,scene)
  for item in case['objects']:
   tid=item['track_id'];p=np.unique(cp[cl==tid]);core_arrays.append(p);target_ids.append(fixed[canonical[tid]])
   audit.append(dict(track_id=tid,persistent_id=fixed[canonical[tid]],core_points=len(p)))
 owner,claimkeys=core_claims(core_arrays,target_ids,len(scene.xyz),association['instance_base'])
 union=np.flatnonzero(owner!=-1)
 h.f.atomic_npz(R/'modified_repair/room0/core_claims.npz',surface_point_index=union,claimed_persistent_id=owner[union],raw_point_ID_claims=claimkeys,unique_owned_surface_point_index=np.flatnonzero(owner>0),ambiguous_surface_point_index=np.flatnonzero(owner==-2))
 dump(R/'modified_repair/room0/core_preflight.json',dict(objects=audit,unique_owned_points=int((owner>0).sum()),ambiguous_points=int((owner==-2).sum()),GT_used=False))
def all_pixel_history(frame,residual,table,scene,prior,owner,vis):
 # Source-pixel journal for changing TSDF geometry and sampled association support.
 # Applying the same rule at every raw source pixel also defines odd-pixel support;
 # the original stride-2 projection remains exactly identical.
 image=table[residual].astype(np.uint16)
 yy,xx=np.nonzero((residual>0)&np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<10))
 z=frame.depth_m[yy,xx].astype(np.float64)
 cam=np.column_stack(((xx-frame.camera.cx)*z/frame.camera.fx,(yy-frame.camera.cy)*z/frame.camera.fy,z))
 world=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
 dist,p=scene.tree.query(world,k=1,workers=8);ok=np.isfinite(dist)&(dist<.015)
 p=p[ok];yy=yy[ok];xx=xx[ok]
 ids,_=obs.revise_all_history(p,image[yy,xx],np.ones(len(p),bool),prior,owner,vis)
 image[yy,xx]=ids
 return image
def replay():
 started=time.perf_counter();scene=r.Scene('room0')
 cases,assoc,obj,info,canonical,fixed=h.catalog(R);base=assoc['instance_base']
 decisions={x['frame']:x for x in read(R/'validated/full_track/frame_decisions.json')['frames']}
 region=h.load(R/'e1/region.npz');prior,owner=region['prior_owner'],region['owner']
 apoints=np.flatnonzero((prior!=-1)|(owner!=-1))
 bk,bv=h.encode(scene.pair,base)
 frames={name:[] for name in ('baseline',*STAGES)}
 removed={name:[] for name in STAGES};added={name:[] for name in STAGES};scopes={name:[] for name in STAGES}
 can_to_pid=np.zeros(max(fixed)+1,np.int64)
 for cid,pid in fixed.items():can_to_pid[cid]=pid
 receipts=[];timings={'partial_reconstruction':0.,'core_history':0.,'expanded_history':0.,'pixel_journal':0.,'vote_audit':0.}
 for index,fid in enumerate(sorted(scene.records)):
  tick=time.perf_counter();frame=scene.source.load_frame(fid);decision=decisions[fid]
  table=np.zeros(int(frame.mask_local.max())+1,np.int64)
  for mid in np.unique(frame.mask_local):
   if mid:table[mid]=scene.lookup[(fid,int(mid))]
  p,mid,u,v=h.sourcepixels(frame,scene);old=table[mid]
  support=h.load(scene.records[fid]['support_file'],scene.records[fid]['support_sha256'])
  original_local=h.keys(p,mid,len(table));np.testing.assert_array_equal(original_local,h.keys(support['surface_point_index'],support['mask_local_id'],len(table)))
  merged=np.asarray(Image.open(R/'validated/full_track/accepted_masks'/f'f{fid:06d}.png')).copy()
  maskp,maskcid=h.project(frame,merged,scene);expected_new=h.keys(maskp,can_to_pid[maskcid],base)
  residual,rp,rl,np_,nc=h.reconstruct_current(frame,table,scene,decision,obj,info,canonical,fixed)
  newkeys=h.keys(np_,can_to_pid[nc],base);np.testing.assert_array_equal(newkeys,expected_new)
  available=residual[v,u]==mid;retained=h.keys(p[available],old[available],base)
  np.testing.assert_array_equal(retained,h.keys(rp,table[rl],base))
  timings['partial_reconstruction']+=time.perf_counter()-tick
  tick=time.perf_counter();ids_core=old.copy();choose=available&(prior[p]>0)&(old!=prior[p]);ids_core[choose]=prior[p[choose]]
  timings['core_history']+=time.perf_counter()-tick
  tick=time.perf_counter();vis=np.zeros(len(scene.xyz),bool)
  vis[apoints],_,_=obs.visible_samples(scene.xyz[apoints],frame.camera_to_world,frame.camera,frame.depth_m,.02)
  ids_expanded,changed=obs.revise_all_history(p,old,available,prior,owner,vis)
  timings['expanded_history']+=time.perf_counter()-tick
  tick=time.perf_counter();baseline_raw=h.keys(p,old,base);baseline_eff=strict_frame(baseline_raw,base)
  frames['baseline'].append(baseline_eff);eff={'baseline':baseline_eff};tx={'baseline_candidate_keys':baseline_raw,'baseline_frame_keys':baseline_eff}
  for name,ids in zip(STAGES,(old,ids_core,ids_expanded)):
   histkeys=h.keys(p[available],ids[available],base);raw=np.union1d(histkeys,newkeys)
   image=table[residual].astype(np.uint16);image[v[available],u[available]]=ids[available]
   hp,hi=h.project(frame,image,scene);np.testing.assert_array_equal(h.keys(hp,hi,base),histkeys)
   eff[name]=strict_frame(raw,base);assert len(eff[name])==len(np.unique(eff[name]//base))
   parent=PARENT[name];rem=np.setdiff1d(eff[parent],eff[name]);add=np.setdiff1d(eff[name],eff[parent])
   scope=np.unique(np.r_[rem//base,add//base]).astype(np.int32)
   if name=='direct':
    retired_local=np.setdiff1d(original_local,h.keys(rp,rl,len(table)))
    direct_scope=np.unique(np.r_[retired_local//len(table),np_]).astype(np.int32)
    assert np.all(np.isin(scope,direct_scope));scope=direct_scope;assert np.all(np.isin(scope,maskp))
   elif name=='core_all':assert np.all(prior[scope]>0)
   else:assert np.all((owner[scope]!=-1)|(prior[scope]!=-1))
   removed[name].append(rem);added[name].append(add);scopes[name].append(scope);frames[name].append(eff[name])
   tx[name+'_candidate_keys']=raw;tx[name+'_frame_keys']=eff[name]
  h.f.atomic_npz(R/'partial/transactions'/f'f{fid:06d}.npz',**tx)
  timings['vote_audit']+=time.perf_counter()-tick
  tick=time.perf_counter();history=all_pixel_history(frame,residual,table,scene,prior,owner,vis)
  hp,hi=h.project(frame,history,scene)
  np.testing.assert_array_equal(h.keys(hp,hi,base),h.keys(p[available],ids_expanded[available],base))
  adopted=can_to_pid[merged].astype(np.uint16)
  journal=R/'commit_candidate/pixel_journal';journal.mkdir(parents=True,exist_ok=True)
  for suffix,im in [('history',history),('adopted',adopted),('residual_local',residual)]:
   Image.fromarray(im.astype(np.uint16)).save(journal/f'f{fid:06d}_{suffix}.png')
  timings['pixel_journal']+=time.perf_counter()-tick
  receipts.append(dict(frame=fid,retained_source_pixels=int(available.sum()),prior_core_relabelled_source_pixels=int(choose.sum()),expanded_region_relabelled_source_pixels=int(changed.sum()),strict_vote_unique=True))
  if (index+1)%10==0:emit('partial_replay',frames=index+1,total=50)
 pk,pv=aggregate(frames.pop('baseline'));np.testing.assert_array_equal(pk,bk);np.testing.assert_array_equal(pv,bv)
 baseline_reduced=r.reduce_surface_votes(bk,bv,len(scene.xyz),base,2,.67)
 for field in r.FIELDS:np.testing.assert_array_equal(baseline_reduced[field],scene.evidence[field])
 results={'baseline':(bk,bv,scene.evidence,scene.final)};reports={};inventory=np.unique(scene.final[scene.final>0])
 for name in STAGES:
  tick=time.perf_counter();pk,pv,pe,pf=results[PARENT[name]]
  rem,add=cat(removed.pop(name)),cat(added.pop(name));nk,nv=h.batch_delta(pk,pv,rem,add)
  backk,backv=h.batch_delta(nk,nv,add,rem);np.testing.assert_array_equal(backk,pk);np.testing.assert_array_equal(backv,pv)
  fullk,fullv=aggregate(frames.pop(name));np.testing.assert_array_equal(nk,fullk);np.testing.assert_array_equal(nv,fullv)
  scope=np.unique(cat(scopes.pop(name))).astype(np.int32)
  if name=='whole_to_partial':
   common,ni,pi=np.intersect1d(nk,pk,return_indices=True)
   scope=np.unique(np.r_[np.setxor1d(nk,pk)//base,common[nv[ni]!=pv[pi]]//base]).astype(np.int32)
  monitor=np.zeros(len(scene.xyz),bool);monitor[scope]=True
  np.testing.assert_array_equal(nk[~monitor[nk//base]],pk[~monitor[pk//base]])
  np.testing.assert_array_equal(nv[~monitor[nk//base]],pv[~monitor[pk//base]])
  evidence=r.reduce_surface_votes(nk,nv,len(scene.xyz),base,2,.67)
  for field in r.FIELDS:np.testing.assert_array_equal(evidence[field][~monitor],pe[field][~monitor])
  native=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32);pair=h.pairtable(nk,nv,base)
  emit('stage_postprocess',stage_name=name)
  variants,_,post=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native,scene.settings)
  final=pf.copy();final[scope]=variants['holes_geodesic'][scope]
  np.testing.assert_array_equal(final[~monitor],pf[~monitor]);folder=R/'partial'/name/'room0'
  for kind,lab in [('native',native),('final',final)]:
   inv=np.union1d(inventory,np.unique(lab[lab>0])).astype(np.int32)
   h.f.atomic_npz(folder/kind/'instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=lab,native_instance_ids=inv)
  h.f.atomic_npz(folder/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**evidence)
  h.f.atomic_npz(folder/'native/surface_instance_frame_votes.npz',**pair)
  h.f.atomic_npz(folder/'allowed_surface_ids.npz',surface_point_index=scope)
  reports[name]=dict(status='PASS',parent=PARENT[name],scope_points=len(scope),removed_votes=len(rem),added_votes=len(add),final_changed_vs_parent=int((final!=pf).sum()),summary=h.summary(evidence,final),postprocessing=post,seconds=time.perf_counter()-tick)
  dump(folder/'complete.json',reports[name]);results[name]=(nk,nv,evidence,final)
  del variants,fullk,fullv,backk,backv,rem,add;gc.collect()
 scene.verify_unchanged()
 dump(R/'partial/frame_receipts.json',receipts)
 dump(R/'partial/mapping_complete.json',dict(status='PASS',seconds=time.perf_counter()-started,stages=reports,timings=timings,whole_control_replayed=False,stages_committed_in_original_order=True,full_replay_equals_delta=True,exact_rollback=True,strict_same_frame_point_vote=True,raw_RGB_cutoff=245,GT_used=False))
 return dict(stages=reports,frame_timings=timings,seconds=time.perf_counter()-started)
def main():
 assert read(R/'tracking_queue_complete.json')['status']=='PASS'
 assert read(R/'tests_complete.json')['status']=='PASS'
 start=time.perf_counter();timings={}
 for name,fn in [('gate',run_gate),('core_region',build_core),('expand',lambda:load_script('_online_expand',C/'expand_room0.py').main()),('partial_core_E1',replay)]:
  emit(name);tick=time.perf_counter();stamp=R/('stage_'+name+'_complete.json')
  if stamp.exists():timings[name]=read(stamp)
  else:
   result=fn();timings[name]=dict(seconds=time.perf_counter()-tick,result=result);dump(stamp,timings[name])
 dump(R/'repair_complete.json',dict(status='PASS',seconds=time.perf_counter()-start,timings=timings,peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,read_bytes=Path('/proc/self/io').read_text(),main_mapping_paused=True,committed_for_continuation=False,GT_used=False))
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'repair_failure.json',dict(traceback=traceback.format_exc()));raise

"""Authorized second checkpoint repair: preserve every existing identity layer."""
from pathlib import Path
import sys,os,json,time,hashlib,gc,traceback,resource
HERE=Path(__file__).resolve().parent
BASEC=HERE.parent
sys.path.insert(0,str(BASEC/'tracking100'))
from bootstrap100 import *
from dataclasses import replace
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from history_core_ops import core_claims,strict_frame
TRACK=R
R=TOP/'repair100'/GROUP
STAGES=('direct','core_all','whole_to_partial')
PARENT=dict(zip(STAGES,('baseline','direct','core_all')))
h=load_script('_repair100_history_helpers',BASEC/'run_history_core.py')
sys.modules['run_history_core']=h
obs=load_script('_repair100_observation_ops',BASEC/'observation_ops.py')
sys.modules['observation_ops']=obs
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def emit(stage,**kw):
 x=dict(status='RUNNING',group=GROUP,stage=stage,unix=time.time(),**kw)
 write(R/'progress.json',x);print(json.dumps(x),flush=True)
def cat(x):return np.concatenate(x) if x else np.empty(0,np.int64)
def aggregate(x):return np.unique(cat(x),return_counts=True)
def catalog(root):
 cases=read(root/'seed_diagnostics.json')['cases'];association=read(root/'global_association.json')
 assert association['GT_used'] is False
 canonical={};fixed={};obj={};info={}
 for row in association['objects']:
  fixed[row['canonical_track_id']]=row['persistent_id']
  for oid in row['member_track_ids']:canonical[oid]=row['canonical_track_id']
 assert len(set(fixed.values()))==len(fixed)
 for case in cases:
  uid=case['case_uid'];cfg=read(case['config_path'])
  complete=read(root/'cases'/uid/'tracking/complete.json')
  assert complete['status']=='PASS' and complete['GT_used'] is False
  assert complete['completed_frames']==complete['total_frames']==496
  assert digest(case['config_path'])==complete['case_config_sha256']
  seedpath=root/'cases'/uid/'prepared_seed.png';assert digest(seedpath)==complete['seed_sha256']
  info[uid]=dict(config=cfg,frames={x['frame']:x for x in complete['frames']},seedpath=seedpath)
  for item in case['objects']:obj[item['track_id']]=uid
 assert set(obj)==set(canonical)
 return cases,association,obj,info,canonical,fixed
h.catalog=catalog
def setup():
 R.mkdir(parents=True,exist_ok=True)
 expected=13 if GROUP=='A' else 12
 assert read(TRACK/'gate_complete.json')['status']=='PASS'
 assert read(TRACK/'tracking_queue_complete.json')['status']=='PASS'
 assert read(TRACK/'gate_complete.json')['objects']==expected
 for name in ['seed_diagnostics.json','global_association.json','seed_manifest.json','seed_id_map.json','input_freeze.json','cases']:
  dst=R/name
  if not dst.exists():dst.symlink_to(TRACK/name,target_is_directory=(name=='cases'))
  assert dst.resolve()==(TRACK/name).resolve()
 dst=R/'validated/full_track';dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(TRACK/'gate',target_is_directory=True)
 assert dst.resolve()==(TRACK/'gate').resolve()
 inputs={str(p):digest(p) for p in [TRACK/'gate/frame_decisions.json',TRACK/'global_association.json',TRACK/'seed_manifest.json',TRACK/'seed_diagnostics.json',TOP/'committed50.json',B/'association/observation_support_3cm.npz',B/'baseline/final/instance_surface.npz']}
 files=list((TRACK/'gate/accepted_masks').glob('*.png'));assert len(files)==100
 for p in files:inputs[str(p)]=digest(p)
 write(R/'repair_input_freeze.json',dict(status='FROZEN',group=GROUP,objects=expected,source_hashes=inputs,raw_cutoff=495,mapping_frame_count=100,tracking_and_gate_reused=True,selection='original A13/B12 before95_90',main_mapping_paused=True,GT_used=False))
 return inputs
def build_core():
 scene=r.Scene('room0');cases,association,obj,info,canonical,fixed=catalog(R)
 arrays=[];ids=[];audit=[]
 for case in cases:
  uid=case['case_uid'];fr=scene.source.load_frame(info[uid]['config']['seed_frame'])
  lab=np.asarray(Image.open(info[uid]['seedpath']));corelab=np.zeros(lab.shape,np.uint16)
  for item in case['objects']:
   tid=item['track_id'];corelab[binary_erosion(lab==tid,iterations=10)]=tid
  cp,cl=h.project(fr,corelab,scene)
  for item in case['objects']:
   tid=item['track_id'];p=np.unique(cp[cl==tid]);arrays.append(p);ids.append(fixed[canonical[tid]])
   audit.append(dict(track_id=tid,persistent_id=fixed[canonical[tid]],core_points=len(p)))
 owner,claimkeys=core_claims(arrays,ids,len(scene.xyz),association['instance_base']);union=np.flatnonzero(owner!=-1)
 h.f.atomic_npz(R/'modified_repair/room0/core_claims.npz',surface_point_index=union,claimed_persistent_id=owner[union],raw_point_ID_claims=claimkeys,unique_owned_surface_point_index=np.flatnonzero(owner>0),ambiguous_surface_point_index=np.flatnonzero(owner==-2))
 write(R/'modified_repair/room0/core_preflight.json',dict(objects=audit,unique_owned_points=int((owner>0).sum()),ambiguous_points=int((owner==-2).sum()),GT_used=False))
 return audit
def expand():
 e=load_script('_repair100_expand',BASEC/'expand_room0.py')
 e.h=h;e.O=R;e.R=R/'e1';e.A=R/'modified_repair/room0';e.C=BASEC
 e.main()
 return read(R/'e1/region_report.json')
def family_targets(accepted,obj,info,canonical):
 result={}
 for tid in accepted:
  for pid in info[obj[tid]]['config']['old_family_ids']:
   result.setdefault(int(pid),set()).add(canonical[tid])
 return result
def partial_layer(layer,merged,eligible):
 residual=layer.copy()
 for pid,targets in eligible.items():
  residual[(layer==pid)&np.isin(merged,list(targets))]=0
 return residual
def all_pixel_layer(frame,residual,scene,prior,owner,visible):
 image=residual.copy()
 yy,xx=np.nonzero((residual>0)&np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<10))
 z=frame.depth_m[yy,xx].astype(np.float64)
 cam=np.column_stack(((xx-frame.camera.cx)*z/frame.camera.fx,(yy-frame.camera.cy)*z/frame.camera.fy,z))
 world=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
 dist,p=scene.tree.query(world,k=1,workers=8);ok=np.isfinite(dist)&(dist<.015)
 p=p[ok];yy=yy[ok];xx=xx[ok]
 ids,_=obs.revise_all_history(p,image[yy,xx],np.ones(len(p),bool),prior,owner,visible)
 image[yy,xx]=ids
 return image
def transform_layer(frame,layer,merged,eligible,scene,prior,owner,visible,base):
 residual=partial_layer(layer,merged,eligible)
 p,pid,u,v=h.sourcepixels(replace(frame,mask_local=layer),scene)
 before=h.keys(p,pid,base)
 rp,ri=h.project(frame,residual,scene)
 available=residual[v,u]>0
 np.testing.assert_array_equal(h.keys(p[available],pid[available],base),h.keys(rp,ri,base))
 core=pid.copy();choose=available&(prior[p]>0)&(pid!=prior[p]);core[choose]=prior[p[choose]]
 final,changed=obs.revise_all_history(p,pid,available,prior,owner,visible)
 candidates={}
 for name,ids in zip(STAGES,(pid,core,final)):
  keys=h.keys(p[available],ids[available],base)
  image=residual.copy();image[v[available],u[available]]=ids[available]
  ip,ii=h.project(frame,image,scene);np.testing.assert_array_equal(h.keys(ip,ii,base),keys)
  candidates[name]=keys
 journal=all_pixel_layer(frame,residual,scene,prior,owner,visible)
 jp,ji=h.project(frame,journal,scene);np.testing.assert_array_equal(h.keys(jp,ji,base),candidates['whole_to_partial'])
 retired=np.setdiff1d(before,h.keys(rp,ri,base))//base
 return dict(before=before,candidates=candidates,journal=journal,retired=retired,
  retained_source_pixels=int(available.sum()),core_relabelled_pixels=int(choose.sum()),expanded_relabelled_pixels=int(changed.sum()))
def replay():
 started=time.perf_counter();scene=r.Scene('room0')
 scene.settings=r.AssignmentSettings()
 cases,association,obj,info,canonical,fixed=catalog(R);base=association['instance_base']
 decisions={x['frame']:x for x in read(R/'validated/full_track/frame_decisions.json')['frames']}
 assert sorted(decisions)==FIDS
 region=h.load(R/'e1/region.npz');prior,owner=region['prior_owner'],region['owner'];apoints=np.flatnonzero((prior!=-1)|(owner!=-1))
 bk,bv=h.encode(scene.pair,base)
 framekeys={name:[] for name in ('baseline',*STAGES)}
 removed={name:[] for name in STAGES};added={name:[] for name in STAGES};scopes={name:[] for name in STAGES}
 to_pid=np.zeros(max(fixed)+1,np.uint16)
 for cid,pid in fixed.items():to_pid[cid]=pid
 receipts=[];manifest=[];timings=dict(layer_partial_core_E1=0.,vote_audit=0.,journal_write=0.,visibility=0.)
 for index,fid in enumerate(FIDS,1):
  frame=scene.source.load_frame(fid);decision=decisions[fid]
  accepted=list(map(int,decision['accepted_track_ids']))
  merged=np.asarray(Image.open(R/'validated/full_track/accepted_masks'/f'f{fid:06d}.png'),np.uint16)
  assert set(np.unique(merged))<={0,*[canonical[t] for t in accepted]}
  layers=scene.identity_layers(fid,frame);assert len(layers)==(2 if fid<=245 else 1)
  tick=time.perf_counter();visible=np.zeros(len(scene.xyz),bool)
  visible[apoints],_,_=obs.visible_samples(scene.xyz[apoints],frame.camera_to_world,frame.camera,frame.depth_m,.02)
  timings['visibility']+=time.perf_counter()-tick
  tick=time.perf_counter();eligible=family_targets(accepted,obj,info,canonical)
  parts=[transform_layer(frame,im,merged,eligible,scene,prior,owner,visible,base) for im in layers]
  np_,nc=h.project(frame,merged,scene);new=h.keys(np_,to_pid[nc],base);adopted=to_pid[merged]
  oldraw=np.unique(cat([x['before'] for x in parts]));old_eff=strict_frame(oldraw,base)
  z=scene.support(fid);savedraw=h.keys(z['surface_point_index'],z['candidate_persistent_id'],base)
  np.testing.assert_array_equal(oldraw,savedraw)
  timings['layer_partial_core_E1']+=time.perf_counter()-tick
  tick=time.perf_counter();eff=dict(baseline=old_eff)
  framekeys['baseline'].append(old_eff);tx=dict(baseline_candidate_keys=oldraw,baseline_frame_keys=old_eff)
  directscope=np.unique(np.r_[cat([x['retired'] for x in parts]),np_]).astype(np.int32)
  assert np.all(np.isin(directscope,np_))
  for name in STAGES:
   raw=np.unique(np.r_[cat([x['candidates'][name] for x in parts]),new]);eff[name]=strict_frame(raw,base)
   assert len(eff[name])==len(np.unique(eff[name]//base))
   parent=PARENT[name];rem=np.setdiff1d(eff[parent],eff[name]);add=np.setdiff1d(eff[name],eff[parent])
   scope=np.unique(np.r_[rem//base,add//base]).astype(np.int32)
   if name=='direct':assert np.all(np.isin(scope,directscope));scope=directscope
   elif name=='core_all':assert np.all(prior[scope]>0)
   else:assert np.all((owner[scope]!=-1)|(prior[scope]!=-1))
   removed[name].append(rem);added[name].append(add);scopes[name].append(scope);framekeys[name].append(eff[name])
   tx[name+'_candidate_keys']=raw;tx[name+'_frame_keys']=eff[name]
  h.f.atomic_npz(R/'partial/transactions'/f'f{fid:06d}.npz',**tx)
  timings['vote_audit']+=time.perf_counter()-tick
  tick=time.perf_counter();journal=R/'journal';journal.mkdir(exist_ok=True)
  output=[x['journal'] for x in parts]+[adopted];rows=[]
  for j,image in enumerate(output):
   path=journal/f'f{fid:06d}_layer{j}.png';Image.fromarray(image.astype(np.uint16)).save(path)
   rows.append(dict(path=str(path),sha256=digest(path),kind='history' if j<len(parts) else 'adopted100'))
  manifest.append(dict(frame_id=fid,old_layer_count=len(parts),layer_files=rows))
  receipts.append(dict(frame=fid,old_layers=len(parts),accepted_tracks=accepted,baseline_candidates_exact=True,
   counts=[{k:v for k,v in x.items() if k in ('retained_source_pixels','core_relabelled_pixels','expanded_relabelled_pixels')} for x in parts]))
  timings['journal_write']+=time.perf_counter()-tick
  if index%10==0:emit('partial_core_E1_replay',frames=index,total=100,seconds=time.perf_counter()-started)
 pk,pv=aggregate(framekeys.pop('baseline'));np.testing.assert_array_equal(pk,bk);np.testing.assert_array_equal(pv,bv)
 baseline_evidence=r.reduce_surface_votes(bk,bv,len(scene.xyz),base,2,.67)
 for field in r.FIELDS:np.testing.assert_array_equal(baseline_evidence[field],scene.evidence[field])
 write(R/'journal/manifest.json',dict(status='PASS',group=GROUP,frames=manifest,old_committed50=scene.commit,raw_cutoff=495))
 results=dict(baseline=(bk,bv,scene.evidence,scene.final));reports={};inventory=np.unique(scene.final[scene.final>0])
 for name in STAGES:
  tick=time.perf_counter();pk,pv,pe,pf=results[PARENT[name]]
  rem,add=cat(removed.pop(name)),cat(added.pop(name));nk,nv=h.batch_delta(pk,pv,rem,add)
  backk,backv=h.batch_delta(nk,nv,add,rem);np.testing.assert_array_equal(backk,pk);np.testing.assert_array_equal(backv,pv)
  fullk,fullv=aggregate(framekeys.pop(name));np.testing.assert_array_equal(nk,fullk);np.testing.assert_array_equal(nv,fullv)
  scope=np.unique(cat(scopes.pop(name))).astype(np.int32)
  if name=='whole_to_partial':
   common,ni,pi=np.intersect1d(nk,pk,return_indices=True)
   scope=np.unique(np.r_[np.setxor1d(nk,pk)//base,common[nv[ni]!=pv[pi]]//base]).astype(np.int32)
  monitor=np.zeros(len(scene.xyz),bool);monitor[scope]=True
  np.testing.assert_array_equal(nk[~monitor[nk//base]],pk[~monitor[pk//base]])
  np.testing.assert_array_equal(nv[~monitor[nk//base]],pv[~monitor[pk//base]])
  evidence=r.reduce_surface_votes(nk,nv,len(scene.xyz),base,2,.67)
  for field in r.FIELDS:np.testing.assert_array_equal(evidence[field][~monitor],pe[field][~monitor])
  native=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
  pair=h.pairtable(nk,nv,base);emit('stage_postprocess',stage_name=name)
  variants,_,post=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native,scene.settings)
  final=pf.copy();final[scope]=variants['holes_geodesic'][scope]
  np.testing.assert_array_equal(final[~monitor],pf[~monitor])
  folder=R/'partial'/name/'room0'
  for kind,lab in [('native',native),('final',final)]:
   inv=np.union1d(inventory,np.unique(lab[lab>0])).astype(np.int32)
   h.f.atomic_npz(folder/kind/'instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=lab,native_instance_ids=inv)
  h.f.atomic_npz(folder/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**evidence)
  h.f.atomic_npz(folder/'native/surface_instance_frame_votes.npz',**pair)
  h.f.atomic_npz(folder/'allowed_surface_ids.npz',surface_point_index=scope)
  reports[name]=dict(status='PASS',parent=PARENT[name],scope_points=len(scope),removed_votes=len(rem),added_votes=len(add),final_changed_vs_parent=int((final!=pf).sum()),summary=h.summary(evidence,final),postprocessing=post,seconds=time.perf_counter()-tick)
  write(folder/'complete.json',reports[name]);results[name]=(nk,nv,evidence,final)
  del variants,fullk,fullv,backk,backv,rem,add;gc.collect()
 scene.verify_unchanged()
 write(R/'partial/frame_receipts.json',receipts)
 report=dict(status='PASS',seconds=time.perf_counter()-started,stages=reports,timings=timings,
  baseline_multilayer_candidates_votes_evidence_exact=True,full_replay_equals_delta=True,exact_rollback=True,
  strict_same_frame_point_vote=True,stages_committed_in_original_order=True,whole_control_replayed=False,raw_RGB_cutoff=495,GT_used=False)
 write(R/'partial/mapping_complete.json',report)
 return report
def main():
 start=time.perf_counter();started_unix=time.time();input_hashes=setup();times={}
 for name,fn in [('core_region',build_core),('expand',expand),('partial_core_E1',replay)]:
  emit(name);stamp=R/('stage_'+name+'_complete.json');tick=time.perf_counter()
  if stamp.exists():times[name]=dict(**read(stamp),reused_stage=True,seconds_this_invocation=0.)
  else:
   result=fn();times[name]=dict(status='PASS',seconds=time.perf_counter()-tick,result=result)
   write(stamp,times[name])
 for path,sha in input_hashes.items():assert digest(path)==sha,path
 report=dict(status='PASS',group=GROUP,objects=13 if GROUP=='A' else 12,seconds=time.perf_counter()-start,
  started_unix=started_unix,ended_unix=time.time(),timings=times,
  source_hashes_unchanged=True,tracking_reused=True,gate_reused=True,main_mapping_paused=True,
  raw_cutoff=495,mapping_frames=100,committed_for_continuation=False,GT_used=False,
  peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_io=Path('/proc/self/io').read_text(),
  implementation='unchanged partial+core+E1 applied independently to all current identity layers before strict union')
 write(R/'repair_complete.json',report);emit('repair_complete',repair_seconds=report['seconds'])
if __name__=='__main__':
 try:main()
 except Exception:write(R/'repair_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise

import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,traceback,concurrent.futures,re
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
C=Path('/home/chenkejun/CVPR/experiments/scannet_ovimap_strict_dbscan_20261104')
sys.path.insert(0,str(C))
from dbscan_common import O,SRC,read,load,save,dump,sha,COUNTS
from dbscan_common import protocol,run
sys.path.insert(0,str(C/'policies/main'));sys.path.insert(0,str(SRC))
from scene_completion.core import complete
from scene_completion.pipeline import write_scene_complete,finalize
from revisable_instance_map.surface_evidence import reduce_surface_votes
T=O/'dbscan/base'
SCENES=['office1','room0'];VARIANTS=['main','online_B']
ARMS=['fill_only','merge_only','merge_fill']
CFG=dict(voxel_m=.01,eps_m=.05,min_samples=10,min_component_cells=30,
 overlap_min=.80,reciprocal_min=.15,max_size_ratio=6.,min_shared_cells=100,
 observation_membership_radius_m=.04,observation_both_fraction=.70,
 observation_min_voxels=20,min_positive_frames=3,min_frame_span=50,
 fill_radius_m=.05,fill_neighbors=32,fill_normal_cos=.85,fill_plane_m=.0075,
 fill_min_same_cells=8,fill_dominance=.90,fill_unobserved_dominance=.95,
 fill_conflict_min_fraction=.25,fill_conflict_relative_to_top=.5,
 anchor_min_votes=4,anchor_min_confidence=.8,create_new_ids=False,
 independent_raw_observation_only=True,repair_coselected_ids_are_cannot_link=True,independent_covisible_cannot_link=True,min_covisible_frames=3,min_covisible_span=50)

def strict(keys,base):
 keys=np.unique(keys);p=keys//base
 u,start,count=np.unique(p,return_index=True,return_counts=True)
 return keys[start[count==1]]

def encode(pair,base):
 return pair['surface_point_index'].astype(np.int64)*base+pair['instance_id']

def voxel_reps(xyz):
 cells=np.rint(xyz/CFG['voxel_m']).astype(np.int32)
 u,ix,inv=np.unique(cells,axis=0,return_index=True,return_inverse=True)
 return u,ix,inv

def build_candidates(xyz,labels,pair,pixel,out,repair_branch=None,allow_repair_history=True):
 # Decisions are made without GT, evaluator outputs, or another method's labels.
 cells,rep,inv=voxel_reps(xyz);centers=cells.astype(float)*CFG['voxel_m']
 ids,counts=np.unique(labels[labels>0],return_counts=True)
 eligible=ids[counts>=100]
 supports={};core={};trees={};cluster_audit=[]
 for iid in eligible:
  pi=pair['surface_point_index'][(pair['instance_id']==iid)&(pair['frame_votes']>=2)]
  ss=np.unique(inv[pi]);supports[int(iid)]=ss
  if len(ss)<CFG['min_component_cells']:continue
  cl=DBSCAN(eps=CFG['eps_m'],min_samples=CFG['min_samples'],algorithm='kd_tree',n_jobs=1).fit_predict(centers[ss])
  vals,n=np.unique(cl[cl>=0],return_counts=True)
  valid=vals[n>=CFG['min_component_cells']]
  # Keep supported components for evidence, never delete TSDF vertices or existing labels.
  cc=ss[np.isin(cl,valid)]
  if len(cc)<CFG['min_component_cells']:continue
  core[int(iid)]=cc;trees[int(iid)]=cKDTree(centers[cc])
  cluster_audit.append(dict(id=int(iid),support_cells=len(ss),core_cells=len(cc),components=len(valid)))
 # Independent repair selection provides a conservative cannot-link relation.
 # Never undo a checkpoint's distinct accepted objects using their own propagated masks.
 branch=Path(repair_branch) if repair_branch is not None else Path(pixel).parents[2];scene_root=branch.parents[1];group=branch.name
 separations={}
 for cp in (sorted(int(p.name) for p in scene_root.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)<=int(branch.parent.name)) if allow_repair_history else []):
  ap=scene_root/str(cp)/group/'repair/global_association.json'
  if not ap.exists():continue
  entries=read(ap).get('objects',[]);chosen=sorted({int(e['persistent_id']) for e in entries})
  for ai,a in enumerate(chosen):
   for b in chosen[ai+1:]:separations.setdefault((a,b),[]).append(cp)
 pixel_data=load(pixel);independent={}
 for obs,iid,fid,layer,lo,hi in zip(pixel_data['observation_ids'],pixel_data['instance_ids'],pixel_data['frame_ids'],pixel_data['layer_index'],pixel_data['offsets'][:-1],pixel_data['offsets'][1:]):
  if int(layer)==0 and int(hi-lo)>=20 and re.fullmatch(r'Replica/[^/]+/f[0-9]+/m[0-9]+',str(obs)):
   independent.setdefault(int(iid),set()).add(int(fid))
 rows=[];pairs=[];ids=sorted(core)
 for ai,i in enumerate(ids):
  for j in ids[ai+1:]:
   a,b=core[i],core[j];inter=np.intersect1d(a,b,assume_unique=True)
   x,y=len(inter)/len(a),len(inter)/len(b)
   if len(inter)<30 or max(x,y)<.3:continue
   reasons=[]
   covisible=sorted(independent.get(i,set())&independent.get(j,set()))
   if len(covisible)>=3 and covisible[-1]-covisible[0]>=50:reasons.append('independent_distinct_raw_masks_covisible_multiple_frames')
   if (i,j) in separations:reasons.append('distinct_accepted_objects_at_same_repair_checkpoint')
   if max(x,y)<CFG['overlap_min']:reasons.append('insufficient_directed_overlap')
   if min(x,y)<CFG['reciprocal_min']:reasons.append('insufficient_reciprocal_overlap')
   if max(len(a),len(b))/min(len(a),len(b))>CFG['max_size_ratio']:reasons.append('part_vs_very_large_surface')
   if len(inter)<CFG['min_shared_cells']:reasons.append('small_shared_region')
   row=dict(ids=[i,j],core_cells=[len(a),len(b)],shared_cells=len(inter),overlap=[x,y],rejection=reasons,separate_checkpoints=separations.get((i,j),[]),independent_covisible_frames=len(covisible),covisible_first_last=[covisible[0],covisible[-1]] if covisible else [],covisible_sample=covisible[:10])
   rows.append(row)
   if not reasons:pairs.append(row)
 # Verify each pair using distinct historical observations, including repaired source layers.
 z=pixel_data;off=z['offsets'];obsids=z['instance_ids'];frames=z['frame_ids'];vox=z['sampled_voxel_coordinates'];scale=float(z['voxel_size_m'][0])
 for row in pairs:
  i,j=row['ids'];positive=[];tested=0
  for k in np.flatnonzero(np.isin(obsids,[i,j])):
   if not re.fullmatch(r'Replica/[^/]+/f[0-9]+/m[0-9]+',str(z['observation_ids'][k])) or int(z['layer_index'][k])!=0:continue
   q=np.unique(vox[off[k]:off[k+1]],axis=0)
   if len(q)<CFG['observation_min_voxels']:continue
   q=(q.astype(float)+.5)*scale
   da=trees[i].query(q,k=1,workers=1)[0];db=trees[j].query(q,k=1,workers=1)[0]
   fa=float(np.mean(da<=CFG['observation_membership_radius_m']));fb=float(np.mean(db<=CFG['observation_membership_radius_m']))
   tested+=1
   if min(fa,fb)>=CFG['observation_both_fraction']:
    positive.append(dict(frame=int(frames[k]),observation=str(z['observation_ids'][k]),fractions=[fa,fb],voxels=len(q)))
  unique=sorted({x['frame'] for x in positive})
  row.update(positive_frames=unique,tested_observations=tested,positive_examples=positive[:6])
  if len(unique)<CFG['min_positive_frames'] or unique[-1]-unique[0]<CFG['min_frame_span']:
   row['rejection'].append('insufficient_separated_multiframe_support')
 # Avoid transitive chain merges: every pair in an accepted family must pass.
 good={tuple(r['ids']):r for r in rows if not r['rejection']}
 families={i:{i} for i in ids};owner={i:i for i in ids};accepted=[]
 for row in sorted(good.values(),key=lambda r:(-max(r['overlap']),-r['shared_cells'],r['ids'])):
  i,j=row['ids'];a,b=owner[i],owner[j]
  if a==b:continue
  aa,bb=families[a],families[b]
  if not all(tuple(sorted((x,y))) in good for x in aa for y in bb):
   row['rejection'].append('incomplete_family_pair_evidence');continue
  union=aa|bb;c=min(union);families.pop(a);families.pop(b);families[c]=union
  for x in union:owner[x]=c
  accepted.append(row['ids'])
 fam=[sorted(x) for x in families.values() if len(x)>1]
 lut=np.arange(int(max(pair['instance_id'].max(),labels.max()))+1,dtype=np.int32)
 for f in fam:lut[f]=min(f)
 report=dict(settings=CFG,GT_used=False,objects=cluster_audit,candidates=rows,accepted_edges=accepted,families=fam,
  feature_gate='independent original source-mask observations only; checkpoint co-selected objects cannot link; not CLIP',revision='guarded_v2',first_trial_preserved=str(O/'ablations/dbscan_identity_completion_20261102'),
  no_TSDF_points_deleted=True)
 dump(out/'merge_decisions.json',report)
 save(out/'identity_map.npz',old_id=np.arange(len(lut),dtype=np.int32),canonical_id=lut)
 return lut,report

def replay_votes(root,pair,base,lut,out,original_only=False):
 keys=encode(pair,base);order=np.argsort(keys);keys=keys[order];votes=pair['frame_votes'][order]
 counters=np.zeros(len(keys),np.int32);extra=[];sources=[]
 transactions=root/'repair/partial/transactions'
 use_transactions=transactions.exists() and not original_only
 frames=sorted(transactions.glob('f*.npz')) if use_transactions else sorted((root/'mapping/baseline/frame_support').glob('f*.npz'))
 assert len(frames)==int(root.parent.name)
 for idx,f in enumerate(frames):
  with np.load(f) as z:
   if use_transactions:raw=z['whole_to_partial_candidate_keys'];old=z['whole_to_partial_frame_keys']
   else:
    raw=np.unique(z['surface_point_index'].astype(np.int64)*base+z['candidate_persistent_id']);old=strict(raw,base)
  np.testing.assert_array_equal(strict(raw,base),old)
  ix=np.searchsorted(keys,old);np.testing.assert_array_equal(keys[ix],old);counters[ix]+=1
  oldpid=raw%base;new=raw//base*base+lut[oldpid]
  nf=strict(new,base);remapped=old//base*base+lut[old%base]
  assert len(np.unique(remapped))==len(remapped) # original counted votes already unique per frame-point
  add=np.setdiff1d(nf,remapped,assume_unique=True)
  assert not len(np.setdiff1d(remapped,nf,assume_unique=True))
  extra.append(add);sources.append(dict(frame=int(f.stem[1:]),sha256=sha(f),recovered_abstained_votes=len(add)))
  if (idx+1)%100==0:print('REPLAY',out.parent.name,out.name,idx+1,flush=True)
 np.testing.assert_array_equal(counters,votes)
 remapped=keys//base*base+lut[keys%base]
 allk=np.r_[remapped,np.concatenate(extra)];allv=np.r_[votes,np.ones(sum(map(len,extra)),np.int32)]
 order=np.argsort(allk);allk=allk[order];allv=allv[order]
 kk,start=np.unique(allk,return_index=True);vv=np.add.reduceat(allv,start).astype(np.int32)
 assert vv.max()<=int(root.parent.name)
 merged=dict(surface_point_index=(kk//base).astype(np.int32),instance_id=(kk%base).astype(np.int32),frame_votes=vv)
 save(out/'merged_votes.npz',**merged)
 dump(out/'frame_replay_audit.json',dict(status='PASS',frames=len(sources),original_pair_votes_exact=True,
  same_frame_duplicates_counted_once=True,formerly_abstained_votes_recovered=int(sum(map(len,extra))),sources=sources))
 return kk,vv,merged

def fill_gray(xyz,normals,valid,labels,ev,pair,out):
 result=labels.copy();targets=np.flatnonzero((labels<0)&valid)
 seed=np.flatnonzero((ev['state']==2)&(ev['top1_votes']>=CFG['anchor_min_votes'])&(ev['confidence']>=CFG['anchor_min_confidence'])&valid&(labels==ev['top1_instance_id']))
 _,unique,_=voxel_reps(xyz[seed]);seed=seed[unique]
 tree=cKDTree(xyz[seed]);proposal=np.full(len(targets),-1,np.int32);dominance=np.zeros(len(targets),np.float32)
 base=int(pair['instance_id'].max())+1;pk=encode(pair,base);order=np.argsort(pk);pk=pk[order];pv=pair['frame_votes'][order]
 for lo in range(0,len(targets),10000):
  q=targets[lo:lo+10000];d,j=tree.query(xyz[q],k=CFG['fill_neighbors'],distance_upper_bound=CFG['fill_radius_m'],workers=2)
  ok=np.isfinite(d)&(j<len(seed));ix=seed[np.minimum(j,len(seed)-1)];delta=xyz[ix]-xyz[q,None]
  dot=np.einsum('qkc,qc->qk',normals[ix],normals[q])
  plane1=np.abs(np.einsum('qkc,qc->qk',delta,normals[q]));plane2=np.abs(np.sum(delta*normals[ix],axis=2))
  ok&=(dot>=CFG['fill_normal_cos'])&(plane1<=CFG['fill_plane_m'])&(plane2<=CFG['fill_plane_m'])
  labs=labels[ix];weight=np.where(ok,np.exp(-d/.02),0)
  eq=labs[:,:,None]==labs[:,None,:]
  score=np.sum(eq*weight[:,None,:],axis=2);score[~ok]=0
  best=score.argmax(1);rr=np.arange(len(q));cid=labs[rr,best]
  frac=np.divide(score[rr,best],weight.sum(1),out=np.zeros(len(q)),where=weight.sum(1)>0)
  same=np.sum(eq[rr,best,:]&ok,axis=1)
  keep=(same>=CFG['fill_min_same_cells'])&(frac>=CFG['fill_dominance'])
  keep&=(ev['state'][q]!=0)|(frac>=CFG['fill_unobserved_dominance'])
  # Spatial inference cannot ignore strong historical disagreement.
  keep&=(ev['state'][q]!=1)|(cid==ev['top1_instance_id'][q])
  wanted=q.astype(np.int64)*base+cid;pos=np.searchsorted(pk,wanted);has=(pos<len(pk))&(pk[np.minimum(pos,len(pk)-1)]==wanted)
  own=np.where(has,pv[np.minimum(pos,len(pv)-1)],0)
  conflict=ev['state'][q]==3
  keep&=~conflict|((own>=2)&(own>=CFG['fill_conflict_min_fraction']*ev['total_frame_votes'][q])&(own>=CFG['fill_conflict_relative_to_top']*ev['top1_votes'][q]))
  proposal[lo:lo+len(q)]=np.where(keep,cid,-1);dominance[lo:lo+len(q)]=frac
 cluster_rows=[]
 for iid in np.unique(proposal[proposal>0]):
  ti=np.flatnonzero(proposal==iid);qp=targets[ti]
  near=np.unique(np.concatenate(tree.query_ball_point(xyz[qp],CFG['fill_radius_m'],workers=2)))
  anchors=seed[near];anchors=anchors[labels[anchors]==iid]
  allidx=np.r_[anchors,qp];cells,rep,inv=voxel_reps(xyz[allidx])
  cl=DBSCAN(eps=CFG['eps_m'],min_samples=CFG['min_samples'],algorithm='kd_tree',n_jobs=1).fit_predict(cells.astype(float)*CFG['voxel_m'])
  seedcl=cl[np.unique(inv[:len(anchors)])];vals,cnt=np.unique(seedcl[seedcl>=0],return_counts=True)
  accepted=vals[cnt>=10];good=np.isin(cl[inv[len(anchors):]],accepted)
  result[qp[good]]=int(iid)
  cluster_rows.append(dict(id=int(iid),proposed=len(qp),accepted=int(good.sum()),anchor_cells=len(np.unique(inv[:len(anchors)])),accepted_clusters=len(accepted)))
 assert np.array_equal(result[labels>0],labels[labels>0])
 changed=np.flatnonzero(result!=labels)
 save(out/'fill_diagnostics.npz',target_point_index=targets,proposed_id=proposal,spatial_dominance=dominance,newly_assigned_index=changed,newly_assigned_id=result[changed])
 dump(out/'fill_report.json',dict(status='PASS',settings=CFG,original_anchors=len(seed),target=len(targets),newly_assigned=len(changed),
  by_native_state={str(i):int(np.count_nonzero(ev['state'][changed]==i)) for i in range(4)},clusters=cluster_rows,
  original_assigned_unchanged=True,newly_filled_points_used_as_seeds=False,GT_used=False))
 return result

def evaluate(v,s,arm,original,labels,out,provenance):
 out.mkdir(parents=True,exist_ok=True)
 path=out/'instance_surface.npz';save(path,xyz_m=original['xyz_m'],rgb=original['rgb'],instance_id=labels)
 source=out/'provenance.json';dump(source,dict(status='PASS',variant=v,scene=s,arm=arm,rule='top1>=2,ratio>=0.67; spatial inference explicitly separate',**provenance))
 oldadapt=read(O/'evaluation'/v/s/'r2/adapter_manifest.json')
 run(['adapt-surface','--gt',oldadapt['gt_file'],'--surface',path,'--fixed-correspondence',oldadapt['correspondence_cache'],
  '--source-provenance',source,'--method-name',v+'_dbscan_'+arm,'--method-commit',sha(__file__),'--out',out/'r2'],out/'adapt.log')
 a=read(out/'r2/adapter_manifest.json')
 for k in ['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256']:assert a[k]==oldadapt[k],k
 assert sha(a['correspondence_cache'])==sha(oldadapt['correspondence_cache'])
 old=original['instance_id'];report=dict(status='PASS',variant=v,scene=s,arm=arm,original_prediction=str(O/'published'/v/s/'final_B/instance_surface.npz'),temporary_prediction=str(path),
  original_gray=int((old<0).sum()),temporary_gray=int((labels<0).sum()),gray_to_assigned=int(((old<0)&(labels>0)).sum()),assigned_to_gray=int(((old>0)&(labels<0)).sum()),
  assigned_changed_ID=int(((old>0)&(labels>0)&(old!=labels)).sum()),confirmed_ties=0,
  original_metrics=read(O/'evaluation'/v/s/'r2/metrics.json')['summary'],temporary_metrics=read(out/'r2/metrics.json')['summary'],
  prediction_sha256=sha(path),fixed_correspondence_sha256=sha(a['correspondence_cache']),geometry_GT_protocol_equal=True)
 dump(out/'complete.json',report);return report

def one(v,s):
 out=T/v/s;out.mkdir(parents=True,exist_ok=True);start=time.time()
 if (out/'complete.json').exists():return read(out/'complete.json')
 m=read(O/'published'/v/s/'scene_complete.json')
 for k,p in m['inputs'].items():assert sha(p)==m['input_sha256'][k]
 src=load(m['inputs']['surface']);original=load(O/'published'/v/s/'final_B/instance_surface.npz');ev=load(m['inputs']['evidence'])
 xyz=src['xyz_m'];n=len(xyz);attrs=load(O/'attributes'/s/str(COUNTS[s])/'normals.npz');rgbattrs=load(O/'attributes'/s/str(COUNTS[s])/'rgb.npz')
 valid=attrs['normal_valid']&rgbattrs['color_valid'];normals=attrs['normals'];np.testing.assert_array_equal(attrs['xyz_m'],xyz)
 pair=load(Path(m['inputs']['evidence']).with_name('surface_instance_frame_votes.npz'))
 root=O/('main' if v in ['main','no_repair'] else 'online')/s/str(COUNTS[s])/('A' if v in ['main','no_repair'] else v[-1])
 base=read(root/'repair/global_association.json')['instance_base'] if v!='no_repair' else int(load(root/'mapping/association/observation_support_3cm.npz')['next_instance_id'][0])
 old=reduce_surface_votes(encode(pair,base),pair['frame_votes'],n,base,2,.67)
 for k,a in old.items():np.testing.assert_array_equal(a,ev[k])
 ctl,_,_=complete(xyz,normals,src['instance_id'],old);np.testing.assert_array_equal(ctl,original['instance_id'])
 print('CONTROL_PASS',v,s,flush=True)
 pixel=root/'repair/commit_candidate/source_pixel_support.npz' if v!='no_repair' else O/'dbscan_inputs/no_repair'/s/'source_pixel_support.npz'
 lut,decisions=build_candidates(xyz,original['instance_id'],pair,pixel,out,repair_branch=root,allow_repair_history=v!='no_repair')
 if len(lut)<base:lut=np.r_[lut,np.arange(len(lut),base,dtype=np.int32)]
 print('FAMILIES',v,s,decisions['families'],flush=True)
 fillout=out/'fill_only';fillout.mkdir(exist_ok=True)
 fill=fill_gray(xyz,normals,valid,original['instance_id'],old,pair,fillout)
 # Only replay when a family actually merges; otherwise preserve exact original evidence.
 if decisions['families']:
  kk,vv,mp=replay_votes(root,pair,base,lut,out,original_only=v=='no_repair');merged=reduce_surface_votes(kk,vv,n,base,2,.67)
 else:merged=old;mp=pair;dump(out/'frame_replay_audit.json',dict(status='PASS',no_identity_changes=True,original_votes_reused=True))
 lab=src['instance_id'].copy();assigned=lab>0;lab[assigned]=lut[lab[assigned]]
 was=old['state']==2;now=merged['state']==2
 lab[was&~now]=-1;lab[now]=merged['top1_instance_id'][now]
 mergeout=out/'merge_only';mergeout.mkdir(exist_ok=True)
 save(mergeout/'pre_B.npz',xyz_m=xyz,rgb=src['rgb'],instance_id=lab)
 save(mergeout/'native.npz',xyz_m=xyz,rgb=src['rgb'],instance_id=np.where(now,merged['top1_instance_id'],-1).astype(np.int32))
 save(mergeout/'evidence.npz',xyz_m=xyz,rgb=src['rgb'],**merged)
 manifest=mergeout/'scene_complete.json'
 write_scene_complete(manifest,scene=s,workflow=m['workflow'],frame_ids=m['frame_ids'],inputs={'surface':mergeout/'pre_B.npz','native':mergeout/'native.npz','evidence':mergeout/'evidence.npz','normal_geometry':O/'attributes'/s/str(COUNTS[s])/'normals.npz'},final_processing_complete=True,final_repair_complete=True)
 receipt=finalize(manifest,mergeout/'final_B');merged_labels=load(receipt['final_surface'])['instance_id']
 bothout=out/'merge_fill';bothout.mkdir(exist_ok=True)
 both=fill_gray(xyz,normals,valid,merged_labels,merged,mp,bothout)
 # Freeze decisions and all outputs before evaluation; no GT-guided pair selection.
 for arm,labs in [('fill_only',fill),('merge_only',merged_labels),('merge_fill',both)]:save(out/arm/'frozen_labels.npz',instance_id=labs)
 dump(out/'decision_freeze.json',dict(status='FROZEN_BEFORE_EVALUATION',settings=CFG,families=decisions['families'],GT_used=False,
  outputs={a:sha(out/a/'frozen_labels.npz') for a in ARMS},code_sha256=sha(__file__)))
 rows=[]
 for arm,labs in [('fill_only',fill),('merge_only',merged_labels),('merge_fill',both)]:
  rows.append(evaluate(v,s,arm,original,labs,out/arm,dict(settings=CFG,families=decisions['families'] if arm!='fill_only' else [],GT_used_for_decision=False,
   original_control_exact=True,terminal_only=True,no_online_feedback=True,final_B_once=True,decision_freeze_sha256=sha(out/'decision_freeze.json'))))
 for k,p in m['inputs'].items():assert sha(p)==m['input_sha256'][k]
 result=dict(status='PASS',variant=v,scene=s,seconds=time.time()-start,families=decisions['families'],arms=rows)
 dump(out/'complete.json',result);print('CASE_PASS',v,s,round(result['seconds']),flush=True);return result

def main():
 os.nice(10);T.mkdir(parents=True,exist_ok=True);protocol()
 dump(T/'plan.json',dict(status='AUTHORIZED',approval='User: use professional judgment, try a good temporary version and show metrics/PLY; do not ask again',
  scenes=SCENES,variants=VARIANTS,arms=['original']+ARMS,settings=CFG,production_unchanged=True,
  source_reference='ConceptGraphs object-wise DBSCAN and separate overlap merge; raw multiframe observations and co-selection cannot-link protection',revision='guarded_v2',development_iteration='V1 showed room0 degradation; generic provenance guards added, no GT-based ID whitelist; V1 retained',
  protocol='main object_observed_repair revision2',frozen=False,code_sha256=sha(__file__)))
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(one,v,s) for s in SCENES for v in VARIANTS]
  for f in concurrent.futures.as_completed(fs):
   rows.append(f.result());dump(T/'progress.json',dict(status='RUNNING',completed=len(rows),total=4))
 pools={};gt=None
 for v in VARIANTS:
  pools[v]={}
  for arm in ['original']+ARMS:
   dest=T/'pooled'/v/arm;dest.mkdir(parents=True,exist_ok=True);batch=[]
   for s in SCENES:
    p=O/'evaluation'/v/s/'r2' if arm=='original' else T/v/s/arm/'r2'
    a=read(p/'adapter_manifest.json');batch.append(dict(gt=a['gt_file'],prediction=str(p/'canonical_prediction.npz'),diagnostic_prediction=str(p/'diagnostic_support_prediction.npz')))
   dump(dest/'batch.json',dict(scenes=batch));run(['eval-batch','--scenes',dest/'batch.json','--out',dest],dest/'run.log')
   z=read(dest/'summary.json');count=z['CA_PRF1_0_5']['TP']+z['CA_PRF1_0_5']['FN']
   if gt is None:gt=count
   assert gt==count;pools[v][arm]=z
 dump(T/'comparison.json',dict(status='PASS',scenes=SCENES,GT=gt,pooled=pools,per_scene=rows,production_results_unchanged=True,
  limitation='Two-scene development ablation, not eight-scene benchmark; geometry/GT/correspondence fixed; no end-to-end rerun'))
 dump(T/'progress.json',dict(status='PASS',completed=4,total=4));print('ALL_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(T/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,re,traceback,importlib.util,concurrent.futures,argparse
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
spec=importlib.util.spec_from_file_location('guarded',C/'dbscan_guarded.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
O=g.O;OLD=g.T;PREVIOUS=O/'ablations/dbscan_history_seeded_20261102';T=O/'dbscan';BASE=T/'base';g.T=BASE
CFG=dict(voxel_m=.01,eps_m=.05,min_samples=10,min_cluster_cells=30,min_historical_core_cells=5,
 correspondence_max_m=.012,region_padding_m=.03,normal_cos=.85,plane_m=.0075,
 min_votes=4,min_fraction=.25,min_relative_to_top=.5,historical_core_min_votes=4,historical_core_min_confidence=.8,
 retained_committed_lineage_required=True,distinct_new_identity_only=True,previous_checkpoints_only=True,
 current_competing_repair_core_protected=True,assigned_points_unchanged=True,no_new_ID=True,GT_used=False)
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def core_completion(v,s,basecase,out):
 start=time.time();manifest=g.read(O/'published'/v/s/'scene_complete.json')
 for k,p in manifest['inputs'].items():assert g.sha(p)==manifest['input_sha256'][k]
 original=g.load(O/'published'/v/s/'final_B/instance_surface.npz');baseline=g.load(basecase/'merge_fill/instance_surface.npz');labels=baseline['instance_id'];xyz=baseline['xyz_m'];n=len(xyz)
 ev=g.load(basecase/'merge_only/evidence.npz');pair=g.load(basecase/'merged_votes.npz') if (basecase/'merged_votes.npz').exists() else g.load(Path(manifest['inputs']['evidence']).with_name('surface_instance_frame_votes.npz'))
 lut=g.load(basecase/'identity_map.npz')['canonical_id'];group='A' if v=='main' else v[-1];branch=O/('main' if v=='main' else 'online')/s
 root=branch/str(g.COUNTS[s])/group;pixel=g.load(root/'repair/commit_candidate/source_pixel_support.npz')
 active={}
 for name,pid in zip(pixel['observation_ids'],pixel['instance_ids']):
  m=re.search(r'/repair(\d+)([AB])/',str(name))
  if m and m.group(2)==group:active.setdefault(int(m.group(1)),set()).add(int(pid))
 attrs=g.load(O/'attributes'/s/str(g.COUNTS[s])/'normals.npz');rgb=g.load(O/'attributes'/s/str(g.COUNTS[s])/'rgb.npz');norm=attrs['normals'];valid=attrs['normal_valid']&rgb['color_valid']
 current=g.load(root/'repair/e1/region.npz')['core_owner'];assert len(current)==n
 current=np.where(current>0,lut[np.clip(current,0,len(lut)-1)],current)
 tree=cKDTree(xyz);proposed=np.full(n,-1,np.int32);ambiguous=np.zeros(n,bool);rows=[]
 pairids=pair['instance_id'];pairpts=pair['surface_point_index'];pairvotes=pair['frame_votes']
 for cp in sorted(active):
  if cp>=g.COUNTS[s]:continue
  r=branch/str(cp)/group
  if not (r/'repair/global_association.json').exists():continue
  assoc=g.read(r/'repair/global_association.json')
  targets=[x for x in assoc['objects'] if x['association']=='distinct_new_identity' and x['persistent_id'] in active[cp]]
  if not targets:continue
  oldev=g.load(r/f'repair/partial/whole_to_partial/{s}/native/surface_evidence.npz');oldxyz=oldev['xyz_m'];oldregion=g.load(r/'repair/e1/region.npz')
  for entry in targets:
   oldid=int(entry['persistent_id']);iid=int(lut[oldid]);row=dict(checkpoint=cp,source_id=oldid,canonical_id=iid,track=entry['canonical_track_id']);rows.append(row)
   core=np.flatnonzero((oldregion['core_owner']==oldid)&(oldev['state']==2)&(oldev['top1_instance_id']==oldid)&(oldev['top1_votes']>=CFG['historical_core_min_votes'])&(oldev['confidence']>=CFG['historical_core_min_confidence']))
   region=np.flatnonzero(oldregion['owner']==oldid)
   row['old_core_points']=len(core);row['old_region_points']=len(region)
   if len(core)<5 or not len(region):row['status']='insufficient_historical_core';continue
   cd,ci=tree.query(oldxyz[core],workers=1);rd,ri=tree.query(oldxyz[region],workers=1);ci=np.unique(ci[cd<=.012]);ri=np.unique(ri[rd<=.012]);ci=ci[valid[ci]];ri=ri[valid[ri]]
   if len(ci)<5 or not len(ri):row['status']='geometry_correspondence_insufficient';continue
   take=(pairids==iid)&(pairvotes>=4);pts=pairpts[take];votes=pairvotes[take]
   take=(votes>=.25*ev['total_frame_votes'][pts])&(votes>=.5*ev['top1_votes'][pts])&valid[pts]
   take&=(current[pts]<=0)|(current[pts]==iid)
   take&=(ev['state'][pts]!=2)|(ev['top1_instance_id'][pts]==iid)
   pts=pts[take]
   if not len(pts):row['status']='no_current_supported_points';continue
   dist,j=cKDTree(xyz[ri]).query(xyz[pts],workers=1);near=ri[j];delta=xyz[pts]-xyz[near]
   ok=(dist<=.03)&(np.sum(norm[pts]*norm[near],axis=1)>=.85)&(np.abs(np.sum(delta*norm[near],axis=1))<=.0075)&(np.abs(np.sum(delta*norm[pts],axis=1))<=.0075)
   pts=pts[ok];ci=np.intersect1d(ci,pts)
   row.update(current_supported=len(pts),remaining_core_points=len(ci))
   if len(ci)<5 or len(pts)<30:row['status']='insufficient_supported_core_or_region';continue
   cells=np.rint(xyz[pts]/.01).astype(np.int32);cells,rep,inv=np.unique(cells,axis=0,return_index=True,return_inverse=True)
   cl=DBSCAN(eps=.05,min_samples=10,algorithm='kd_tree',n_jobs=1).fit_predict(cells.astype(float)*.01)
   corecells=np.unique(inv[np.isin(pts,ci)]);u,cnt=np.unique(cl[corecells],return_counts=True);cu,cn=np.unique(cl,return_counts=True);counts=dict(zip(cu,cn));keep=[int(k) for k,c in zip(u,cnt) if k>=0 and c>=5 and counts[k]>=30]
   add=pts[np.isin(cl[inv],keep)&(labels[pts]<0)]
   conflict=(proposed[add]>0)&(proposed[add]!=iid);ambiguous[add[conflict]]=True;proposed[add[~conflict]]=iid
   row.update(status='PASS',accepted_clusters=len(keep),proposed=len(add),conflicting_proposals=int(conflict.sum()))
 result=labels.copy();take=(proposed>0)&~ambiguous;result[take]=proposed[take]
 assert np.array_equal(result[labels>0],labels[labels>0]);assert np.all(np.isin(result[result>0],pairids))
 g.save(out/'frozen_labels.npz',instance_id=result);g.save(out/'completion_trace.npz',point_index=np.flatnonzero(take),assigned_id=result[take],ambiguous_point_index=np.flatnonzero(ambiguous))
 report=dict(status='FROZEN_BEFORE_EVALUATION',settings=CFG,rows=rows,added_gray=int(take.sum()),ambiguous=int(ambiguous.sum()),active_history={str(k):sorted(v) for k,v in active.items()},seconds=time.time()-start,GT_used=False,production_unchanged=True,base_comparison=str(basecase/'merge_fill/complete.json'),source_hashes=manifest['input_sha256'],code_sha256=g.sha(__file__))
 g.dump(out/'decision_freeze.json',report)
 r=g.evaluate(v,s,'history_seeded',original,result,out,dict(settings=CFG,history_report=str(out/'decision_freeze.json'),base_comparison=str(basecase/'merge_fill/complete.json'),families=g.read(basecase/'complete.json')['families'],GT_used_for_decision=False,final_B_once=True,spatial_assignment_separate_from_native_vote=True,native_vote_rule_unchanged=True,no_online_feedback=True,terminal_only=True))
 r['added_vs_guarded']=int(take.sum());r['base_metrics']=g.read(basecase/'merge_fill/complete.json')['temporary_metrics'];g.dump(out/'complete.json',r)
 for k,p in manifest['inputs'].items():assert g.sha(p)==manifest['input_sha256'][k]
 return r
def one(v,s):
 out=T/v/s
 if (out/'complete.json').exists():return g.read(out/'complete.json')
 oldcase=PREVIOUS/'base'/v/s;oldout=PREVIOUS/v/s
 assert (oldcase/'complete.json').exists() and (oldout/'complete.json').exists()
 proof=T/'reuse_audit'/v/s;proof.mkdir(parents=True,exist_ok=True)
 original=g.load(O/'published'/v/s/'final_B/instance_surface.npz');manifest=g.read(O/'published'/v/s/'scene_complete.json')
 for k,p in manifest['inputs'].items():assert g.sha(p)==manifest['input_sha256'][k]
 pair=g.load(Path(manifest['inputs']['evidence']).with_name('surface_instance_frame_votes.npz'))
 root=O/('main' if v=='main' else 'online')/s/'400'/('A' if v=='main' else 'B')
 lut,newdec=g.build_candidates(original['xyz_m'],original['instance_id'],pair,root/'repair/commit_candidate/source_pixel_support.npz',proof)
 olddec=g.read(oldcase/'merge_decisions.json');same=newdec['families']==olddec['families']
 audit=dict(status='PASS',families_numerically_equal=same,previous_families=olddec['families'],current_families=newdec['families'],source_hashes_verified=True,completion_algorithm_unchanged=True,GT_used=False)
 g.dump(proof/'reuse_decision.json',audit)
 basecase=BASE/v/s;basecase.parent.mkdir(parents=True,exist_ok=True);out.parent.mkdir(parents=True,exist_ok=True)
 if same:
  if not basecase.exists():basecase.symlink_to(oldcase,target_is_directory=True)
  if not out.exists():out.symlink_to(oldout,target_is_directory=True)
  print('REUSE_EXACT',v,s,flush=True);return g.read(out/'complete.json')
 out.mkdir(parents=True,exist_ok=True)
 g.one(v,s)
 result=core_completion(v,s,basecase,out);print('COVISIBLE_PASS',v,s,result['added_vs_guarded'],flush=True);return result
def pool(scenes,tag):
 pools={}
 for v in ['main','online_B']:
  pools[v]={}
  for arm in ['original','guarded','history_seeded']:
   dest=T/'pooled'/tag/v/arm;dest.mkdir(parents=True,exist_ok=True);batch=[]
   for s in scenes:
    p=O/'evaluation'/v/s/'r2' if arm=='original' else BASE/v/s/'merge_fill/r2' if arm=='guarded' else T/v/s/'r2';a=g.read(p/'adapter_manifest.json')
    batch.append(dict(gt=a['gt_file'],prediction=str(p/'canonical_prediction.npz'),diagnostic_prediction=str(p/'diagnostic_support_prediction.npz')))
   g.dump(dest/'batch.json',dict(scenes=batch));g.run(['eval-batch','--scenes',dest/'batch.json','--out',dest],dest/'run.log');pools[v][arm]=g.read(dest/'summary.json')
 gt=pools['main']['original']['CA_PRF1_0_5'];gt=gt['TP']+gt['FN'];expected=350 if len(scenes)==8 else 95;assert gt==expected
 for p in pools.values():
  for z in p.values():assert z['CA_PRF1_0_5']['TP']+z['CA_PRF1_0_5']['FN']==gt
 report=dict(status='PASS',scenes=scenes,GT=gt,pooled=pools,settings=dict(history=CFG,merge=g.CFG),production_results_unchanged=True,development=True,terminal_ablation=True,per_scene=[g.read(T/v/s/'complete.json') for s in scenes for v in ['main','online_B']])
 g.dump(T/(tag+'_comparison.json'),report);return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args();os.nice(10);T.mkdir(parents=True,exist_ok=True);g.protocol()
 scenes=SCENES if args.full else ['office1','room0'];tag='full8' if args.full else 'pilot';g.dump(T/(tag+'_plan.json'),dict(approval='User authorized adapt based on actual diagnosis and validate eight scenes after useful pilot',scenes=scenes,variants=['main','online_B'],settings=CFG,code_sha256=g.sha(__file__)))
 done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pooler:
  fs=[pooler.submit(one,v,s) for s in scenes for v in ['main','online_B']]
  for f in concurrent.futures.as_completed(fs):f.result();done+=1;g.dump(T/(tag+'_progress.json'),dict(status='RUNNING',completed=done,total=len(fs)))
 pool(scenes,tag);g.dump(T/(tag+'_progress.json'),dict(status='PASS',completed=done,total=len(fs)));print(tag.upper()+'_ALL_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  g.dump(T/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

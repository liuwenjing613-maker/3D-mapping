"""Compare full closed-loop trusted policy with the original online-B, using main R2 only."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys
import numpy as np
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
BASE=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
NEW=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def run(args,log):
 with Path(log).open('w') as f:subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT,check=True)
def metrics(s):
 owner=s.get('owner_surface',s.get('diagnostics',{}))
 return {'F1':s['CA_PRF1_0_5']['F1'],'AP50':s['CA_AP50_uniform'],'AP':s['CA_AP_uniform'],'PQ':s['CA_PQ']['PQ'],
 'TP':s['CA_PRF1_0_5']['TP'],'FP':s['CA_PRF1_0_5']['FP'],'FN':s['CA_PRF1_0_5']['FN'],
 'correct_owner_coverage':owner.get('Correct_owner_Coverage'),'wrong_owner_coverage':owner.get('Wrong_owner_Coverage'),
 'mean_best_gt_IoU':s.get('CA_mCov',s.get('diagnostics',{}).get('mean_best_gt_iou'))}
def compare(a,b):return {k:{'before':a[k],'after':b[k],'delta':None if a[k] is None or b[k] is None else b[k]-a[k]} for k in a}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate-root',type=Path,default=NEW);ap.add_argument('--out',type=Path,default=NEW/'comparison');ap.add_argument('--self-check',action='store_true');args=ap.parse_args()
 cand=BASE if args.self_check else args.candidate_root
 out=args.out.resolve()
 if not out.is_relative_to(NEW.resolve()):raise ValueError('Comparison outputs must stay under the new result root')
 out.mkdir(parents=True,exist_ok=True)
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert (lock['evaluation_profile'],lock['profile_revision'],lock['evaluable_GT_count'],lock['frozen'])==('object_observed_repair',2,350,False)
 assert sha(MAIN/'unified_eval/configs'/lock['config_file'])==lock['config_sha256']
 for f,h in lock['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/f)==h,f
 audit=[];batch=[]
 fields=['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256','correspondence_sha256','correspondence_file_sha256','profile_semantics_sha256','profile_revision']
 # Read and verify every scene before launching any scoring.
 for scene in SCENES:
  paths=[root/'evaluation/online_B'/scene for root in [BASE,cand]]
  manifests=[]
  for root,ev in zip([BASE,cand],paths):
   receipt=read(ev/'complete.json');assert receipt['status']=='PASS'
   assert sha(receipt['prediction'])==receipt['prediction_sha256']
   m=read(ev/'r2/adapter_manifest.json');assert m['profile_config_sha256']==lock['config_sha256']
   assert m['evaluator_code_sha256']==lock['evaluator_code_sha256']
   assert m['gt_file_sha256']==lock['canonical_GT'][scene]['file_sha256']
   assert sha(m['gt_file'])==m['gt_file_sha256']
   assert sha(m['correspondence_cache'])==m['correspondence_file_sha256']
   assert sha(ev/'r2/canonical_prediction.npz')==m['canonical_prediction_sha256']
   assert sha(ev/'r2/diagnostic_support_prediction.npz')==m['diagnostic_prediction_sha256']
   assert m['source_surface_sha256']==receipt['prediction_sha256']
   final=read(root/'published/online_B'/scene/'final_B/scene_finalization.json')
   assert final['status']=='PASS' and final['C_enabled'] is False and final['online_state_updated'] is False
   assert final['frame_count']==400
   manifests.append(m)
  old,new=manifests
  for key in fields:assert old[key]==new[key],(scene,key)
  with np.load(paths[0]/'r2/overlap_matrix.npz') as a,np.load(paths[1]/'r2/overlap_matrix.npz') as b:
   np.testing.assert_array_equal(a['gt_ids'],b['gt_ids']);np.testing.assert_array_equal(a['gt_size'],b['gt_size'])
  audit.append({'scene':scene,'same_geometry_scope_correspondence':True,'geometry_xyz_sha256':new['geometry_xyz_sha256'],'gt_file_sha256':new['gt_file_sha256'],'correspondence_sha256':new['correspondence_sha256'],'baseline_prediction':str(paths[0]/'r2/canonical_prediction.npz'),'candidate_prediction':str(paths[1]/'r2/canonical_prediction.npz')})
  batch.append({'gt':new['gt_file'],'prediction':str(paths[1]/'r2/canonical_prediction.npz'),'diagnostic_prediction':str(paths[1]/'r2/diagnostic_support_prediction.npz')})
 dump(out/'input_audit.json',{'status':'PASS','candidate_root':str(cand),'protocol_lock_sha256':sha(MAIN/'unified_eval/configs/current_protocol.json'),'scenes':audit})
 dump(out/'batch.json',{'scenes':batch})
 pooled=out/'pooled';pooled.mkdir(exist_ok=True)
 run(['eval-batch','--scenes',out/'batch.json','--out',pooled],out/'pooled.log')
 oldsum=read(BASE/'pooled/online_B/summary.json');newsum=read(pooled/'summary.json')
 assert newsum['CA_PRF1_0_5']['TP']+newsum['CA_PRF1_0_5']['FN']==350
 assert oldsum['reference_scope']==newsum['reference_scope']
 oldrows={r['scene_id']:r for r in oldsum['per_scene']};newrows={r['scene_id']:r for r in newsum['per_scene']}
 instance_rows=[];scene_rows=[]
 for scene in SCENES:
  pair=out/'pairs'/scene;pair.mkdir(parents=True,exist_ok=True)
  run(['eval-repair-pair','--gt',lock['canonical_GT'][scene]['file'],'--before',BASE/'evaluation/online_B'/scene/'r2/canonical_prediction.npz','--after',cand/'evaluation/online_B'/scene/'r2/canonical_prediction.npz','--out',pair],pair/'run.log')
  d=read(pair/'paired_repair_metrics.json')
  for r in d['per_gt']:
   row={'scene':scene,**r};row['lost_iou50_qualification']=r['before_best_iou']>.5 and r['after_best_iou']<=.5
   row['gained_iou50_qualification']=r['before_best_iou']<=.5 and r['after_best_iou']>.5
   row['drop_at_least_0_10']=r['delta_iou']<=-.1
   row['severe_degradation']=row['lost_iou50_qualification'] or row['drop_at_least_0_10']
   instance_rows.append(row)
  scene_rows.append({'scene':scene,'metrics':compare(metrics(oldrows[scene]),metrics(newrows[scene])),**{k:d[k] for k in ['corrected_instances','degraded_instances','unchanged_instances','correct_owner_vertex_gain']}})
 assert len(instance_rows)==350
 severe=sorted([r for r in instance_rows if r['severe_degradation']],key=lambda r:r['delta_iou'])
 counts={k:sum(r['change']==k for r in instance_rows) for k in ['IMPROVED','DEGRADED','UNCHANGED']}
 counts.update(severe_degradation=len(severe),lost_iou50_qualification=sum(r['lost_iou50_qualification']for r in instance_rows),gained_iou50_qualification=sum(r['gained_iou50_qualification']for r in instance_rows))
 result={'status':'PASS','scope':'full_closed_loop_final_published_maps_400_frames_8_scenes_R2_350_GT','self_check':args.self_check,'baseline_root':str(BASE),'candidate_root':str(cand),'pooled_metrics':compare(metrics(oldsum),metrics(newsum)),'instance_counts':counts,'per_scene':scene_rows,'per_gt':instance_rows,'worst_degradations':sorted([r for r in instance_rows if r['delta_iou']<0],key=lambda r:r['delta_iou'])[:30],'largest_improvements':sorted(instance_rows,key=lambda r:-r['delta_iou'])[:30],'severe_degradations':severe,'interpretation':{'instance_identity':'raw_gt_id within scene; never assumes native IDs are stable between runs','all_instances_audited':350,'repair_region_grouping':'Not classified: no unsupported repaired/untouched causal attribution; all GT including other instances are reported','severe_degradation':'best IoU decreases by at least 0.10, or crosses from >0.5 to <=0.5; latter is a best-IoU qualification change, not an asserted one-to-one TP assignment','protocol':'object_observed_repair revision 2; frozen=false; final400GT never used for online prefixes'}}
 if args.self_check:
  assert all(v['delta'] in (0,None) for v in result['pooled_metrics'].values())
  assert counts['UNCHANGED']==350 and counts['DEGRADED']==counts['IMPROVED']==0
 dump(out/'comparison.json',result)
 with (out/'per_gt.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(instance_rows[0]));w.writeheader();w.writerows(instance_rows)
 with (out/'severe_degradations.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(instance_rows[0]));w.writeheader();w.writerows(severe)
 print(json.dumps({'status':'PASS','out':str(out),'self_check':args.self_check,'pooled_metrics':result['pooled_metrics'],'instance_counts':counts},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

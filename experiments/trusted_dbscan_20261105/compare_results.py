"""Four-corner terminal DBSCAN comparison. All scoring uses the current main R2."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys
import numpy as np
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
BASE=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
TRUST=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
DB=BASE/'ablations/dbscan_history_covisible_20261102'
NEW=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
ARMS=['old_raw','old_dbscan','trusted_raw','trusted_dbscan']
EDGES={'old_postprocess':('old_raw','old_dbscan'),'trusted_postprocess':('trusted_raw','trusted_dbscan'),'policy_raw':('old_raw','trusted_raw'),'policy_dbscan':('old_dbscan','trusted_dbscan')}
FIELDS=['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256','correspondence_sha256','correspondence_file_sha256','profile_semantics_sha256','profile_revision']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(args,log):
 with Path(log).open('w') as f:subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT,check=True)
def metrics(s):
 o=s.get('owner_surface',s.get('diagnostics',{}));p=s['CA_PRF1_0_5']
 return dict(F1=p['F1'],AP50=s['CA_AP50_uniform'],AP=s['CA_AP_uniform'],PQ=s['CA_PQ']['PQ'],TP=p['TP'],FP=p['FP'],FN=p['FN'],correct_owner_coverage=o.get('Correct_owner_Coverage'),wrong_owner_coverage=o.get('Wrong_owner_Coverage'),mean_best_gt_IoU=s.get('CA_mCov',s.get('diagnostics',{}).get('mean_best_gt_iou')))
def delta(a,b):return {k:dict(before=a[k],after=b[k],delta=None if a[k] is None or b[k] is None else b[k]-a[k]) for k in a}
def severe(a,b):return (a>.5 and b<=.5) or b-a<=-.1
def classify(a,b):return 'IMPROVED' if b>a+1e-12 else 'DEGRADED' if b<a-1e-12 else 'UNCHANGED'
def flags(a,b):return dict(lost_iou50_qualification=a>.5 and b<=.5,gained_iou50_qualification=a<=.5 and b>.5,drop_at_least_0_10=b-a<=-.1,severe_degradation=severe(a,b))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate-root',type=Path,default=NEW);ap.add_argument('--out',type=Path,default=NEW/'comparison');args=ap.parse_args()
 out=args.out.resolve();candidate=args.candidate_root.resolve()
 assert out.is_relative_to(NEW.resolve()) and candidate.is_relative_to(NEW.resolve())
 out.mkdir(parents=True,exist_ok=True)
 dump(out/'comparison.json',dict(schema='trusted_dbscan_four_corners_v1',status='RUNNING',self_check=False,stage='validating_all_four_corners'))
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert (lock['evaluation_profile'],lock['profile_revision'],lock['evaluable_GT_count'],lock['frozen'])==('object_observed_repair',2,350,False)
 assert sha(MAIN/'unified_eval/configs'/lock['config_file'])==lock['config_sha256']
 for f,h in lock['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/f)==h
 rawcmp=read(TRUST/'comparison/comparison.json')
 assert rawcmp['status']=='PASS' and rawcmp['self_check'] is False and len(rawcmp['per_gt'])==350
 original_severe={(r['scene'],int(r['raw_gt_id'])) for r in rawcmp['per_gt'] if r['severe_degradation']}
 assert len(original_severe)==7
 control=read(candidate/'control_complete.json');assert control['status']=='PASS' and sorted(control['scenes'])==sorted(SCENES)
 paths={a:{} for a in ARMS};manifests={a:{} for a in ARMS};batches={a:[] for a in ARMS};audit=[]
 for s in SCENES:
  paths['old_raw'][s]=BASE/'evaluation/online_B'/s/'r2'
  paths['old_dbscan'][s]=DB/'online_B'/s/'r2'
  paths['trusted_raw'][s]=TRUST/'evaluation/online_B'/s/'r2'
  paths['trusted_dbscan'][s]=candidate/'candidate'/s/'r2'
  final=read(candidate/'candidate'/s/'final_scene_complete.json')
  assert final['status']=='PASS' and final['mode']=='candidate' and final['control_exact'] is True
  assert final.get('scene',s)==s
  control_path=candidate/'control'/s/'final_scene_complete.json'
  assert Path(final['control_receipt']).resolve()==control_path.resolve()
  ctl=read(control_path)
  assert ctl['status']=='PASS' and ctl['mode']=='control' and ctl['control_exact'] is True
  assert ctl.get('scene',s)==s and sha(candidate/'control'/s/'instance_surface.npz')==ctl['prediction_sha256']
  oldgt=None;oldsize=None;row=dict(scene=s,arms={})
  for a in ARMS:
   p=paths[a][s];m=read(p/'adapter_manifest.json');manifests[a][s]=m
   assert m['profile_config_sha256']==lock['config_sha256'] and m['evaluator_code_sha256']==lock['evaluator_code_sha256']
   assert m['evaluation_profile']=='object_observed_repair' and m['profile_revision']==2
   gtlock=lock['canonical_GT'][s]
   assert m['gt_file_sha256']==gtlock['file_sha256']
   for k in ['gt_scope_sha256','gt_observed_support_sha256']:assert m[k]==gtlock[k]
   for f,k in [(m['gt_file'],'gt_file_sha256'),(m['correspondence_cache'],'correspondence_file_sha256'),(p/'canonical_prediction.npz','canonical_prediction_sha256'),(p/'diagnostic_support_prediction.npz','diagnostic_prediction_sha256'),(m['source_surface'],'source_surface_sha256')]:
    assert sha(f)==m[k],(a,s,k)
   if a in ['old_raw','trusted_raw']:
    root=BASE if a=='old_raw' else TRUST;r=read(root/'evaluation/online_B'/s/'complete.json')
    assert r['status']=='PASS' and r['prediction_sha256']==m['source_surface_sha256']
    z=read(root/'published/online_B'/s/'final_B/scene_finalization.json')
    assert z['status']=='PASS' and z['frame_count']==400 and z['C_enabled'] is False and z['online_state_updated'] is False
   elif a=='old_dbscan':
    r=read(DB/'online_B'/s/'complete.json');assert r['status']=='PASS' and r['prediction_sha256']==m['source_surface_sha256']
    assert read(DB/'reuse_audit/online_B'/s/'reuse_decision.json')['status']=='PASS'
   else:
    assert Path(m['source_surface']).resolve()==(candidate/'candidate'/s/'instance_surface.npz').resolve()
    assert final['prediction_sha256']==m['source_surface_sha256']
   with np.load(p/'overlap_matrix.npz') as z:
    if oldgt is None:oldgt=z['gt_ids'].copy();oldsize=z['gt_size'].copy()
    np.testing.assert_array_equal(z['gt_ids'],oldgt);np.testing.assert_array_equal(z['gt_size'],oldsize)
    assert len(oldgt)==gtlock['evaluable_GT_count']
   for k in FIELDS:assert m[k]==manifests['old_raw'][s][k],(a,s,k)
   row['arms'][a]=dict(r2=str(p),source_surface=m['source_surface'],prediction_sha256=m['source_surface_sha256'],canonical_prediction_sha256=m['canonical_prediction_sha256'])
   batches[a].append(dict(gt=m['gt_file'],prediction=str(p/'canonical_prediction.npz'),diagnostic_prediction=str(p/'diagnostic_support_prediction.npz')))
  audit.append(row)
 dump(out/'input_audit.json',dict(status='PASS',scope='8_scenes_350_GT_same_geometry_GT_correspondence_main_R2',protocol_lock_sha256=sha(MAIN/'unified_eval/configs/current_protocol.json'),scenes=audit))
 pooled={};scene_metrics={}
 for a in ARMS:
  p=out/'pooled'/a;p.mkdir(parents=True,exist_ok=True);dump(p/'batch.json',dict(scenes=batches[a]))
  run(['eval-batch','--scenes',p/'batch.json','--out',p],p/'run.log')
  z=read(p/'summary.json');assert z['CA_PRF1_0_5']['TP']+z['CA_PRF1_0_5']['FN']==350
  if a!='old_raw':assert z['reference_scope']==read(out/'pooled/old_raw/summary.json')['reference_scope']
  pooled[a]=metrics(z);scene_metrics[a]={r['scene_id']:metrics(r) for r in z['per_scene']}
 # Exact existing summaries are controls for the three existing corners.
 existing={'old_raw':BASE/'pooled/online_B/summary.json','old_dbscan':DB/'pooled/full8/online_B/history_seeded/summary.json','trusted_raw':TRUST/'comparison/pooled/summary.json'}
 for a,p in existing.items():
  expected=metrics(read(p))
  for k,v in pooled[a].items():assert v==expected[k],(a,k,v,expected[k])
 comparisons={};corners={}
 for edge,(a,b) in EDGES.items():
  rows=[];sr=[]
  for s in SCENES:
   p=out/'pairs'/edge/s;p.mkdir(parents=True,exist_ok=True)
   run(['eval-repair-pair','--gt',lock['canonical_GT'][s]['file'],'--before',paths[a][s]/'canonical_prediction.npz','--after',paths[b][s]/'canonical_prediction.npz','--out',p],p/'run.log')
   pair=read(p/'paired_repair_metrics.json')
   for r in pair['per_gt']:
    row=dict(scene=s,**r,**flags(r['before_best_iou'],r['after_best_iou']));rows.append(row)
    c=corners.setdefault((s,int(r['raw_gt_id'])),dict(scene=s,raw_gt_id=int(r['raw_gt_id'])))
    for arm,val in [(a,r['before_best_iou']),(b,r['after_best_iou'])]:
     if arm in c:assert c[arm]==val,(s,arm)
     c[arm]=val
   sr.append(dict(scene=s,metrics=delta(scene_metrics[a][s],scene_metrics[b][s]),**{k:pair[k] for k in ['corrected_instances','degraded_instances','unchanged_instances','correct_owner_vertex_gain']}))
  assert len(rows)==350 and len({(r['scene'],r['raw_gt_id']) for r in rows})==350
  sev=sorted([r for r in rows if r['severe_degradation']],key=lambda r:r['delta_iou'])
  counts={k:sum(r['change']==k for r in rows) for k in ['IMPROVED','DEGRADED','UNCHANGED']}
  counts.update(severe_degradation=len(sev),lost_iou50_qualification=sum(r['lost_iou50_qualification'] for r in rows),gained_iou50_qualification=sum(r['gained_iou50_qualification'] for r in rows))
  comparisons[edge]=dict(before_arm=a,after_arm=b,pooled_metrics=delta(pooled[a],pooled[b]),instance_counts=counts,per_scene=sr,per_gt=rows,severe_degradations=sev)
  csvwrite(out/(edge+'_per_gt.csv'),rows)
  csvwrite(out/(edge+'_severe.csv'),sev)
 assert len(corners)==350
 prior={(r['scene'],int(r['raw_gt_id'])):r for r in rawcmp['per_gt']}
 labels=read(TRUST/'comparison_gt_labels.json');assert labels['status']=='PASS' and labels['count']==350
 for key,c in corners.items():
  name=labels['labels'][c['scene']][str(c['raw_gt_id'])]
  c['semantic_class']=name['semantic_class'];c['semantic_class_zh']=name['semantic_class_zh']
  assert c['old_raw']==prior[key]['before_best_iou'] and c['trusted_raw']==prior[key]['after_best_iou'],key
  c['old_postprocess_delta']=c['old_dbscan']-c['old_raw'];c['trusted_postprocess_delta']=c['trusted_dbscan']-c['trusted_raw']
  c['policy_raw_delta']=c['trusted_raw']-c['old_raw'];c['policy_dbscan_delta']=c['trusted_dbscan']-c['old_dbscan']
  c['postprocess_interaction_delta']=c['policy_dbscan_delta']-c['policy_raw_delta']
  c['original_severe']=key in original_severe
  c['severe_vs_old_dbscan']=severe(c['old_dbscan'],c['trusted_dbscan'])
  c['severe_vs_old_raw']=severe(c['old_raw'],c['trusted_dbscan'])
  c['restored_to_old_raw']=c['trusted_dbscan']>=c['old_raw']-1e-12
  c['restored_to_old_dbscan']=c['trusted_dbscan']>=c['old_dbscan']-1e-12
  c['dbscan_effect']=classify(c['trusted_raw'],c['trusted_dbscan'])
  c['recovery_status']='RESTORED' if c['restored_to_old_raw'] and c['restored_to_old_dbscan'] else 'PARTIALLY_IMPROVED' if c['dbscan_effect']=='IMPROVED' else 'WORSENED' if c['dbscan_effect']=='DEGRADED' else 'UNCHANGED'
 rows=sorted(corners.values(),key=lambda r:(SCENES.index(r['scene']),r['raw_gt_id']))
 focus=sorted([r for r in rows if r['original_severe']],key=lambda r:r['policy_raw_delta'])
 newly=[r for r in rows if r['severe_vs_old_dbscan'] and not r['original_severe']]
 result=dict(schema='trusted_dbscan_four_corners_v1',status='PASS',scope='terminal_same_postprocessing_8_scenes_400_frames_R2_350_GT',self_check=False,scene_count=8,gt_count=350,arms={a:dict(label={'old_raw':'原在线 final B','old_dbscan':'PPT 原在线 + DBSCAN','trusted_raw':'可信区域在线 final B','trusted_dbscan':'可信区域在线 + 同一 DBSCAN'}[a],pooled_metrics=pooled[a]) for a in ARMS},comparisons=comparisons,per_gt=rows,original_severe_cases=focus,new_severe_vs_old_dbscan=newly,per_scene=[dict(scene=s,arms={a:scene_metrics[a][s] for a in ARMS}) for s in SCENES],interpretation=dict(severe='best IoU drop >=0.10 or >0.5 to <=0.5; qualification change is not an asserted one-to-one TP assignment',restored='trusted DBSCAN reaches BOTH old raw and old DBSCAN IoU; individual restored flags also supplied',partial='trusted DBSCAN improves its raw IoU but has not reached both old baselines',interaction='(trusted DBSCAN-old DBSCAN)-(trusted raw-old raw), descriptive difference of differences, not independent causal ablation',native_IDs='Only scene and raw_gt_id paired; native prediction IDs never assumed stable',scope='All four are fixed final400 geometry. Terminal postprocessing only, no rerun of online feedback; main object_observed_repair revision2 frozen=false'))
 dump(out/'comparison.json',result);csvwrite(out/'per_gt.csv',rows);csvwrite(out/'original_severe_cases.csv',focus);csvwrite(out/'new_severe_vs_old_dbscan.csv',newly)
 print(json.dumps(dict(status='PASS',output=str(out),pooled={a:pooled[a] for a in ARMS},counts={e:comparisons[e]['instance_counts'] for e in EDGES},original_severe_recovery={x:sum(r['recovery_status']==x for r in focus) for x in ['RESTORED','PARTIALLY_IMPROVED','WORSENED','UNCHANGED']}),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

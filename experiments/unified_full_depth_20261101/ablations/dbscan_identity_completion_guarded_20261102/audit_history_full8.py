from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101');T=O/'ablations/dbscan_history_covisible_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
d=read(T/'full8_comparison.json');assert d['status']=='PASS' and len(d['scenes'])==8 and d['GT']==350
keys=['geometry_xyz_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','correspondence_sha256','profile_config_sha256','evaluator_code_sha256']
rows=[];counts={};geometries={}
for v in ['main','online_B']:
 counts[v]=dict(original_gray=0,adjusted_gray=0,added_by_history=0)
 for s in d['scenes']:
  p=T/v/s;receipt=read(p/'complete.json');old=read(O/'evaluation'/v/s/'r2/adapter_manifest.json');new=read(p/'r2/adapter_manifest.json');proof=read(T/'reuse_audit'/v/s/'reuse_decision.json')
  for k in keys:assert old[k]==new[k],(v,s,k)
  assert new['evaluation_profile']=='object_observed_repair' and new['profile_revision']==2 and new['profile_config']['frozen'] is False
  assert sha(p/'instance_surface.npz')==receipt['prediction_sha256']==new['source_surface_sha256']
  with np.load(p/'instance_surface.npz') as a,np.load(T/'base'/v/s/'merge_fill/instance_surface.npz') as b,np.load(O/'published'/v/s/'final_B/instance_surface.npz') as c:
   np.testing.assert_array_equal(a['xyz_m'],c['xyz_m']);np.testing.assert_array_equal(a['rgb'],c['rgb']);before=b['instance_id'];after=a['instance_id'];np.testing.assert_array_equal(after[before>0],before[before>0]);assert int((after<0).sum())==receipt['temporary_gray']
  oldm=receipt['original_metrics'];newm=receipt['temporary_metrics'];gt=oldm['CA_PRF1_0_5']['TP']+oldm['CA_PRF1_0_5']['FN'];assert gt==newm['CA_PRF1_0_5']['TP']+newm['CA_PRF1_0_5']['FN']
  geo={k:new[k] for k in keys if k not in ['evaluator_code_sha256']}
  if s in geometries:assert geo==geometries[s]
  else:geometries[s]=geo
  row=dict(variant=v,scene=s,status='PASS',GT=gt,old_F1=oldm['CA_PRF1_0_5']['F1'],new_F1=newm['CA_PRF1_0_5']['F1'],F1_delta_pp=(newm['CA_PRF1_0_5']['F1']-oldm['CA_PRF1_0_5']['F1'])*100,PQ_delta_pp=(newm['CA_PQ']['PQ']-oldm['CA_PQ']['PQ'])*100,AP50_delta_pp=(newm['CA_AP50_uniform']-oldm['CA_AP50_uniform'])*100,AP_delta_pp=(newm['CA_AP_uniform']-oldm['CA_AP_uniform'])*100,original_gray=receipt['original_gray'],adjusted_gray=receipt['temporary_gray'],history_added=receipt['added_vs_guarded'],numerically_identical_prior_result_reused=proof['families_numerically_equal'],merge_families=proof['current_families'],prediction_sha256=receipt['prediction_sha256'])
  rows.append(row);counts[v]['original_gray']+=row['original_gray'];counts[v]['adjusted_gray']+=row['adjusted_gray'];counts[v]['added_by_history']+=row['history_added'];print('AUDIT_PASS',v,s,flush=True)
report=dict(status='PASS',cases=16,scene_count=8,GT=350,protocol='main object_observed_repair revision2',frozen=False,official_results_unchanged=True,terminal_postprocessing_only=True,GT_not_used_for_assignment=True,native_vote_rule='top1>=2 and confidence>=0.67',same_geometry_RGB_GT_correspondence_and_evaluator=True,history_completion_preserves_assigned_points=True,common_covisibility_guard_all_scenes=True,reuse_proofs='reuse_audit/<variant>/<scene>/reuse_decision.json',development_limitation='Parameters developed after room0/office1 diagnosis; full8 feedback tightened a generic co-visibility guard. This is not an untouched holdout benchmark.',gray_counts=counts,per_scene=rows)
(T/'final_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
compact=dict(pooled={v:{a:{'F1':z['CA_PRF1_0_5']['F1']*100,'PQ':z['CA_PQ']['PQ']*100,'AP50':z['CA_AP50_uniform']*100,'AP':z['CA_AP_uniform']*100,'TP':z['CA_PRF1_0_5']['TP'],'FP':z['CA_PRF1_0_5']['FP'],'FN':z['CA_PRF1_0_5']['FN']} for a,z in arms.items()} for v,arms in d['pooled'].items()},gray_counts=counts,per_scene=rows)
(T/'concise_results.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2));print('FINAL_AUDIT_PASS',flush=True)

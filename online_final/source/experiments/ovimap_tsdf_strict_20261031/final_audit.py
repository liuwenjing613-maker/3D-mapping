"""Final cross-check of completed primary/auxiliary results and frozen provenance."""
from pathlib import Path
import hashlib,json,math,platform,sys
import numpy as np
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
M=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert read(O/'driver_status.json')['stage']=='COMPLETE','R2 rollout incomplete'
 assert read(O/'official_rule_status.json')['stage']=='COMPLETE','Primary evaluation incomplete'
 validations=['native_smoke_validation_20.json','causal_resume_validation.json','prefix_5_vs_20_validation.json',
 'prefix_input_isolation_validation.json','frozen_reference_all8_validation.json','bounded_vote_validation.json',
 'publication_threshold_validation.json','official_rule_adapter_validation.json',
 'native_export_validator_v2_regression.json','native_final_export_validation_v2.json']
 for name in validations:assert read(O/name)['status']=='PASS',name
 for n,h in read(O/'implementation_bounded_freeze.json')['files'].items():assert sha(C/n)==h,n
 for n,h in read(O/'external_dependency_freeze.json').items():assert sha(n)==h,n
 for n,h in read(O/'official_rule_implementation_freeze.json')['files'].items():assert sha(n)==h,n
 lock=read(M/'unified_eval/configs/current_protocol.json')
 config=M/'unified_eval/configs'/lock['config_file']
 assert sha(config)==lock['config_sha256']=='6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0'
 for n,h in lock['evaluator_code_sha256'].items():assert sha(M/'unified_eval'/n)==h,n
 scope=read(O/'evaluation_scope_comparison.json')
 assert scope['all_reference_geometry_equal'] and scope['official_GT_total']==542 and scope['R2_GT_total']==350
 primary=read(O/'official_rule_comparison.json');aux=read(O/'comparison.json')
 assert primary['scene_count']==8 and primary['GT_count_total']==542
 scene_reports={};prefix_count=0;replay_count=0
 for s in SCENES:
  v=read(O/'verification'/f'{s}.json');p=read(O/'official_rule_evaluation'/s/'complete.json')
  assert v['status']==p['status']=='PASS'
  assert v['frames']==p['frames']==400
  for key in ['same_geometry_all_prefixes','readonly_all_prefixes','strict_votes_all_prefixes','no_repair','no_hole_fill']:assert v[key],(s,key)
  native=O/'native'/f'{s}_400_trace';ours=O/'ours'/f'{s}_400_main'
  assert sha(native/'prefix_audit.jsonl')==v['native_audit_sha256']
  assert sha(ours/'audit.jsonl')==v['input_audit_sha256']
  a=[json.loads(x) for x in (native/'prefix_audit.jsonl').read_text().splitlines()]
  b=[json.loads(x) for x in (ours/'audit.jsonl').read_text().splitlines()]
  assert len(a)==len(b)==400
  for i,(x,y) in enumerate(zip(a,b)):
   assert x['frame_id']==y['frame_id']==i*5
   assert x['geometry_sha256']==y['geometry_sha256'] and x['readonly_verified'] and y['strict_vote_verified']
  checkpoints=[n for n in [1,2,5,10,20,50,100,200,300,400] if b[n-1]['full_replay_verified']]
  assert len(checkpoints)==10
  prefix_count+=len(a);replay_count+=len(checkpoints)
  for method,source in [('ours',ours/'maps/f001995.npz'),('ovimap',native/'ovi/f001995.npz')]:assert sha(source)==p['prediction_sha256'][method]
  assert v['prediction_sha256']==p['prediction_sha256']
  pair=[]
  for method in ['ours','ovimap']:
   a=read(O/'evaluation'/s/method/'adapter_manifest.json');pair.append(a)
   assert a['source_surface_sha256']==p['prediction_sha256'][method]
   assert a['gt_file_sha256']==lock['canonical_GT'][s]['file_sha256']
   assert a['profile_config_sha256']==lock['config_sha256']
  assert pair[0]['correspondence_file_sha256']==pair[1]['correspondence_file_sha256']
  scene_reports[s]={'same_prediction_files_primary_and_auxiliary':True,'all_400_prefix_geometry_equal':True}
 for method in ['ours','ovimap']:
  assert aux[method]['TP']+aux[method]['FN']==350
  expected=2*aux[method]['TP']/(2*aux[method]['TP']+aux[method]['FP']+aux[method]['FN'])
  assert math.isclose(aux[method]['F1'],expected,abs_tol=1e-15)
  for key,value in primary['comparison'][method].items():
   mean=math.fsum(read(O/'official_rule_evaluation'/s/'complete.json')['comparison'][method][key] for s in SCENES)/8
   assert math.isclose(value,mean,abs_tol=1e-15)
 report={'status':'PASS','scenes':scene_reports,'shared_geometry_prefixes':prefix_count,'full_replay_checkpoints':replay_count,
  'primary':'OVI-MAP evaluation-rule adapter, 5 cm, 542 full-GT instances, 400 frames/scene, eight-scene macro mean',
  'auxiliary':'unchanged main R2, 1 cm, fixed 350 observed GT, pooled metrics',
  'no_repair':True,'no_hole_fill':True,'core_frozen_sources_and_dependencies_unchanged':True,
  'native_palette_validation_scope':'v2 certifies decodable unambiguous points only; missing log entries and PLY rounding exclusions remain explicit. V1 false failures are archived.',
  'validation_reports':validations,'primary_result_sha256':sha(O/'official_rule_comparison.json'),
  'auxiliary_result_sha256':sha(O/'comparison.json'),'audit_source_sha256':sha(__file__),'python':sys.version,'numpy':np.__version__}
 (O/'final_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'audit':'PASS','primary':primary,'auxiliary':aux}),flush=True)
if __name__=='__main__':main()

"""Reuse only the already validated partial replay of the identical room2 E1 source."""
from pathlib import Path
import hashlib,json,time
from bootstrap import R,SOURCE,SOURCE_CODE,SCENE,COUNT,verify_sources
assert SCENE=='room2' and COUNT==35
OLD=Path('/data/chenkejun/CVPR/results/e1_partial_autov2_20261019/room2')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if not (OLD/'run_complete.json').exists():print('NO_REUSABLE_RESULT');return
 assert read(OLD/'run_complete.json')['status']=='PASS'
 policy=read(OLD/'policy.json');assert policy['source']==str(SOURCE) and policy['selected_masks']==COUNT
 assert policy['only_algorithm_change']=='old_observation_plans values whole -> partial'
 freeze=read(OLD/'predictions_freeze.json')
 assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT' and not freeze['GT_read_by_mapper']
 for group in ['code_sha256','output_sha256','source_sha256']:
  for p,h in freeze[group].items():assert sha(Path(p))==h,p
 assert freeze['checks']['only_whole_plan_values_changed']
 assert freeze['checks']['original_E1_all_stages_candidates_votes_evidence_and_scopes_exact']
 assert read(OLD/'evaluation_r2/comparison.json')['baseline_and_old_E1_scores_reproduced_exactly']
 audit=read(OLD/'input_selection_audit.json')
 assert sha(SOURCE/'confirmed_selection.json')==audit['original_E1_selection_sha256']
 for name in ['whole_to_partial','predictions_freeze.json','policy.json','tests_complete.json','mapping_complete.json','input_selection_audit.json','frame_receipts.json']:
  p=R/name;assert not p.exists();p.symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())
 result=dict(status='PASS',source=str(OLD),original_E1_source=str(SOURCE),selection_sha256=sha(SOURCE/'confirmed_selection.json'),source_freeze_sha256=sha(OLD/'predictions_freeze.json'),all_recorded_source_code_and_prediction_hashes_verified=True,identical_35_masks_tracking_gate_regions_and_whole_to_partial_rule=True,time=time.time())
 (R/'reuse_receipt.json').write_text(json.dumps(result,indent=2));verify_sources();print('ROOM2_PARTIAL_REUSED')
if __name__=='__main__':main()

import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/ppt5_staged_postid_full_depth_20261110')
import write_fence
"""Apply exactly the office1-validated source-only rule after semantic classification."""
from common_continue import *
import copy
def main():
 p=CANDIDATE_ROOT/'semantic_route/semantic.json';original=read(p);sem=copy.deepcopy(original);disabled=[]
 for cid,row in sem['candidates'].items():
  if row['hit'] and row['target_family']=='cabinet':
   disabled.append(cid);row['hit_before_window_only']=True;row['hit']=False
   row['decision_reason']='CABINET_RETURNS_TO_ORDINARY_SELECTION'
 for cid,row in original['candidates'].items():
  if cid not in disabled:assert row==sem['candidates'][cid]
 sem['stats']['hit_cabinet']=0;sem['stats']['not_hit']+=len(disabled)
 sem['config']['source_route_ablation']=dict(direct_SAM_families=['window'],cabinet='ordinary CropFormer/SAM selection',classification_scores_unchanged=True)
 dump(p.with_name('semantic_before_window_only.json'),original);dump(p,sem)
 dump(CANDIDATE_ROOT/'window_source_policy.json',dict(status='PASS',scene=SCENE,checkpoint=CHECKPOINT,group=GROUP,
  disabled_cabinet_hits=disabled,all_other_semantic_decisions_identical=True,PR_exemption_unchanged=True,formula_unchanged=True))
if __name__=='__main__':main()

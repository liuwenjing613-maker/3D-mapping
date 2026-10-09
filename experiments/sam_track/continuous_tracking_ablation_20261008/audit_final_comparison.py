"""Posthoc report with a fixed original GT cohort and paired ledger/map equivalence checks."""
from pathlib import Path
import sys,json
import numpy as np
import run_ablation as core


def audit(scene):
    engine=core.load_engine(scene);root=core.OUT/scene
    summary=core.read(root/'evaluation_summary.json');assert summary['status']=='PASS'
    source=core.read(engine.ROOT/'evaluation_summary.json');assert summary['fixed_original_target_GT_ids']==source['target_GT_ids']
    sys.path.insert(0,str(core.LEGACY));import evaluate_pilot_v3 as e
    per_old=e.per_gt(e.load_overlap(engine.ROOT/'validated/full_track/v3/overlap.npz'))
    per_base=e.per_gt(e.load_overlap(e.ROOT/'v3/baseline'/scene/'overlap.npz'))
    target=set(source['target_GT_ids'])
    conditions={}
    for name in ['gap0','gap1']:
        per=e.per_gt(e.load_overlap(root/name/'v3/overlap.npz'))
        assert set(per)==set(per_old)==set(per_base)
        orig_to_now=lambda g:{'GT_id':g,'before':per_old[g],'after':per[g]}
        conditions[name]={
          'target_mean_best_IoU':float(np.mean([per[g]['best_IoU'] for g in target])),
          'target_mean_completeness':float(np.mean([per[g]['completeness_best_single_instance'] for g in target])),
          'target_count_matched':sum(per[g]['matched_IoU_gt_0_5'] for g in target),
          'non_target_IoU_degraded_over_0_01':[orig_to_now(g) for g in sorted(per) if g not in target and per[g]['best_IoU']<per_old[g]['best_IoU']-.01],
          'target_IoU_improved_over_0_01':[orig_to_now(g) for g in sorted(target) if per[g]['best_IoU']>per_old[g]['best_IoU']+.01],
          'target_IoU_degraded_over_0_01':[orig_to_now(g) for g in sorted(target) if per[g]['best_IoU']<per_old[g]['best_IoU']-.01],
          'target_match_gained':[g for g in sorted(target) if per[g]['matched_IoU_gt_0_5'] and not per_old[g]['matched_IoU_gt_0_5']],
          'target_match_lost':[g for g in sorted(target) if per_old[g]['matched_IoU_gt_0_5'] and not per[g]['matched_IoU_gt_0_5']],
          'per_GT_comparison':[orig_to_now(g) for g in sorted(per)]}
    def means(per):return {'target_mean_best_IoU':float(np.mean([per[g]['best_IoU'] for g in target])),
       'target_mean_completeness':float(np.mean([per[g]['completeness_best_single_instance'] for g in target])),
       'target_count_matched':sum(per[g]['matched_IoU_gt_0_5'] for g in target)}
    reports={name:core.read(root/name/'full_track/complete.json') for name in ['gap0','gap1']}
    a=reports['gap0']['output_sha256'];b=reports['gap1']['output_sha256'];equal={}
    for path,hash_a in a.items():
        rel=Path(path).relative_to(root/'gap0/full_track');other=root/'gap1/full_track'/rel
        if str(other) not in b:continue
        if hash_a==b[str(other)]:equal[str(rel)]={'file_sha256_equal':True}
        elif path.endswith('.npz'):
            with np.load(path) as x,np.load(other) as y:
                keys=set(x.files)|set(y.files)
                eq={k:k in x.files and k in y.files and bool(np.array_equal(x[k],y[k])) for k in sorted(keys)}
            equal[str(rel)]={'file_sha256_equal':False,'array_equal':eq,'all_arrays_equal':all(eq.values())}
        else:equal[str(rel)]={'file_sha256_equal':False,'content_note':'Inspect metadata differences separately'}
    core.dump(root/'comparison_audit.json',{'status':'PASS','scene':scene,'baseline':means(per_base),'unrestricted':means(per_old),
      'conditions':conditions,'frozen_target_GT_ids':sorted(target),'target_cohort_not_redefined':True,
      'gap0_gap1_output_comparison':equal,'GT_used_for_filter_or_repair':False,
      'posthoc_only':True,'audit_code_sha256':engine.r.sha(Path(__file__))})
    print(scene,json.dumps({'baseline':means(per_base),'unrestricted':means(per_old),
      'conditions':{name:{k:v for k,v in row.items() if k!='per_GT_comparison'} for name,row in conditions.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    for scene in ['room0','room2']:audit(scene)

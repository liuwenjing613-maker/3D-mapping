"""Posthoc diagnosis of every >1pp loss; predictions and R2 scores stay fixed."""
from pathlib import Path
import gc,json,sys
import numpy as np
import evaluate_history_core_r2 as ev
from unified_eval.metrics import owner_coverage

def native_alignment(pred,score):
    uid_to_id={str(x.instance_uid):int(x.metadata['native_instance_id']) for x in pred.instances}
    return {uid_to_id[str(row['instance_uid'])]:int(row['raw_gt_id']) for row in score['scoring_owner_alignment']}
def candidate_rows(pred,overlap,score,raw):
    j=int(np.flatnonzero(overlap.gt_ids==raw)[0]);by_uid={str(x.instance_uid):int(x.metadata['native_instance_id']) for x in pred.instances}
    match=native_alignment(pred,score);rows=[]
    for i,uid in enumerate(overlap.pred_uids):
        recall=float(overlap.recall[i,j]);iou=float(overlap.iou[i,j])
        if recall<=0:continue
        pid=by_uid[str(uid)]
        rows.append({'native_instance_id':pid,'IoU':iou,'recall':recall,
            'scored_owner_GT':match.get(pid),'GT_support_qualifies':raw in overlap.gt_ids.tolist()})
    return sorted(rows,key=lambda r:r['recall'],reverse=True)[:12]
def main():
    completed=ev.read(ev.OUT/'evaluation_complete.json');assert completed['status']=='PASS'
    comp=ev.read(ev.OUT/'comparison.json');assert ev.sha256_file(ev.OUT/'comparison.json')==completed['comparison_sha256']
    freeze=ev.read(ev.ROOT/'predictions_freeze.json');parent=ev.read(ev.STRICT/'predictions_freeze.json')
    assert ev.require_current_protocol(ev.CURRENT_CONFIG)
    protocol=ev.Protocol.from_dict(ev.read(ev.CURRENT_CONFIG));results={}
    for scene in ('room0','room2'):
        gtpath=ev.CURRENT_PROTOCOL['canonical_GT'][scene]['file'];assert ev.require_current_gt(gtpath)==scene
        gt=ev.load_gt(gtpath);oldfile=parent['scenes'][scene]['predictions']['strict_full_track_final']['path']
        with np.load(oldfile) as z:xyz=z['xyz_m'].copy();old=z['instance_id'].copy()
        with np.load(ev.ROOT/scene/'final/instance_surface.npz') as z:new=z['instance_id'].copy();np.testing.assert_array_equal(xyz,z['xyz_m'])
        cache=ev.STRICT/'evaluation_r2/geometry_cache'/scene/'fixed_gt_to_tsdf_r2.npz'
        c=ev.load_fixed_surface_correspondence(cache,xyz,gt.xyz_ref)
        before=ev.load_prediction(ev.STRICT/'evaluation_r2/evaluation'/scene/'strict_full_track_final/canonical_prediction.npz')
        after=ev.load_prediction(ev.OUT/'evaluation'/scene/'history_core_final/canonical_prediction.npz')
        a=ev.quality(gt,before,protocol,old,c);b=ev.quality(gt,after,protocol,new,c)
        sa=owner_coverage(a['overlap']);sb=owner_coverage(b['overlap']);ma=native_alignment(before,sa);mb=native_alignment(after,sb)
        affected=[r for r in comp['paired'][scene]['history_core_final_vs_strict_full_track']['per_gt'] if r['delta_iou']<-.01 or r['delta_best_recall']<-.01]
        details=[]
        for r in affected:
            raw=r['raw_gt_id'];mask=a['target']&(gt.instance_id==raw)
            transition,n=np.unique(np.column_stack((a['identity'][mask],b['identity'][mask])),axis=0,return_counts=True)
            changes=[{'before_native_ID':int(p[0]),'after_native_ID':int(p[1]),'reference_vertices':int(nn)}
                for p,nn in zip(transition,n) if p[0]!=p[1]]
            details.append({'GT_result':r,'before_top_recall_predictions':candidate_rows(before,a['overlap'],sa,raw),
                'after_top_recall_predictions':candidate_rows(after,b['overlap'],sb,raw),
                'changed_raw_identity_transitions':sorted(changes,key=lambda r:r['reference_vertices'],reverse=True),
                'unchanged_native_identity_vertices':int(np.sum(mask&(a['identity']==b['identity']))),
                'matching_quality_transition':ev.matrix(a['quality'],b['quality'],mask).tolist()})
        results[scene]={'all_more_than_1pp_losses_included':True,'losses':details,
            'native_ID_owner_alignment_changes':{str(pid):[ma.get(pid),mb.get(pid)] for pid in sorted(set(ma)|set(mb)) if ma.get(pid)!=mb.get(pid)},
            'clarifies_method_specific_UID_changes':'Native persistent IDs are matched across conditions; different method prefixes in instance_uid are not physical identity changes.'}
        del gt,c,a,b,before,after;gc.collect()
    for p,d in completed['output_sha256'].items():assert ev.sha256_file(p)==d,p
    for p,d in freeze['output_sha256'].items():assert ev.sha256_file(p)==d,p
    for p,d in comp['protected_sha256'].items():assert ev.sha256_file(p)==d,p
    output=ev.OUT/'analysis/remaining_case_diagnosis.json'
    ev.dump(output,{'status':'PASS','profile_revision':2,'GT_used_to_change_predictions':False,
        'original_metrics_and_predictions_unchanged':True,'source_comparison_sha256':completed['comparison_sha256'],'scenes':results})
    print(json.dumps({'status':'PASS','diagnosis':results}),flush=True)

if __name__=='__main__':main()

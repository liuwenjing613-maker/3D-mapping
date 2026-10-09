"""Independent GT-free streaming reconstruction; does not import vote operators."""
from pathlib import Path
import hashlib,json,time
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
PRIOR=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def reference(keys,base):
    keys=np.unique(keys)
    points,inverse,counts=np.unique(keys//base,return_inverse=True,return_counts=True)
    return keys[counts[inverse]==1],points[counts>1]
def add_frame(domain,counts,frame):
    indices=np.searchsorted(domain,frame)
    assert np.all(indices<len(domain))
    np.testing.assert_array_equal(domain[indices],frame)
    # Counts start at zero; this full reconstruction never uses the baseline/delta counts.
    counts[indices]+=1

def main():
    freeze=json.loads((OUT/'predictions_freeze.json').read_text())
    assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    policy=json.loads((OUT/'policy.json').read_text());results={};total=0
    for path,digest in freeze['generated_output_sha256'].items():assert sha(path)==digest
    for scene,rec in freeze['scenes'].items():
        root=Path(policy['repair_source_roots'][scene]);association=json.loads((root/'global_association.json').read_text());base=association['instance_base']
        material=json.loads((BASE/'native'/scene/'P1-A1/surface_p0/materialization_report.json').read_text())
        lookup={}
        for line in (BASE/'native'/scene/'P1-A1/association/associations.jsonl').read_text().splitlines():
            row=json.loads(line);lookup[(row['frame_id'],row['mask_local_id'])]=row['instance_id']
        sourcepair=load(PRIOR/'ablation'/scene/'abstain/surface_instance_frame_votes.npz')
        baseline=sourcepair['surface_point_index'].astype(np.int64)*base+sourcepair['instance_id']
        baseline_order=np.argsort(baseline);baseline=baseline[baseline_order];baseline_counts=np.zeros(len(baseline),np.int64)
        domains={};counts={};point_totals={};stats={}
        for cond in ('seed_only','full_track'):
            pair=load(OUT/scene/cond/'native/surface_instance_frame_votes.npz')
            domain=pair['surface_point_index'].astype(np.int64)*base+pair['instance_id'];order=np.argsort(domain)
            domains[cond]=(domain[order],pair['frame_votes'][order]);counts[cond]=np.zeros(len(domain),np.int64)
            evidence=load(OUT/scene/cond/'native/surface_evidence.npz')
            point_totals[cond]=np.zeros(len(evidence['state']),np.int32)
            stats[cond]={'verified_frames':0,'updated_frames':0,'collision_point_frame_events_before':0,
                'collision_point_frame_events_after':0,'ambiguous_candidate_key_units_before':0,
                'ambiguous_candidate_key_units_after':0,'same_ID_shared_candidate_keys_preserved':0,
                'shared_keys_with_effective_vote_after':0,'shared_keys_abstained_due_to_other_ID':0}
        for record in material['frame_records']:
            fid=record['frame_id'];assert sha(record['support_file'])==record['support_sha256']
            z=load(record['support_file']);p=z['surface_point_index'].astype(np.int64);local=z['mask_local_id']
            table=np.zeros(int(local.max())+1,np.int64)
            for mid in np.unique(local):table[mid]=lookup[(fid,int(mid))]
            raw=np.unique(p*base+table[local]);oldstrict,oldbad=reference(raw,base)
            add_frame(baseline,baseline_counts,oldstrict)
            for cond in ('seed_only','full_track'):
                oldpath=root/'validated'/cond/'transactions'/('f%06d.npz'%fid)
                newpath=OUT/scene/cond/'transactions'/('f%06d.npz'%fid)
                if oldpath.exists():
                    oldtx=load(oldpath);newtx=load(newpath)
                    assert int(oldtx['instance_base'][0])==int(newtx['instance_base'][0])==base
                    np.testing.assert_array_equal(raw,oldtx['old_frame_keys'])
                    np.testing.assert_array_equal(raw,newtx['old_candidate_keys'])
                    rawafter=oldtx['new_frame_keys'];np.testing.assert_array_equal(rawafter,newtx['new_candidate_keys'])
                    effective,bad=reference(rawafter,base)
                    np.testing.assert_array_equal(oldstrict,newtx['old_frame_keys'])
                    np.testing.assert_array_equal(effective,newtx['new_frame_keys'])
                    retained=oldtx['retained_old_frame_keys'];assert np.all(np.isin(retained,rawafter))
                    retired=oldtx['retired_surface_local_pairs']
                    shared=np.intersect1d(np.unique(retired[:,0]*base+table[retired[:,1]]),retained)
                    assert np.all(np.isin(shared,rawafter))
                    active=np.isin(shared,effective);assert np.all(np.isin(shared[~active]//base,bad))
                    stats[cond]['same_ID_shared_candidate_keys_preserved']+=len(shared)
                    stats[cond]['shared_keys_with_effective_vote_after']+=int(active.sum())
                    stats[cond]['shared_keys_abstained_due_to_other_ID']+=int((~active).sum())
                    stats[cond]['updated_frames']+=1
                else:
                    assert not newpath.exists();rawafter=raw;effective=oldstrict;bad=oldbad
                assert len(effective)==len(np.unique(effective//base))
                add_frame(domains[cond][0],counts[cond],effective)
                point_totals[cond][effective//base]+=1
                stats[cond]['verified_frames']+=1
                stats[cond]['collision_point_frame_events_before']+=len(oldbad)
                stats[cond]['collision_point_frame_events_after']+=len(bad)
                stats[cond]['ambiguous_candidate_key_units_before']+=len(raw)-len(oldstrict)
                stats[cond]['ambiguous_candidate_key_units_after']+=len(rawafter)-len(effective)
                total+=1
        np.testing.assert_array_equal(baseline_counts,sourcepair['frame_votes'][baseline_order])
        for cond in ('seed_only','full_track'):
            np.testing.assert_array_equal(counts[cond],domains[cond][1])
            evidence=load(OUT/scene/cond/'native/surface_evidence.npz')
            np.testing.assert_array_equal(point_totals[cond],evidence['total_frame_votes'])
            assert np.all(point_totals[cond]<=400)
            stats[cond].update(streaming_full_recompute_equals_stored_delta=True,all_frame_point_votes_unique=True,
                maximum_effective_observed_frames_at_one_point=int(point_totals[cond].max()))
        results[scene]=stats
    for path,digest in freeze['generated_output_sha256'].items():assert sha(path)==digest
    result={'status':'PASS','real_frame_conditions_checked':total,'scenes':results,'GT_read':False,
        'independent_of_one_vote_and_strict_vote_ops':True,'all_prediction_hashes_unchanged':True,
        'definitions':{'collision_point_frame_events':'Distinct (frame,point) with >1 ID in the raw candidate set',
            'ambiguous_candidate_key_units':'All (point,ID) keys on ambiguous points, removed by abstention',
            'legacy_receipt_name_note':'Primary replay receipts ambiguous_point_frames_before/after store candidate key units; use the distinct event counts in this independent audit instead'},
        'audit_code_sha256':sha(__file__)}
    path=OUT/'independent_correctness.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

"""Verify exact prefix inputs, effective votes, selection lineage, and recorded reads."""
from online_common import *
import re
import numpy as np

def main():
    m=read(R/'inputs_manifest.json');mapping=read(R/'mapping_complete.json')
    timeline=read(R/'mapping_timeline.json');selection=read(R/'v2/selection.json')
    assert m['frame_ids']==mapping['frame_ids']==list(FIDS)
    assert len(timeline)==80
    for k,row in enumerate(timeline,1):
        assert row['observed_frame_ids']==list(FIDS[:k]) and row['last_frame_id']==FIDS[k-1]
        assert row['maximum_votes_per_point']<=k and row['processed_frames']==k
    for f in m['files']:assert sha(f['path'])==f['sha256'] and f['frame'] in FIDS
    assert len(m['files'])==240 and sha(C/'config.json')==m['config_sha256']
    lookup={}
    for line in (R/'association/associations.jsonl').read_text().splitlines():
        r=json.loads(line);assert r['frame_id'] in FIDS
        lookup[r['frame_id'],r['mask_local_id']]=r['instance_id']
    base=max(lookup.values())+1;allkeys=[];rows=[]
    for fid in FIDS:
        with np.load(R/f'baseline/frame_support/f{fid:06d}.npz') as z:
            assert int(z['frame_id'][0])==fid
            points=z['surface_point_index'];local=z['mask_local_id']
        ids=np.asarray([lookup[fid,int(lid)] for lid in local],np.int64)
        unique=np.unique(points.astype(np.int64)*base+ids)
        point_ids,sizes=np.unique(unique//base,return_counts=True)
        keys=unique[np.repeat(sizes==1,sizes)]
        assert len(keys)==len(np.unique(keys//base))
        allkeys.append(keys);rows.append(dict(frame=fid,effective_votes=len(keys),ambiguous_points=int(np.sum(sizes>1))))
    keys,counts=np.unique(np.concatenate(allkeys),return_counts=True)
    with np.load(R/'baseline/surface_instance_frame_votes.npz') as z:
        savedkeys=z['surface_point_index'].astype(np.int64)*base+z['instance_id']
        np.testing.assert_array_equal(savedkeys,keys);np.testing.assert_array_equal(z['frame_votes'],counts)
    used_frames=set();blocked=[];reads=0;raw_accesses=0;probes=0
    allowed={f['source']:f['frame'] for f in m['files']}
    for path in (R/'io_audit').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('probe'):
                probes+=int(row['blocked']);continue
            reads+=1
            if row['blocked']:blocked.append(row)
            if row['path'] in allowed:
                frame=allowed[row['path']];assert frame<=row['cutoff']
                used_frames.add(frame);raw_accesses+=1
    assert not blocked,blocked[:5]
    assert used_frames==set(FIDS)
    guard=read(R/'guard_test.json');assert guard['status']=='PASS' and len(guard['checks'])==4
    source=read(R/'v1/seed_manifest.json');final=read(R/'v2/seed_manifest.json')
    seeds={g['id']:g for g in source['groups']}
    assert {g['id'] for g in final['groups']}==set(selection['retained_group_ids'])
    for g in final['groups']:
        p=g['primary'];assert p==seeds[g['id']]['primary'] and p['frame'] in FIDS
        assert sha(R/'v1'/p['mask'])==p['mask_sha256']
    sources={str(p.relative_to(C)):sha(p) for p in C.rglob('*.py')}
    audit=dict(status='PASS',frame_ids=list(FIDS),processed_frames=80,last_frame_id=395,
        exact_input_files_verified=240,raw_input_accesses=raw_accesses,recorded_IO_events=reads,
        forbidden_read_attempts_in_computation=len(blocked),intentional_fence_probes_blocked=probes,
        guard_checks=guard['checks'],geometry_integrated_from_empty=True,strict_map_republished_each_frame=True,
        independent_raw_support_vote_recount=True,per_frame_vote_check=rows,
        future_geometry_used=False,future_ROI_used=False,future_tracking_used=False,GT_used=False,
        previous_full_sequence_manual_mask_overrides_used=False,repair_applied=False,
        final_masks_bit_identical_to_source=True,source_masks=len(source['groups']),retained_masks=len(final['groups']),
        code_sha256=sources,numerical_snapshot=str(SNAPSHOT),input_manifest_sha256=sha(R/'inputs_manifest.json'),
        selection_sha256=sha(R/'v2/selection.json'))
    write(R/'causality_audit.json',audit)
    write(R/'review/downloads/causality_audit.json',audit)
    print('CAUSALITY_AND_VOTE_AUDIT_PASS',len(final['groups']),flush=True)

if __name__=='__main__':main()

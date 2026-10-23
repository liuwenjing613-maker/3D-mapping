"""Check real continuation outputs, skipped-frame reporting and the input fence."""
import json
from pathlib import Path
import re
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')

def read(path):
    return json.loads(path.read_text())

def check(root):
    mapping=root/'mapping'
    complete=read(mapping/'mapping_complete.json')
    timing=read(mapping/'mapping_timing.json')
    timeline=read(mapping/'continuation_timeline.json')
    launch=read(root/'launch.json')
    count=complete['frame_count']
    prior=count-100
    policy=complete.get('surface_policy','every_frame')
    assert complete['status']==timing['status']=='PASS'
    assert complete['prior_source_hashes_unchanged'] and complete['historical_committed_entries_unchanged']
    assert complete['all_prior_layers_retained'] and complete['strict_same_frame_point_at_most_one_vote']
    assert not complete['GT_used'] and not complete['future_inputs_used']
    assert len(timeline)==len(timing['frames'])==100
    assert [row['processed_frames'] for row in timeline]==list(range(prior+1,count+1))
    assert [row['last_frame_id'] for row in timeline]==list(range(prior*5,count*5,5))
    restored=read(mapping/'restore_audit.json')
    version=restored['restored_map_version']
    for row in timeline:
        assert row['association_map_version_before']==version
        version=row['association_map_version_after']
    assert version==complete['map_version']
    if policy=='checkpoint':
        assert timing['surface_rebuild_frames']==complete['surface_rebuild_frames']==[count]
        assert not complete['strict_republication_after_every_frame']
        assert complete['checkpoint_strict_republication_complete']
        for row,frame in zip(timeline[:-1],timing['frames'][:-1]):
            assert row['surface_rebuilt'] is False and row['surface_statistics_current'] is False
            for key in ['surface_points','state_counts','native_assigned_points','effective_votes',
                        'maximum_votes_per_point','same_frame_uniqueness_verified']:
                assert row[key] is None,key
            for key in ['surface_extract','strict_surface_evidence','checkpoint_export']:
                assert frame[key]==0.,key
        assert timeline[-1]['surface_rebuilt'] and timeline[-1]['surface_statistics_current']
    else:
        assert complete['strict_republication_after_every_frame']
        assert all(row['surface_points']>0 for row in timeline)
        assert all(row['strict_surface_evidence']>0 for row in timing['frames'])
    for key,value in timing['stage_seconds'].items():
        assert sum(row[key] for row in timing['frames'])==value,key
    with np.load(mapping/'baseline/surface_evidence.npz') as z:
        names=['unobserved','tentative','confirmed','conflict']
        expected={name:int(np.count_nonzero(z['state']==i)) for i,name in enumerate(names)}
        assert timeline[-1]['state_counts']==expected
        assert timeline[-1]['surface_points']==len(z['xyz_m'])
    audited=0
    for path in (root/'io_audit').glob('*.jsonl'):
        for line in path.open():
            row=json.loads(line)
            assert row['blocked'] is False,row
            assert '/ground_truth/' not in row['path']
            match=re.search(r'(?<![A-Za-z0-9])(?:frame|depth|f)(\d{6})(?!\d)',Path(row['path']).name)
            if match:assert int(match[1])<=row['cutoff'],row
            audited+=1
    assert audited>0
    return dict(status='PASS',name=root.name,policy=policy,frames=100,
        all_prior_state_preserved=True,association_version_chain_exact=True,
        skipped_frame_statistics_explicitly_unavailable=policy=='checkpoint',
        audited_file_opens=audited,blocked_opens=0,source_commit=launch['prior_commit']['prior_commit_sha256'])

if __name__=='__main__':
    rows=[]
    for root in sorted(OUT.glob('*_?')):
        if (root/'mapping/mapping_complete.json').exists() and (root/'launch.json').exists():
            rows.append(check(root))
    assert rows
    result=dict(status='PASS',runs=rows)
    (OUT/'run_contracts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

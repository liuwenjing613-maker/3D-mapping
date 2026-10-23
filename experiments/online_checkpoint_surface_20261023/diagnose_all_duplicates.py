"""Explain independent-run differences; never override strict comparison failures."""
from pathlib import Path
import json
import numpy as np
from compare_outputs import load,order_rows

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
names=['fresh_comparison_200_A','original_repeat_comparison_200_A','archive_comparison_300_A',
       'archive_comparison_200_B','archive_comparison_300_B']
results=[]
for name in names:
    report=json.loads((OUT/(name+'.json')).read_text())
    paths=[Path(report[k]) for k in ['left','right']]
    data=[load(p/'baseline/surface_evidence.npz') for p in paths]
    groups=[np.unique(z['xyz_m'],axis=0,return_inverse=True,return_counts=True) for z in data]
    assert np.array_equal(groups[0][0],groups[1][0])
    assert np.array_equal(groups[0][2],groups[1][2])
    arrays=[]
    for p,z,g in zip(paths,data,groups):
        columns=[g[1][:,None]]+[z[k][:,None] for k in sorted(z) if k not in ['xyz_m','rgb']]
        columns.extend([load(p/'baseline/native/instance_surface.npz')['instance_id'][:,None],
                        load(p/'baseline/final/instance_surface.npz')['instance_id'][:,None]])
        t=np.column_stack(columns)
        arrays.append(t[order_rows(t)])
    same_spatial_records=np.array_equal(*arrays)
    # Geometry-only alignment on coordinates; unique coordinates cannot be permuted.
    orders=[order_rows(z['xyz_m']) for z in data]
    unique_mask=groups[0][2][groups[0][1][orders[0]]]==1
    nonduplicate_differences={k:int(np.count_nonzero(data[0][k][orders[0]][unique_mask]!=data[1][k][orders[1]][unique_mask]))
        for k in data[0] if k not in ['xyz_m','rgb']}
    votes=[]
    for p,g in zip(paths,groups):
        z=load(p/'baseline/surface_instance_frame_votes.npz')
        t=np.column_stack((g[1][z['surface_point_index']],z['instance_id'],z['frame_votes']))
        votes.append(t[order_rows(t)])
    row=dict(comparison=name,duplicate_vertices=int(np.sum(groups[0][2]-1)),
        nonduplicate_evidence_differences=nonduplicate_differences,
        per_coordinate_full_evidence_native_final_multiset_exact=bool(same_spatial_records),
        coordinate_instance_vote_records_exact=bool(np.array_equal(*votes)),
        strict_comparison_status_unchanged=report['status'])
    results.append(row)
    print(json.dumps(row),flush=True)
(OUT/'duplicate_scope_audit.json').write_text(json.dumps(dict(results=results,
    note='Diagnostic only. Keeps every record, including duplicates. Does not change acceptance status or online point ordering.'),indent=2)+'\n')

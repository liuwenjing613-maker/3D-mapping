"""Diagnose unchanged Gate's index-based 2048-point sampling; never alter its samples."""
from pathlib import Path
from collections import Counter
import argparse,json
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def records(a):return Counter(row.tobytes() for row in np.ascontiguousarray(a))
def cases(root):
    xyz=load(root/'mapping/baseline/surface_evidence.npz')['xyz_m']
    result={}
    for case in read(root/'tracking/seed_diagnostics.json')['cases']:
        z=load(root/'tracking/cases'/case['case_uid']/'seed_projected_support.npz')
        for obj in case['objects']:
            tid=obj['track_id'];points=np.unique(z['surface_point_index'][z['track_id']==tid])
            samples=points[np.linspace(0,len(points)-1,min(len(points),2048),dtype=int)]
            result[tid]=dict(case=case['case_uid'],support=records(xyz[points]),anchors=records(xyz[samples]))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--left',default='reference');p.add_argument('--right',default='checkpoint');p.add_argument('--name',default='anchor_order_diagnosis');args=p.parse_args()
    roots=[OUT/args.left,OUT/args.right];a,b=map(cases,roots);rows=[]
    for tid in sorted(a.keys()&b.keys()):
        x,y=a[tid],b[tid]
        rows.append(dict(track_id=tid,source_cases=[x['case'],y['case']],
            full_seed_support_coordinate_multiset_exact=x['support']==y['support'],
            anchor_coordinate_multiset_exact=x['anchors']==y['anchors'],
            anchor_count=[sum(x['anchors'].values()),sum(y['anchors'].values())],
            common_anchor_coordinates=sum((x['anchors']&y['anchors']).values())))
    result=dict(left=str(roots[0]),right=str(roots[1]),same_track_ids=sorted(a)==sorted(b),objects=rows,
        unchanged_source_sampling='points[np.linspace(0,len(points)-1,min(len(points),2048),dtype=int)]',
        note='Diagnostic only; neither online vertex order nor Gate sample choices were modified.')
    (OUT/(args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(objects=len(rows),changed_support=sum(not r['full_seed_support_coordinate_multiset_exact'] for r in rows),
                         changed_anchor_sets=sum(not r['anchor_coordinate_multiset_exact'] for r in rows)),indent=2))
if __name__=='__main__':main()

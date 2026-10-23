"""Compare physical ROI membership/ranking without treating temporary point IDs as identity."""
from pathlib import Path
import hashlib,json
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def order(x):return np.lexsort(tuple(x[:,i] for i in range(x.shape[1]-1,-1,-1)))
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def signature(root):
    cat=root/'candidates/catalog/room0'
    m=read(cat/'candidate_manifest.json');z=load(cat/'membership.npz')
    xyz=load(root/'mapping/baseline/surface_evidence.npz')['xyz_m']
    rows=[]
    for e in m['entries']:
        k=e['component_index']
        core=xyz[z['core_point_index'][z['core_offsets'][k]:z['core_offsets'][k+1]]]
        context=xyz[z['context_point_index'][z['context_offsets'][k]:z['context_offsets'][k+1]]]
        rows.append(dict(alias=e['id'],group=e['group'],rank=e['rank'],count=e['count'],counts=e['counts'],
            core_points=len(core),core_xyz_multiset_sha256=digest(core[order(core)]),
            context_points=len(context),context_xyz_multiset_sha256=digest(context[order(context)]),
            views=[{k:v[k] for k in ['view','frame','RGB_sha256','crop_xyxy','visible_counts','visible_target_points']} for v in e['views']]))
    return rows

def main():
    a,b=signature(OUT/'reference'),signature(OUT/'checkpoint')
    assert [r['alias'] for r in a]==[r['alias'] for r in b]
    differences=[dict(alias=x['alias'],fields=[k for k in x if x[k]!=y[k]],left=x,right=y) for x,y in zip(a,b) if x!=y]
    membership=lambda rows:sorted((r['group'],r['core_xyz_multiset_sha256'],r['count']) for r in rows)
    result=dict(status='PASS' if not differences else 'DIFFERENT',entries=len(a),
        ranking_and_physical_membership_exact=not differences,
        unordered_state_candidate_sets_exact=membership(a)==membership(b),differences=differences,
        ignored_metadata='Only temporary vertex/component IDs, their index hashes, generated timestamps and path prefixes; coordinate floats have zero tolerance',
        left=a,right=b)
    (OUT/'roi_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['left','right','differences']},indent=2))
    for d in differences:print(d['alias'],d['fields'])
if __name__=='__main__':main()

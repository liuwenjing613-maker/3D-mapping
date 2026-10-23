"""Exact comparison: align TSDF block keys and surface coordinates, never use tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.sparse import load_npz

def read(path):
    return json.loads(path.read_text())

def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def order_rows(array):
    return np.lexsort(tuple(array[:, i] for i in range(array.shape[1]-1, -1, -1)))

class Comparison:
    def __init__(self):
        self.checks = []

    def array(self, name, left, right):
        same_shape = left.shape == right.shape
        same_dtype = left.dtype == right.dtype
        equal = same_shape and same_dtype and np.array_equal(left, right)
        row = dict(name=name, exact=bool(equal), left_shape=list(left.shape), right_shape=list(right.shape),
                   left_dtype=str(left.dtype), right_dtype=str(right.dtype))
        if same_shape and not equal:
            different = left != right
            row['different_elements'] = int(np.count_nonzero(different))
            if left.ndim:
                row['different_rows'] = int(np.count_nonzero(different.reshape(len(left), -1).any(axis=1)))
            if left.dtype.kind in 'fiu' and right.dtype.kind in 'fiu':
                row['max_abs_difference'] = float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)), initial=0))
        self.checks.append(row)
        print(name, 'PASS' if equal else 'DIFFERENT', row.get('different_rows', ''), flush=True)
        return equal

    def value(self, name, left, right):
        equal = left == right
        self.checks.append(dict(name=name, exact=bool(equal)))
        print(name, 'PASS' if equal else 'DIFFERENT', flush=True)
        return equal

def compare(left, right):
    c = Comparison()
    assert read(left/'mapping_complete.json')['status']=='PASS'
    assert read(right/'mapping_complete.json')['status']=='PASS'
    for rel in ['prior_commit.json', 'association/associations.jsonl', 'association/observations.jsonl']:
        c.value(rel, (left/rel).read_bytes(), (right/rel).read_bytes())
    for rel in ['association/observation_support_3cm.npz', 'association/identity_counts.npz']:
        a, b = load(left/rel), load(right/rel)
        c.value(rel+'/keys', sorted(a), sorted(b))
        for key in sorted(a.keys() & b.keys()):
            c.array(rel+'/'+key, a[key], b[key])
    a, b = load(left/'geometry/tsdf_grid.npz'), load(right/'geometry/tsdf_grid.npz')
    ia, ib = order_rows(a['key']), order_rows(b['key'])
    c.value('tsdf_grid/keys', sorted(a), sorted(b))
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        if key == 'key' or key.startswith('value_'):
            x, y = x[ia], y[ib]
        c.array('tsdf_grid/'+key, x, y)
    del a, b
    clouds = [o3d.io.read_point_cloud(str(p/'geometry/surface.ply')) for p in (left,right)]
    attributes = [np.column_stack((np.asarray(x.points), np.asarray(x.normals), np.asarray(x.colors))) for x in clouds]
    # Open3D can emit identical vertices in a different order. Match full evidence
    # records ONLY after all geometry fields, giving one global point permutation.
    # Every vote, per-frame support and graph edge is then checked under that same
    # permutation; sorting each output independently could conceal a discrepancy.
    orders=[]
    for path,geometry in zip((left,right),attributes):
        evidence=load(path/'baseline/surface_evidence.npz')
        native=load(path/'baseline/native/instance_surface.npz')['instance_id']
        final=load(path/'baseline/final/instance_surface.npz')['instance_id']
        records=[geometry]+[evidence[k][:,None] for k in sorted(evidence) if k not in ('xyz_m','rgb')]
        records.extend([native[:,None],final[:,None]])
        orders.append(order_rows(np.column_stack(records)))
    ia, ib = orders
    c.value('surface/raw_order_exact', bool(np.array_equal(attributes[0], attributes[1])), True)
    c.array('surface/xyz_normals_rgb_aligned', attributes[0][ia], attributes[1][ib])
    inverse = []
    for idx in orders:
        inv = np.empty(len(idx), np.int64)
        inv[idx] = np.arange(len(idx))
        inverse.append(inv)
    duplicate_counts = [len(x)-len(np.unique(x[:,:3], axis=0)) for x in attributes]
    for rel in ['baseline/surface_evidence.npz','baseline/native/instance_surface.npz','baseline/final/instance_surface.npz']:
        a, b = load(left/rel), load(right/rel)
        c.value(rel+'/keys', sorted(a), sorted(b))
        for key in sorted(a.keys() & b.keys()):
            x, y = a[key], b[key]
            if key != 'native_instance_ids':
                x, y = x[ia], y[ib]
            c.array(rel+'/'+key, x, y)
    for rel in ['baseline/surface_instance_frame_votes.npz']:
        tables=[]
        for p, inv in zip((left,right), inverse):
            z=load(p/rel)
            table=np.column_stack((inv[z['surface_point_index']],z['instance_id'],z['frame_votes']))
            tables.append(table[order_rows(table)])
        c.array(rel, *tables)
    support_left=sorted((left/'baseline/frame_support').glob('*.npz'))
    support_right=sorted((right/'baseline/frame_support').glob('*.npz'))
    c.value('frame_support/files',[p.name for p in support_left],[p.name for p in support_right])
    frame_support_failed=[]
    for pa,pb in zip(support_left,support_right):
        a,b=load(pa),load(pb)
        tables=[]
        for z, inv in zip((a,b),inverse):
            t=np.column_stack((inv[z['surface_point_index']],z['mask_local_id'],z['candidate_persistent_id']))
            tables.append(t[order_rows(t)])
        extra=['frame_id','source_mask_sha256','repaired_source_pixel_journal']
        if not all(np.array_equal(a[k],b[k]) for k in extra) or not np.array_equal(*tables):
            frame_support_failed.append(pa.name)
    c.value('frame_support/all_projected_candidates_exact',frame_support_failed,[])
    a,b=[load_npz(p/'geometry/review_surface_graph_n02_p08.npz').tocsr()[idx][:,idx] for p,idx in zip((left,right),orders)]
    for graph in (a,b):
        graph.sort_indices()
    for key in ['data','indices','indptr']:
        c.array('review_graph/'+key,getattr(a,key),getattr(b,key))
    for rel in ['baseline/frame_vote_audit.json']:
        c.value(rel,read(left/rel),read(right/rel))
    timing = {name:read(path/'mapping_timing.json') for name,path in [('left',left),('right',right)]}
    # Raw Open3D enumeration order is diagnostic, not a spatial equality requirement.
    required=[x for x in c.checks if x['name']!='surface/raw_order_exact']
    return dict(status='PASS' if all(x['exact'] for x in required) else 'DIFFERENT',
        left=str(left),right=str(right),exact_without_float_tolerance=True,
        alignment='one global permutation by xyz, normals, rgb; identical-geometry ties by full evidence/native/final record; all votes, supports and graph edges checked with the same permutation; TSDF by block key',
        duplicate_coordinate_points=duplicate_counts,checks=c.checks,
        different_frame_support_files=frame_support_failed,
        timing={name:{k:v[k] for k in ['total_seconds','restore_seconds','main_mapping_seconds',
            'main_loop_including_checkpoint_seconds','checkpoint_export_seconds','stage_seconds']} for name,v in timing.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('left',type=Path)
    p.add_argument('right',type=Path)
    p.add_argument('report',type=Path)
    args=p.parse_args()
    result=compare(args.left,args.right)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)
    raise SystemExit(0 if result['status']=='PASS' else 1)

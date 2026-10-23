"""Verification only: replay unchanged voting on one identical saved surface.

No vertex sorting, merging, tolerances, mask changes or GT input. This separates
point enumeration nondeterminism from scheduling equivalence. Not an online run.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import open3d as o3d
from PIL import Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
CODE=REPO/'experiments/online_room0_repair50_to100_20261022/continuation300'
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
ARCHIVE=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/A/mapping')

def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def read(path):
    return json.loads(path.read_text())

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def replay(name,code_path,source):
    os.environ.update(CVPR_CHECKPOINT='200',CVPR_TRACK_GROUP='A',CVPR_MAPPING_RUN_ROOT=str(OUT/name))
    sys.path.insert(0,str(CODE))
    spec=importlib.util.spec_from_file_location(name,str(code_path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root=OUT/name
    assert not root.exists(),str(root)
    for rel in ['baseline/frame_support','baseline/native','baseline/final','geometry','ply','inputs']:
        (root/rel).mkdir(parents=True,exist_ok=True)
    (root/'inputs/masks').symlink_to(source/'inputs/masks',target_is_directory=True)
    (root/'geometry/surface.ply').symlink_to(ARCHIVE/'geometry/surface.ply')
    prior=read(source/'prior_commit.json')
    journal=read(Path(prior['journal_manifest']))
    assert sha(Path(prior['journal_manifest']))==prior['journal_manifest_sha256']
    module.JOURNAL={}
    for row in journal['frames']:
        layers=[]
        for item in row['layer_files']:
            assert sha(Path(item['path']))==item['sha256']
            with Image.open(item['path']) as im:layers.append(np.asarray(im,np.uint16).copy())
        module.JOURNAL[row['frame_id']]=tuple(layers)
    with np.load(prior['checkpoint']) as z:module.REPAIR_MAX_ID=int(z['next_instance_id'][0])-1
    decisions=[json.loads(line) for line in (source/'association/associations.jsonl').read_text().splitlines()]
    lookup={(row['frame_id'],row['mask_local_id']):row['instance_id'] for row in decisions}
    fs=module.ReplicaFrameSource(source/'config.json')
    frames=[fs.load_frame(fid) for fid in module.FIDS]
    assert len(frames)==200
    cloud=o3d.io.read_point_cloud(str(ARCHIVE/'geometry/surface.ply'))
    # PLY color encoding can quantize colors; strict_surface's exact RGB is stored in NPZ.
    fixed=load(ARCHIVE/'baseline/surface_evidence.npz')
    cloud.points=o3d.utility.Vector3dVector(fixed['xyz_m'].astype(np.float64))
    cloud.colors=o3d.utility.Vector3dVector(fixed['rgb'].astype(np.float64))
    module.R=root
    start=time.perf_counter()
    summary=module.strict_surface(frames,lookup,cloud,200,True)
    elapsed=time.perf_counter()-start
    write(root/'replay_provenance.json',dict(status='PASS',verification_only=True,
        source_mapping=str(source),code_path=str(code_path),code_sha256=sha(code_path),
        fixed_geometry_source=str(ARCHIVE),fixed_evidence_sha256=sha(ARCHIVE/'baseline/surface_evidence.npz'),
        fixed_ply_sha256=sha(ARCHIVE/'geometry/surface.ply'),strict_surface_seconds=elapsed,summary=summary))
    return root

def compare_arrays(left,right):
    rows=[]
    assert {p.relative_to(left) for p in (left/'baseline').rglob('*.npz')} == {
        p.relative_to(right) for p in (right/'baseline').rglob('*.npz')}
    for path in sorted((left/'baseline').rglob('*.npz')):
        rel=path.relative_to(left)
        a,b=load(path),load(right/rel)
        assert sorted(a)==sorted(b),str(rel)
        for key in a:
            x,y=a[key],b[key]
            exact=x.dtype==y.dtype and x.shape==y.shape and np.array_equal(x,y)
            rows.append(dict(path=str(rel),key=key,exact=bool(exact)))
    for rel in ['baseline/frame_vote_audit.json','geometry/review_surface_graph_n02_p08.npz']:
        if rel.endswith('.json'):
            rows.append(dict(path=rel,exact=read(left/rel)==read(right/rel)))
        else:
            a,b=load(left/rel),load(right/rel)
            assert sorted(a)==sorted(b),rel
            for key in a:rows.append(dict(path=rel,key=key,
                exact=bool(a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key]))))
    return dict(status='PASS' if all(r['exact'] for r in rows) else 'DIFFERENT',
        left=str(left),right=str(right),comparisons=len(rows),different=[r for r in rows if not r['exact']],
        raw_array_order=True,zero_float_tolerance=True)

if __name__=='__main__':
    ref=OUT/'reference_200_A/mapping'
    new=OUT/'checkpoint_200_A/mapping'
    for p in (ref,new):assert read(p/'mapping_complete.json')['status']=='PASS'
    for rel in ['association/associations.jsonl','association/observations.jsonl','prior_commit.json']:
        assert (ref/rel).read_bytes()==(new/rel).read_bytes(),rel
    old_replay=replay('common_surface_reference_200_A',HERE/'reference/map_continue.py',ref)
    new_replay=replay('common_surface_checkpoint_200_A',CODE/'map_continue.py',new)
    result=dict(old_new=compare_arrays(old_replay,new_replay),
        original_archive=compare_arrays(ARCHIVE,old_replay),
        scope='fixed identical full surface, all historical input and original strict_surface/assignment; not end-to-end repair')
    write(OUT/'common_surface_comparison_200_A.json',result)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['old_new']['status']==result['original_archive']['status']=='PASS' else 1)

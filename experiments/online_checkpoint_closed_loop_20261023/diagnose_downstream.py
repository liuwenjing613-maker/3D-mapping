"""Quantify spatial, historical-pixel and identity-vote differences by physical keys."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import open3d as o3d
from PIL import Image

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
ROOTS=[OUT/m for m in ['reference','checkpoint']]

def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def order(a):return np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))

def main():
    aligned=[]
    for root in ROOTS:
        cloud=o3d.io.read_point_cloud(str(root/'mapping/geometry/surface.ply'))
        geometry=np.column_stack((np.asarray(cloud.points),np.asarray(cloud.normals),np.asarray(cloud.colors)))
        z=load(root/'mapping/baseline/surface_evidence.npz')
        native=load(root/'mapping/baseline/native/instance_surface.npz')['instance_id']
        final=load(root/'mapping/baseline/final/instance_surface.npz')['instance_id']
        records=[geometry]+[z[k][:,None] for k in sorted(z) if k not in ['xyz_m','rgb']]+[native[:,None],final[:,None]]
        idx=order(np.column_stack(records))
        labels=load(root/'repair/partial/whole_to_partial/room0/final/instance_surface.npz')['instance_id']
        aligned.append((geometry[idx],labels[idx]))
    assert np.array_equal(aligned[0][0],aligned[1][0])
    delta=aligned[0][1]!=aligned[1][1]
    changed=[dict(xyz_m=aligned[0][0][i,:3].tolist(),reference_id=int(aligned[0][1][i]),
        checkpoint_id=int(aligned[1][1][i])) for i in np.flatnonzero(delta)]
    finals=dict(points=len(delta),changed_points=int(delta.sum()),changed_percent=float(100*delta.mean()),
        changed_point_records=changed,positive_instance_counts=[int(len(np.unique(a[a>0]))) for _,a in aligned])
    journals=[read(r/'repair/journal/manifest.json')['frames'] for r in ROOTS]
    frames=[]
    for a,b in zip(*journals):
        assert a['frame_id']==b['frame_id'] and len(a['layer_files'])==len(b['layer_files'])
        union=None;layer_entries=0
        for x,y in zip(a['layer_files'],b['layer_files']):
            if x['sha256']==y['sha256']:continue
            left,right=[np.asarray(Image.open(Path(t['path']))) for t in [x,y]]
            d=left!=right;layer_entries+=int(d.sum());union=d if union is None else union|d
        if layer_entries:frames.append(dict(raw_frame=a['frame_id'],different_layer_pixel_entries=layer_entries,
            unique_different_pixel_positions=int(union.sum())))
    pixels=dict(different_frames=len(frames),different_layer_pixel_entries=sum(r['different_layer_pixel_entries'] for r in frames),
        unique_frame_pixel_positions=sum(r['unique_different_pixel_positions'] for r in frames),frames=frames)
    counts=[]
    for root in ROOTS:
        c=read(root/'repair/commit_complete.json');z=load(Path(c['identity_counts']))
        voxel=np.repeat(z['voxel_coordinates'],np.diff(z['offsets']),axis=0)
        assert len(voxel)==len(z['instance_ids'])==len(z['frame_votes'])
        records={tuple(map(int,row)) : int(vote) for row,vote in zip(np.column_stack((voxel,z['instance_ids'])),z['frame_votes'])}
        assert len(records)==len(voxel)
        counts.append(records)
    changed_keys=[key for key in set(counts[0])|set(counts[1]) if counts[0].get(key,0)!=counts[1].get(key,0)]
    votes=dict(different_voxel_instance_pairs=len(changed_keys),different_spatial_voxels=len(set(k[:3] for k in changed_keys)),
        max_abs_vote_difference=max((abs(counts[0].get(k,0)-counts[1].get(k,0)) for k in changed_keys),default=0),
        absolute_vote_difference_sum=sum(abs(counts[0].get(k,0)-counts[1].get(k,0)) for k in changed_keys),
        affected_instance_ids=sorted(set(k[3] for k in changed_keys)))
    r=dict(final_map=finals,history_pixels=pixels,committed_identity_votes= votes,
        no_GT_quality_score=True,no_modification_of_actual_results=True)
    (OUT/'downstream_difference_diagnosis.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()

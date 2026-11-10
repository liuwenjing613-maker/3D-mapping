"""Exercise frozen repair imports and real causal SceneContinue with the new camera reader."""
from pathlib import Path
import os,sys,json,hashlib,time
scene=sys.argv[1]
CROOT=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
OROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
sys.path.insert(0,str(CROOT/'online'/scene))
from bootstrap_continue import *
import revisable_instance_map.frame_io as frame_io
import numpy as np
expected=CROOT/'mapping_snapshot/revisable_instance_map/src/revisable_instance_map/frame_io.py'
assert Path(frame_io.__file__).resolve()==expected.resolve()
assert r.ReplicaFrameSource is frame_io.ReplicaFrameSource
start=time.time();world=r.Scene(SCENE)
rows=[]
for fid in [0,5,245]:
    frame=world.source.load_frame(fid)
    meta=read(OROOT/'frame_meta'/scene/f'f{fid:06d}.json')
    np.testing.assert_array_equal(frame.camera.intrinsic,meta['intrinsics'])
    np.testing.assert_array_equal(frame.camera_to_world,meta['pose'])
    assert hashlib.sha256(np.ascontiguousarray(frame.depth_m).tobytes()).hexdigest()==meta['depth_decoded_sha256']
    points,ids,stats=r.project_frame_regions(frame,world.tree,pixel_stride=2,max_distance_m=.015,workers=4)
    assert len(points)>0 and np.all(points<len(world.xyz))
    rows.append(dict(frame=fid,projected_points=len(points),camera_exact=True,depth_exact=True))
reference=read(TOP/'tracking100/gate50_equivalence.json')
assert reference['status']=='PASS' and digest(CONT/'track_pure_gate.py')==reference['pure_gate_sha256']
world.verify_unchanged()
record=dict(status='PASS',scene=scene,checkpoint=50,per_frame_reader=str(expected),
    original_gate_hash_equivalence=True,frozen_sources_verified=True,real_prefix_projection=rows,
    GT_used=False,seconds=time.time()-start,scope='runtime import and camera/projection only; full repair awaits selected seeds')
out=ROOT/'repair_runtime_adapter_validation.json';assert not out.exists();dump(out,record);print(json.dumps(record))

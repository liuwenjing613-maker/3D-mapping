from bootstrap import *
import unittest,traceback,os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
def main():
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('test_observation_ops','test_history_core','test_gate_v1','test_strict_repair','test_adaptive_core'))
 result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
 scene=r.Scene('room0')
 from revisable_instance_map.association import OnlineVoxelAssociator
 a=OnlineVoxelAssociator.from_checkpoint(B/'association/observation_support_3cm.npz')
 a.identity_evidence.validate();assert a.last_processed_frame_id==245
 grid=o3d.t.geometry.VoxelBlockGrid.load(str(B/'geometry/tsdf_grid.npz'))
 cloud=grid.extract_point_cloud(weight_threshold=1.).to_legacy();xyz=np.asarray(cloud.points).astype(np.float32)
 assert len(xyz)==len(scene.xyz);dist,idx=scene.tree.query(xyz,k=1,workers=4)
 assert float(dist.max())==0
 # Preserve coincident TSDF vertices with multiplicity; nearest-neighbor indices are not unique.
 np.testing.assert_array_equal(xyz[np.lexsort(xyz.T)],scene.xyz[np.lexsort(scene.xyz.T)])
 guard=read(TOP/'input_guard50.json');raw=next(Path(p) for p in guard['allowed_source_files'] if p.endswith('frame000000.jpg'))
 probes=[raw.parent/'frame000250.jpg',Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room0/abstain/final/instance_surface.npz'),TOP/'ground_truth/gt.npz']
 os.environ['CVPR_GUARD_TEST']='1'
 for p in probes:
  try:open(p,'rb')
  except PermissionError:pass
  else:raise AssertionError('guard allowed '+str(p))
 os.environ.pop('CVPR_GUARD_TEST')
 dump(R/'tests_complete.json',dict(status='PASS',numerical_unit_tests=result.testsRun,identity_checkpoint_consistent=True,TSDF_reload_same_points=True,TSDF_point_order_same=bool(np.array_equal(xyz,scene.xyz)),future_GT_full_result_guard_blocked=True))
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'preflight_failure.json',dict(traceback=traceback.format_exc()));raise

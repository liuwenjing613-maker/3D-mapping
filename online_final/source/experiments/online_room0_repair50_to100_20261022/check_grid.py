from bootstrap import *
import numpy as np,open3d as o3d
scene=r.Scene('room0')
grid=o3d.t.geometry.VoxelBlockGrid.load(str(B/'geometry/tsdf_grid.npz'))
cloud=grid.extract_point_cloud(weight_threshold=1.).to_legacy()
xyz=np.asarray(cloud.points).astype(np.float32);dist,idx=scene.tree.query(xyz,k=1,workers=4)
print('GRID_CHECK',len(xyz),len(scene.xyz),np.quantile(dist,[0,.5,.9,.99,1]).tolist(),len(np.unique(idx)),np.sum(dist>1e-5),flush=True)

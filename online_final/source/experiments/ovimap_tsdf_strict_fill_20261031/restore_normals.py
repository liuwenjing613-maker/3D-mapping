"""Read-only TSDF snapshot -> identical native marching cubes + geometric normals."""
import sys
sys.path.insert(0,'/experiment/build')
from pathlib import Path
import numpy as np
import consistent_gsm as native
from common import *
def restore(source,reference,out):
 before={str(p):sha(p) for p in [source,reference]}
 state=load(source);ref=load(reference)['xyz_m']
 raw=native.cvprRestoreSurface(state['voxel_index'],state['distance_weight'])
 xyz,inv=np.unique(raw['xyz'],axis=0,return_inverse=True)
 np.testing.assert_array_equal(xyz,ref)
 normal=np.asarray(raw['normals'],np.float64)
 finite=np.isfinite(normal).all(axis=1)&(np.linalg.norm(normal,axis=1)>.5)
 sums=np.zeros((len(xyz),3),np.float64)
 for j in range(3):sums[:,j]=np.bincount(inv[finite],weights=normal[finite,j],minlength=len(xyz))
 lengths=np.linalg.norm(sums,axis=1);valid=lengths>1e-6
 sums[valid]/=lengths[valid,None];sums[~valid]=0
 save(out,xyz_m=xyz,normals=sums.astype(np.float32),normal_valid=valid)
 for p,h in before.items():assert sha(p)==h
 report=dict(status='PASS',geometry_bit_exact=True,points=len(xyz),raw_vertices=len(inv),
  valid_normal_points=int(valid.sum()),invalid_normal_points=int((~valid).sum()),
  source_sha256=before,attributes_sha256=sha(out),GT_used=False,
  normal_method='unchanged upstream marching-cubes face normals averaged at exact duplicate coordinates',
  voxel_size_m=.01,voxels_per_side=8,min_weight=1e-4)
 dump(Path(out).with_suffix('.json'),report);print('NORMALS',json.dumps(report),flush=True)
def test():
 # A plane crossing negative and positive block coordinates exercises floor division.
 coords=np.array([(x,y,z) for x in range(-8,8) for y in range(-8,8) for z in range(-8,8)],np.int32)
 values=np.column_stack(((coords[:,2]+.5)*np.float32(.01)-np.float32(.013),np.ones(len(coords)))).astype(np.float32)
 r=native.cvprRestoreSurface(coords,values)
 assert len(r['xyz'])>0
 np.testing.assert_allclose(r['xyz'][:,2],.013,atol=2e-8,rtol=0)
 ok=np.isfinite(r['normals']).all(1)
 assert ok.all()
 np.testing.assert_allclose(np.abs(r['normals'][:,2]),1,atol=1e-6)
 np.testing.assert_allclose(r['normals'][:,:2],0,atol=1e-6)
 dump(O/'restore_plane_test.json',dict(status='PASS',vertices=len(r['xyz']),negative_blocks=True))
if __name__=='__main__':
 if sys.argv[1]=='test':test()
 else:restore(*(Path(x) for x in sys.argv[1:4]))

from alignment_common import *
import sys
sys.path.insert(0,'/experiment/build')
import consistent_gsm as native
source,reference,out=map(Path,sys.argv[1:4])
state=load(source);ref=load(reference)['xyz_m']
raw=native.cvprRestoreSurface(state['voxel_index'],state['distance_weight'])
xyz,inv=np.unique(raw['xyz'],axis=0,return_inverse=True);np.testing.assert_array_equal(xyz,ref)
normal=np.asarray(raw['normals'],np.float64);finite=np.isfinite(normal).all(1)&(np.linalg.norm(normal,axis=1)>.5)
sums=np.zeros((len(xyz),3),np.float64)
for j in range(3):sums[:,j]=np.bincount(inv[finite],weights=normal[finite,j],minlength=len(xyz))
lengths=np.linalg.norm(sums,axis=1);valid=lengths>1e-6;sums[valid]/=lengths[valid,None];sums[~valid]=0
save(out,xyz_m=xyz,normals=sums.astype(np.float32),normal_valid=valid)
dump(out.with_suffix('.json'),dict(status='PASS',geometry_bit_exact=True,points=len(xyz),GT_used=False))

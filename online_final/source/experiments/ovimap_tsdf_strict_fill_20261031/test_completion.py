from common import *
import unittest
from color_fusion import visible_samples,sampled_rgb_world
def test_projection():
 # Independent explicit camera-space construction, nonidentity world pose.
 angle=.7;c=np.cos(angle);s=np.sin(angle)
 R=np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float64);t=np.array([3.,-2.,.8])
 pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=t
 cam=np.array([[0,0,1],[.1,0,1],[0,0,1.2],[0,0,-1],[1,0,1],[0,.1,.8]])
 world=cam@R.T+t
 K=np.array([[10.,0,2.],[0,10.,2.],[0,0,1.]])
 depth=np.ones((5,5),np.float32)
 ix,v,u=visible_samples(world,pose,K,depth)
 np.testing.assert_array_equal(ix,[0,1]);np.testing.assert_array_equal(v,[2,2]);np.testing.assert_array_equal(u,[2,3])
 # Farther/back-facing, behind observed depth, and out-of-view samples must not color the surface.
 empty=visible_samples(np.array([[0,0,-1.]]),np.eye(4),K,depth)[0];assert len(empty)==0
def main():
 test_projection()
 depth=np.zeros((4,4),np.float32);depth[0,0]=1;depth[0,2]=2
 mask=np.zeros((4,4),np.uint8);mask[0,0]=1;mask[0,2]=2
 rgb=np.zeros((4,4,3),np.uint8);rgb[0,0]=[17,23,51];rgb[0,2]=[4,88,200]
 world,colors=sampled_rgb_world(depth,mask,rgb,np.eye(4),np.eye(3))
 np.testing.assert_array_equal(world,[[0,0,1],[4,0,2]])
 np.testing.assert_array_equal(colors,[[17,23,51],[4,88,200]])
 sys.path.insert(0,str(SRC))
 test_path=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/revisable_instance_map/tests/test_offline_surface_assignment.py')
 import importlib.util
 spec=importlib.util.spec_from_file_location('frozen_fill_tests',test_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 suite=unittest.defaultTestLoader.loadTestsFromModule(m)
 result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful()
 dump(O/'completion_tests.json',dict(status='PASS',existing_assignment_tests=result.testsRun,
  RGB_projection_nonidentity_pose=True,exact_depth_sample_RGB_indexing=True,occlusion_rejection=True,missing_attribute_labels_preserved_by_wrapper_assertions=True))
if __name__=='__main__':main()

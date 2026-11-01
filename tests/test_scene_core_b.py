import unittest
import numpy as np
from scene_completion.core import complete

def fixture(state=0,top=-1,ids=None,z=0.,flip=False):
    angle=np.arange(6)*np.pi/3
    pts=np.c_[.01*np.cos(angle),.01*np.sin(angle),np.full(6,z)]
    xyz=np.r_[pts,[[0.,0.,0.]]]
    normals=np.tile([0.,0.,1.],(7,1))
    if flip:normals[:6]*=-1
    labels=np.r_[np.full(6,7) if ids is None else ids,-1].astype(np.int32)
    ev={'state':np.r_[np.full(6,2),state].astype(np.uint8),'top1_instance_id':np.r_[labels[:6],top].astype(np.int32)}
    return xyz,normals,labels,ev

class CompletionTests(unittest.TestCase):
    def test_unobserved_consensus(self):
        x,n,l,e=fixture();out,_,_=complete(x,n,l,e)
        self.assertEqual(out[-1],7);np.testing.assert_array_equal(out[:6],l[:6]);self.assertEqual(l[-1],-1)
    def test_tentative_requires_own_top1(self):
        for top,expected in [(7,7),(8,-1)]:
            x,n,l,e=fixture(1,top);self.assertEqual(complete(x,n,l,e)[0][-1],expected)
    def test_conflict_stays_unassigned(self):
        x,n,l,e=fixture(3,7);self.assertEqual(complete(x,n,l,e)[0][-1],-1)
    def test_competing_ids_rejected(self):
        x,n,l,e=fixture(ids=np.array([7,7,7,8,8,8]));self.assertEqual(complete(x,n,l,e)[0][-1],-1)
    def test_parallel_sheet_rejected(self):
        x,n,l,e=fixture(z=.004);self.assertEqual(complete(x,n,l,e)[0][-1],-1)
    def test_opposite_normals_rejected(self):
        x,n,l,e=fixture(flip=True);self.assertEqual(complete(x,n,l,e)[0][-1],-1)
    def test_six_required(self):
        x,n,l,e=fixture();x[0,0]=1.;self.assertEqual(complete(x,n,l,e)[0][-1],-1)
    def test_existing_filled_points_never_become_seeds(self):
        x,n,l,e=fixture()
        x=np.r_[x,np.tile([[.005,0.,0.]],(6,1)),[[.029,0.,0.]]]
        n=np.tile([0.,0.,1.],(len(x),1))
        l=np.r_[l,np.full(6,7),-1].astype(np.int32)
        e={k:np.r_[v,np.zeros(6,dtype=v.dtype) if k=='state' else np.full(6,-1,dtype=v.dtype),0 if k=='state' else -1] for k,v in e.items()}
        out,_,_=complete(x,n,l,e);self.assertEqual(out[6],7);self.assertEqual(out[-1],-1)
        np.testing.assert_array_equal(out[l>0],l[l>0])
if __name__=='__main__':unittest.main()

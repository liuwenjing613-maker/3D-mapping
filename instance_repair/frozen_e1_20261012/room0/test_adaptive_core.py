import unittest
import numpy as np
from scipy.ndimage import binary_erosion
from adaptive_core import choose_core

class AdaptiveCoreTests(unittest.TestCase):
    def mask(self, h, w):
        a=np.zeros((h+4,w+4),bool);a[2:-2,2:-2]=True;return a
    def project(self, a): return np.flatnonzero(a)
    def test_large_keeps_original(self):
        m=self.mask(100,100);c,d=choose_core(m,self.project)
        self.assertEqual(d['radius'],10)
        np.testing.assert_array_equal(c,binary_erosion(m,iterations=10))
    def test_small_retains_interior(self):
        m=self.mask(10,20);c,d=choose_core(m,self.project)
        self.assertEqual(d['radius'],3)
        self.assertGreaterEqual(c.sum(),40)
        self.assertTrue(np.all(~c|binary_erosion(m,iterations=1)))
    def test_thin_abstains_without_boundary_fallback(self):
        m=self.mask(1,100);c,d=choose_core(m,self.project)
        self.assertFalse(c.any());self.assertIsNone(d['radius'])
    def test_counts_unique_3d_points_not_pixels(self):
        m=self.mask(40,40);c,d=choose_core(m,lambda a:np.zeros(int(a.sum()),int))
        self.assertFalse(c.any());self.assertEqual(d['seed_points'],1)
    def test_projection_holes_trigger_less_erosion(self):
        m=self.mask(30,30);valid=np.zeros_like(m);valid[4:7,4:28]=True
        c,d=choose_core(m,lambda a:np.flatnonzero(a&valid))
        self.assertLess(d['radius'],10);self.assertGreaterEqual(d['retained_points'],20)
    def test_empty_has_no_core(self):
        c,d=choose_core(np.zeros((10,10),bool),self.project)
        self.assertFalse(c.any());self.assertEqual(d['seed_points'],0)

if __name__=='__main__':unittest.main()

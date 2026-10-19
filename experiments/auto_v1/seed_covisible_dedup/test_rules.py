import unittest
import numpy as np
from rules import pair_metrics, fixed_primary_groups, order_key


class RulesTests(unittest.TestCase):
    def pair(self, common=30, extras=70):
        a=np.arange(common+extras)
        b=np.r_[np.arange(common),np.arange(1000,1000+extras)]
        return a,b,np.arange(common),np.arange(common)

    def test_new_evidence_can_rescue_low_original_iou(self):
        m=pair_metrics(*self.pair())
        self.assertLess(m['original_iou'],.25)
        self.assertEqual(m['fraction_a'],.3)
        self.assertEqual(m['score'],1)
        self.assertTrue(m['duplicate'])

    def test_both_fractions_required(self):
        a,b,av,bv=self.pair()
        b=np.r_[b,2000]
        self.assertFalse(pair_metrics(a,b,av,bv)['duplicate'])
        self.assertFalse(pair_metrics(b,a,bv,av)['duplicate'])

    def test_minimum_points_inclusive(self):
        self.assertTrue(pair_metrics(*self.pair(20,0))['duplicate'])
        self.assertFalse(pair_metrics(*self.pair(19,0))['duplicate'])

    def test_iou_boundary_inclusive(self):
        a=np.arange(25);b=np.arange(20)
        self.assertTrue(pair_metrics(a,b,a,b)['duplicate'])
        self.assertFalse(pair_metrics(np.arange(26),b,np.arange(26),b)['duplicate'])

    def test_empty_evidence_keeps(self):
        z=np.array([],dtype=int)
        self.assertFalse(pair_metrics(z,z,z,z)['duplicate'])

    def test_no_intersection_keeps(self):
        a=np.arange(100);b=np.arange(100,200)
        self.assertFalse(pair_metrics(a,b,a,b)['duplicate'])

    def test_symmetry(self):
        a,b,av,bv=self.pair()
        x=pair_metrics(a,b,av,bv);y=pair_metrics(b,a,bv,av)
        for key in ['score','original_iou','duplicate','reason','evidence_sufficient']:
            self.assertEqual(x[key],y[key])

    def test_fixed_primary_no_transitive_deletion(self):
        edges={frozenset(('A','B')),frozenset(('B','C'))}
        self.assertEqual(fixed_primary_groups(['A','B','C'],lambda a,b:frozenset((a,b)) in edges),[['A','B'],['C']])

    def test_order_preserves_original_tiebreak(self):
        def s(n,f,m,d):return dict(support_count=n,frame=f,representative={'model':m},digest=d)
        rows=[s(1,1,1,'b'),s(2,9,9,'c'),s(1,1,0,'d'),s(1,1,1,'a')]
        self.assertEqual(sorted(range(4),key=lambda i:order_key(rows[i])),[1,2,3,0])


if __name__=='__main__':unittest.main()

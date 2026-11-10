import unittest
import numpy as np
from coverage import measure


class CoverageTests(unittest.TestCase):
    def test_unassigned_in_seed_and_instance_outside_seed(self):
        labels=np.array([1]*6+[-1]*2+[1]*4)
        m=measure(np.arange(8),np.arange(12),labels,np.full(12,2))
        self.assertEqual(m['p'],6/8);self.assertEqual(m['r'],6/10)
        self.assertEqual(m['instance_visible_count'],10)
        self.assertEqual(m['final_unassigned_count'],2)

    def test_map_split_has_low_p_but_high_r(self):
        m=measure(np.arange(10),np.arange(10),np.array([1]*5+[2]*5),np.full(10,2))
        self.assertEqual(m['p'],.5);self.assertEqual(m['r'],1)
        self.assertEqual(m['tied_largest_ids'],[1,2]);self.assertFalse(m['both95'])

    def test_partial_seed_or_overmerged_map_is_not_resolved_by_coverage(self):
        m=measure(np.arange(10),np.arange(100),np.ones(100,int),np.full(100,2))
        self.assertEqual(m['p'],1);self.assertEqual(m['r'],.1)
        self.assertFalse(m['both95']);self.assertNotIn('skip',m)

    def test_inclusive95_boundary(self):
        labels=np.array([1]*19+[-1,1])
        m=measure(np.arange(20),np.arange(21),labels,np.full(21,2))
        self.assertEqual(m['p'],.95);self.assertEqual(m['r'],.95);self.assertTrue(m['both95'])
        labels=np.r_[labels,1]
        self.assertFalse(measure(np.arange(20),np.arange(22),labels,np.full(22,2))['both95'])

    def test_no_assigned_points(self):
        m=measure(np.arange(2),np.arange(3),np.array([-1,-1,9]),np.array([0,3,2]))
        self.assertIsNone(m['instance_id']);self.assertEqual((m['p'],m['r']),(0,0));self.assertFalse(m['both95'])

    def test_conflict_is_measured_not_silently_filtered(self):
        m=measure(np.arange(20),np.arange(20),np.ones(20,int),np.array([3]+[2]*19))
        self.assertTrue(m['both95']);self.assertEqual(m['native_state_counts']['conflict'],1)
        self.assertEqual(m['assigned_but_native_conflict_count'],1)

    def test_instance_denominator_uses_only_visible_points(self):
        m=measure(np.arange(20),np.arange(20),np.ones(50,int),np.full(50,2))
        self.assertEqual(m['r'],1);self.assertEqual(m['instance_visible_count'],20)


if __name__=='__main__':unittest.main()

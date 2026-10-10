"""Independent set/dictionary checks for simultaneous multi-object semantics."""
import unittest
import numpy as np
from history_core_ops import core_claims,retarget_selection,strict_frame

class HistoryCoreTests(unittest.TestCase):
    def test_alias_union_and_conflicting_manual_targets(self):
        owner,_=core_claims([[0,1],[1,2],[2,3]],[796,796,179],5,1000)
        np.testing.assert_array_equal(owner,[796,796,-2,179,-1])
    def test_conflicting_point_keeps_old_observation(self):
        owner,_=core_claims([[0],[0]],[796,179],2,1000)
        self.assertFalse(retarget_selection([0],[57],[True],owner)[0])
    def test_retired_observation_is_not_recreated(self):
        owner,_=core_claims([[0]],[796],2,1000)
        np.testing.assert_array_equal(retarget_selection([0,0],[179,179],[False,True],owner),[False,True])
    def test_existing_correct_support_stays_and_deduplicates(self):
        owner,_=core_claims([[0]],[796],2,1000)
        choose=retarget_selection([0,0],[796,57],[True,True],owner)
        np.testing.assert_array_equal(choose,[False,True])
        np.testing.assert_array_equal(strict_frame([796,796],1000),[796])
    def test_protected_new_mask_keeps_conflict(self):
        # Original179 is re-associated, but frozen new179 support remains.
        self.assertEqual(len(strict_frame([179,796],1000)),0)
    def test_empty_core_has_no_fallback(self):
        owner,_=core_claims([[]],[796],3,1000)
        self.assertFalse(retarget_selection([0,1],[179,57],[True,True],owner).any())
    def test_order_invariance(self):
        a,_=core_claims([[0,1,1],[1,2],[3]],[4,5,6],5,100)
        b,_=core_claims([[3],[1,2],[1,0]],[6,5,4],5,100)
        np.testing.assert_array_equal(a,b)
    def test_independent_randomized_point_and_pixel_oracle(self):
        rng=np.random.default_rng(74109)
        for _ in range(100):
            claims=[rng.integers(0,20,size=int(rng.integers(0,25))).tolist() for _ in range(5)]
            targets=rng.integers(1,8,size=5).tolist();expected={p:set() for p in range(20)}
            for points,pid in zip(claims,targets):
                for p in points:expected[p].add(pid)
            owner,_=core_claims(claims,targets,20,100)
            oracle=[next(iter(expected[p])) if len(expected[p])==1 else -2 if expected[p] else -1 for p in range(20)]
            np.testing.assert_array_equal(owner,oracle)
            p=rng.integers(0,20,size=80);old=rng.integers(1,8,size=80);ret=rng.integers(0,2,size=80).astype(bool)
            choose=retarget_selection(p,old,ret,owner)
            for i in range(len(p)):
                want=ret[i] and len(expected[int(p[i])])==1 and int(old[i]) not in expected[int(p[i])]
                self.assertEqual(bool(choose[i]),bool(want))
            candidates=list(p*100+np.where(choose,owner[p],old));by_point={}
            for key in candidates:by_point.setdefault(int(key)//100,set()).add(int(key)%100)
            exact=sorted(point*100+next(iter(ids)) for point,ids in by_point.items() if len(ids)==1)
            np.testing.assert_array_equal(strict_frame(candidates,100),exact)

if __name__=='__main__':unittest.main()

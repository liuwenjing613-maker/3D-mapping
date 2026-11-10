"""Meaningful tests for overlapping-case vote and observation correctness."""
import unittest
import numpy as np
import joint_batch_ops as j

class JointTests(unittest.TestCase):
    def test_aliases_share_one_identity(self):
        c=j.canonical_map([1,5,9],{9:5}); self.assertEqual(c,{1:1,5:5,9:5})
    def test_alias_cycles_rejected(self):
        with self.assertRaises(ValueError): j.canonical_map([1,2],{1:2,2:1})
    def test_missing_alias_target_rejected(self):
        with self.assertRaises(ValueError): j.canonical_map([1,2],{2:3})
    def test_same_object_overlap_is_union_not_conflict(self):
        labels,conflict=j.merge_foregrounds([(5,np.array([[1,1,0]],bool)),(9,np.array([[0,1,1]],bool))],{5:5,9:5})
        np.testing.assert_array_equal(labels,[[5,5,5]]);self.assertFalse(conflict.any())
    def test_different_objects_overlap_abstains(self):
        labels,conflict=j.merge_foregrounds([(4,np.array([[1,1,0]],bool)),(8,np.array([[0,1,1]],bool))],{4:4,8:8})
        np.testing.assert_array_equal(labels,[[4,0,8]]);np.testing.assert_array_equal(conflict,[[0,1,0]])
    def test_case_order_does_not_change_conflicts(self):
        p=[(4,np.array([[1,1,0]],bool)),(8,np.array([[0,1,1]],bool))]
        a=j.merge_foregrounds(p,{4:4,8:8});b=j.merge_foregrounds(p[::-1],{4:4,8:8})
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    def test_unexplained_sibling_prevents_whole_retirement(self):
        self.assertEqual(j.combined_plan(['whole'],[4,8],[4],False),'partial')
    def test_all_siblings_explained_allows_old_whole_rule(self):
        self.assertEqual(j.combined_plan(['whole'],[4,8],[4,8],False),'whole')
    def test_ambiguous_pixel_preserves_old_observation(self):
        self.assertEqual(j.combined_plan(['whole'],[4],[4],True),'partial')
    def test_partial_case_cannot_escalate_to_whole(self):
        self.assertEqual(j.combined_plan(['partial'],[4],[4],False),'partial')
    def test_empty_claims_retain(self):
        self.assertEqual(j.combined_plan([],[],[],False),'retain')
    def test_duplicate_same_frame_vote_rejected(self):
        with self.assertRaises(ValueError): j.count_vote_keys([np.array([10,10])])
    def test_votes_from_distinct_frames_count_separately(self):
        k,v=j.count_vote_keys([np.array([10,20]),np.array([10,30])])
        np.testing.assert_array_equal(k,[10,20,30]);np.testing.assert_array_equal(v,[2,1,1])

if __name__=='__main__': unittest.main()

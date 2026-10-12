import unittest
import numpy as np
from types import SimpleNamespace
from observation_ops import *

class TestObservationOps(unittest.TestCase):
 def test_occluded_surface_not_visible(self):
  cam=SimpleNamespace(fx=1,fy=1,cx=0,cy=0);v,_,_=visible_samples(np.array([[0,0,1],[0,0,2],[0,0,-1],[3,0,1]]),np.eye(4),cam,np.ones((2,2)))
  np.testing.assert_array_equal(v,[True,False,False,False])
 def test_duplicate_view_frames_do_not_count(self):
  p=np.array([[0,0,0],[0,0,.01],[0,0,.06]]);np.testing.assert_array_equal(choose_view_representatives(p),[0,2])
 def test_same_camera_center_is_not_independent_view(self):
  self.assertEqual(len(choose_view_representatives(np.zeros((2,3)))),1)
 def test_disconnected_support_does_not_expand(self):
  c,r=connected_expansion(np.array([[0,0,0],[.01,0,0],[1,0,0]]),[1,0,0],[0,1,1]);np.testing.assert_array_equal(c,[1,1,0])
 def test_weak_bridge_is_not_used(self):
  c,r=connected_expansion(np.array([[0,0,0],[.015,0,0],[.03,0,0]]),[1,0,0],[0,0,1]);np.testing.assert_array_equal(c,[1,0,0])
 def test_normal_discontinuity_stops_growth(self):
  c,r=connected_expansion(np.array([[0,0,0],[.01,0,0]]),[1,0],[0,1],normals=np.array([[0,0,1],[1,0,0]]));np.testing.assert_array_equal(c,[1,0])
 def test_history_preserves_unavailable_and_occluded_pixels(self):
  ids,change=revise_history(np.array([0,1,2]),np.array([218,218,218]),np.array([1,0,1],bool),np.full(3,-1),np.ones(3,bool),np.array([1,1,0],bool));np.testing.assert_array_equal(ids,[817,218,218])
 def test_shared_vote_survives_one_support_removal(self):
  p=np.array([7,7]);ids=np.array([218,218]);self.assertEqual(len(keyset(p[1:],ids[1:],832)),1)
 def test_source_split_then_same_frame_conflict_abstains(self):
  k=keyset([7,7,8],[817,218,218],832);np.testing.assert_array_equal(strict_frame(k,832),[8*832+218])
 def test_foreign_new_pixels_local_only(self):
  reject,amb=constrain_new(np.array([0,1,2]),np.array([218,218,817]),np.array([817,-1,218]),np.ones(3,bool));np.testing.assert_array_equal(reject,[1,0,1]);self.assertFalse(amb.any())
 def test_ambiguous_core_is_not_assigned(self):
  reject,amb=constrain_new(np.array([0]),np.array([817]),np.array([-2]),np.ones(1,bool));self.assertFalse(reject.any());self.assertTrue(amb[0])
 def test_strict_unique_even_repeated_supports(self):
  k=strict_frame(keyset([0,0,1,1],[817,817,218,817],832),832);np.testing.assert_array_equal(k,[817])

class Room0Tests(unittest.TestCase):
 def test_alias_and_conflict_order_independent(self):
  sets=[np.array([0,1]),np.array([1,2]),np.array([2,3])];ids=[10,10,20]
  np.testing.assert_array_equal(resolve_claims(sets,ids,5),[10,10,-2,20,-1])
  np.testing.assert_array_equal(resolve_claims(sets,ids,5),resolve_claims(sets[::-1],ids[::-1],5))
 def test_visible_only_history_split(self):
  p=np.array([0,0,1,2,3]);old=np.array([20]*5);avail=np.array([1,0,1,1,1],bool);prior=np.array([-1,-1,-1,20]);owner=np.array([10,10,-2,30]);vis=np.array([1,0,1,1],bool)
  ids,c=revise_all_history(p,old,avail,prior,owner,vis)
  np.testing.assert_array_equal(ids,[10,20,20,20,30]);np.testing.assert_array_equal(c,[1,0,0,0,1])
 def test_local_new_constraints(self):
  reject,amb=constrain_all_new(np.arange(5),np.array([10,20,20,20,20]),np.array([10,10,-2,10,-1]),np.array([1,1,1,0,1],bool))
  np.testing.assert_array_equal(reject,[0,1,0,0,0]);np.testing.assert_array_equal(amb,[0,0,1,0,0])
 def test_ambiguous_bridge_does_not_extend(self):
  xyz=np.array([[0,0,0],[.01,0,0],[.02,0,0]])
  chosen,_=connected_expansion(xyz,np.array([1,0,0],bool),np.array([0,0,1],bool),.015)
  np.testing.assert_array_equal(chosen,[1,0,0])
 def test_no_seed_no_unanchored_growth(self):
  c,_=connected_expansion(np.array([[0,0,0],[.01,0,0]]),[False,False],[True,True],.02)
  assert not c.any()
 def test_reverse_support_not_sparse_projection_frequency(self):
  cam=SimpleNamespace(fx=1,fy=1,cx=0,cy=0)
  vis,pos=mask_support(np.array([[0,0,1],[.2,0,1],[0,0,2]]),np.eye(4),cam,np.ones((2,2)),np.ones((2,2),bool))
  np.testing.assert_array_equal(pos,[True,True,False])

if __name__=='__main__':unittest.main()

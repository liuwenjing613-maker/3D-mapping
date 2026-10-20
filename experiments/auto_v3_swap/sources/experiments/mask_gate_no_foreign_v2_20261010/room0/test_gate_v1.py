import unittest
from gate_v1 import object_reliability,visibility_metrics
P={'minimum_visible_seed_surface_samples_each_target':20,'minimum_visible_seed_fraction_each_target':.05,
 'minimum_tracked_coverage_of_visible_seed_samples_each_target':.8,'minimum_new_surface_support_fraction_inside_target_bbox':.2,
 'maximum_new_mask_pixel_overlap_fraction_with_other_old_instance_families':.05,'minimum_visible_protected_seed_samples':20,
 'maximum_new_union_coverage_of_other_visible_seed_samples':.1}
class GateTests(unittest.TestCase):
 def metric(self,**kw):
  m=dict(visible_seed_samples=100,visible_seed_fraction=.05,tracked_coverage=.8,new_surface_bbox_fraction=.2,foreign_old_mask_pixel_fraction=.05,other_seed_overlaps=[]);m.update(kw);return m
 def test_inclusive_boundaries(self):self.assertEqual(object_reliability(self.metric(),P),(True,[]))
 def test_count_and_fraction_independent(self):
  for kw in [dict(visible_seed_samples=19,visible_seed_fraction=1),dict(visible_seed_samples=100,visible_seed_fraction=.04999)]:self.assertIn('insufficient_seed_visibility',object_reliability(self.metric(**kw),P)[1])
 def test_2048_anchor_count(self):
  self.assertLess(visibility_metrics(2048,102)['visible_seed_fraction'],.05);self.assertGreater(visibility_metrics(2048,103)['visible_seed_fraction'],.05)
 def test_no_anchor(self):self.assertEqual(visibility_metrics(0,0)['visible_seed_fraction'],0)
 def test_invalid_anchor_count(self):
  with self.assertRaises(ValueError):visibility_metrics(10,11)
 def test_new_surface_expansion_allowed(self):self.assertTrue(object_reliability(self.metric(new_surface_bbox_fraction=.21),P)[0])
 def test_extreme_drift(self):self.assertIn('extreme_spatial_drift',object_reliability(self.metric(new_surface_bbox_fraction=.1999),P)[1])
 def test_anchor_coverage_still_required(self):self.assertIn('anchor_coverage',object_reliability(self.metric(tracked_coverage=.7999),P)[1])
 def test_old_family_overlap_diagnostic_only(self):
  for value in [0,.05,.05001,.5,1.]:self.assertEqual(object_reliability(self.metric(foreign_old_mask_pixel_fraction=value),P),(True,[]))
 def test_other_failures_still_reject_at_full_foreign_overlap(self):
  for kw,reason in [(dict(tracked_coverage=.7),'anchor_coverage'),(dict(new_surface_bbox_fraction=.1),'extreme_spatial_drift'),(dict(visible_seed_samples=19),'insufficient_seed_visibility')]:self.assertIn(reason,object_reliability(self.metric(foreign_old_mask_pixel_fraction=1.,**kw),P)[1])
 def test_other_target_overlap_diagnostic_only(self):
  for n,c in [(20,.1),(19,1),(20,.10001),(2048,1)]:self.assertEqual(object_reliability(self.metric(other_seed_overlaps=[dict(track_id=2,visible_seed_samples=n,coverage=c)]),P),(True,[]))
 def test_trimmed_mask_rejected(self):
  self.assertTrue(object_reliability(self.metric(tracked_coverage=.9),P)[0]);self.assertFalse(object_reliability(self.metric(tracked_coverage=.7),P)[0])
if __name__=='__main__':unittest.main()

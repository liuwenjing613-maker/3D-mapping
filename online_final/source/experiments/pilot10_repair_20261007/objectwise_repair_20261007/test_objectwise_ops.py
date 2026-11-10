"""Regression tests for reassociation and partial evidence preservation."""
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sam_track_publication_20261007/sam_track_code_20261007/pilot10_repair_20261007'))
sys.path.insert(0, str(HERE.parent))
import objectwise_ops as o
from joint_vote_ops import keys_from_pairs, batch_delta


class ObjectwiseTests(unittest.TestCase):
    def setUp(self):
        import json
        self.policy = json.loads((HERE / 'policy.json').read_text())
        self.original = np.array([[8, 8, 0], [7, 7, 2]], np.uint16)
        self.objects = [{'track_id': 4, 'original_mask_id': 8}, {'track_id': 7, 'original_mask_id': 7}]
        self.rows = [{'track_id': 4, 'persistent_id': 52}, {'track_id': 7, 'persistent_id': 355}]
        self.metrics = {'visible_seed_samples': 25, 'tracked_coverage': .95, 'new_surface_bbox_fraction': .99,
                        'foreign_old_mask_pixel_fraction': .01, 'other_seed_overlaps': []}

    def test_original_reassociate_keeps_each_mask_exactly(self):
        labels = o.seed_labels(self.original, None, self.objects, 'original_reassociate')
        for obj in self.objects:
            np.testing.assert_array_equal(labels == obj['track_id'], self.original == obj['original_mask_id'])
        rows = o.freeze_association(self.rows, self.objects, 'original_reassociate', {8: 52, 7: 52})
        self.assertEqual([r['persistent_id'] for r in rows], [52, 355])

    def test_explicit_control_keeps_bad_many_to_one(self):
        rows = o.freeze_association(self.rows, self.objects, 'original_control', {8: 52, 7: 52})
        self.assertEqual([r['persistent_id'] for r in rows], [52, 52])

    def test_ambiguous_original_mode_rejected(self):
        with self.assertRaises(ValueError):
            o.seed_labels(self.original, None, self.objects, 'original')

    def test_new_mask_requires_full_coordinates(self):
        with self.assertRaises(ValueError):
            o.seed_labels(self.original, np.ones((1, 1), np.uint16), self.objects, 'new_mask_reassociate')

    def test_missing_original_mask_rejected(self):
        with self.assertRaises(ValueError):
            o.seed_labels(self.original, None, [{'track_id': 1, 'original_mask_id': 99}], 'original_reassociate')

    def test_repair_rejects_many_to_one(self):
        with self.assertRaises(ValueError):
            o.freeze_association([{'track_id': 4, 'persistent_id': 52}, {'track_id': 7, 'persistent_id': 52}], self.objects, 'original_reassociate', {})

    def test_a_reliable_b_invisible_a_still_accepted(self):
        self.assertTrue(o.object_reliability(self.metrics, self.policy)[0])
        invisible = {**self.metrics, 'visible_seed_samples': 0}
        self.assertFalse(o.object_reliability(invisible, self.policy)[0])
        self.assertEqual(o.observation_plan([4], [4, 7], 1., True, self.policy), 'partial')

    def test_other_object_anchor_is_protected(self):
        metrics = {**self.metrics, 'other_seed_overlaps': [{'track_id': 7, 'visible_seed_samples': 25, 'coverage': .2}]}
        self.assertFalse(o.object_reliability(metrics, self.policy)[0])

    def test_both_reliable_full_replacement(self):
        self.assertEqual(o.observation_plan([4, 7], [4, 7], .95, True, self.policy), 'whole')

    def test_large_old_mask_preserves_unexplained_part(self):
        self.assertEqual(o.observation_plan([4, 7], [4, 7], .8, True, self.policy), 'partial')

    def test_nothing_reliable_no_retirement(self):
        self.assertEqual(o.observation_plan([], [4, 7], 1., True, self.policy), 'retain')

    def test_partial_retains_unknown_and_shared_pixel_support(self):
        # One merged observation; point 0 has both explained and retained pixels.
        points = np.array([0, 1, 2, 3]); local = np.array([1, 1, 1, 2]); table = np.array([0, 52, 9])
        revised, retained, retired, scope = o.replace_observations(points, local, table, {1: 'partial'},
            np.array([0, 2]), np.array([1, 1]), np.array([0, 1]), np.array([355, 355]), 1000)
        np.testing.assert_array_equal(retained, [52, 2052, 3009])
        np.testing.assert_array_equal(revised, [52, 355, 1355, 2052, 3009])
        np.testing.assert_array_equal(retired, [[1, 1]])
        np.testing.assert_array_equal(scope, [0, 1])
        self.assertNotIn(2, scope)  # unknown residual not broadened into intervention

    def test_unretired_other_mask_still_supplies_same_frame_vote(self):
        points = np.array([0, 0, 1]); local = np.array([1, 2, 1]); table = np.array([0, 52, 52])
        revised, retained, _, _ = o.replace_observations(points, local, table, {1: 'whole'},
            np.array([], int), np.array([], int), np.array([1]), np.array([355]), 1000)
        np.testing.assert_array_equal(retained, [52])
        np.testing.assert_array_equal(revised, [52, 1355])

    def test_delta_replay_and_exact_rollback(self):
        old = np.array([52, 1052, 2052]); revised = np.array([52, 1355, 2052]); votes = np.array([2, 3, 4])
        rem, add = np.setdiff1d(old, revised), np.setdiff1d(revised, old)
        k, v = batch_delta(old, votes, rem, add)
        np.testing.assert_array_equal(k, [52, 1052, 1355, 2052])
        np.testing.assert_array_equal(v, [2, 2, 1, 4])
        bk, bv = batch_delta(k, v, add, rem)
        np.testing.assert_array_equal(bk, old); np.testing.assert_array_equal(bv, votes)

    def test_residual_cannot_invent_old_contribution(self):
        with self.assertRaises(ValueError):
            o.replace_observations(np.array([0]), np.array([1]), np.array([0, 52]), {1: 'partial'},
                np.array([99]), np.array([1]), np.array([], int), np.array([], int), 1000)


if __name__ == '__main__':
    unittest.main()

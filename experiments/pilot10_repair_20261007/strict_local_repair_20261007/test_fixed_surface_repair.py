"""Small regression cases for correspondence invariance and commit boundaries."""
from pathlib import Path
import os
import unittest
import uuid

import numpy as np
import fixed_surface_repair as f


class RepairSafetyTests(unittest.TestCase):
    def setUp(self):
        # Normal inherited permissions also work in the Windows workspace sandbox.
        parent = Path(os.environ.get('CVPR_REPAIR_TEST_ROOT', str(Path.cwd() / '.repair_test_runs')))
        self.output = parent / uuid.uuid4().hex
        self.output.mkdir(parents=True)
        self.xyz = np.asarray([[0., 0., 0.], [.005, 0., 0.], [1., 0., 0.]], np.float32)
        self.rgb = np.zeros((3, 3), np.uint8)
        self.surface = {'xyz_m': self.xyz, 'rgb': self.rgb, 'instance_id': np.asarray([52, 52, 8], np.int32)}

    def mapping(self):
        return f.fixed_mapping(self.output / 'map.npz', self.xyz, self.xyz.copy(), .01, workers=1)

    def test_unassigned_vertex_does_not_borrow_neighbour_identity(self):
        mapping = self.mapping()
        labels = f.mapped_labels(mapping, np.asarray([-1, 355, 8], np.int32))
        self.assertEqual(labels[0], -1)
        self.assertEqual(labels[1], 355)

    def test_cache_reused_across_label_changes(self):
        first = self.mapping()
        second = self.mapping()
        self.assertEqual(first['cache_sha256'], second['cache_sha256'])
        np.testing.assert_array_equal(first['nearest_surface_index'], second['nearest_surface_index'])
        f.mapped_labels(second, np.asarray([355, -1, 8], np.int32))
        self.assertEqual(first['cache_sha256'], f.file_sha256(self.output / 'map.npz'))

    def test_reordered_geometry_rejected(self):
        self.mapping()
        with self.assertRaises(ValueError):
            f.fixed_mapping(self.output / 'map.npz', self.xyz[::-1], self.xyz, .01)

    def test_changed_reference_rejected(self):
        self.mapping()
        with self.assertRaises(ValueError):
            f.fixed_mapping(self.output / 'map.npz', self.xyz, self.xyz + .001, .01)

    def test_changed_distance_rejected(self):
        self.mapping()
        with self.assertRaises(ValueError):
            f.fixed_mapping(self.output / 'map.npz', self.xyz, self.xyz, .02)

    def test_corrupt_cached_indices_rejected(self):
        self.mapping()
        archive = f.load_arrays(self.output / 'map.npz')
        archive['nearest_surface_index'][0] = 1
        f.atomic_npz(self.output / 'map.npz', **archive)
        with self.assertRaises(ValueError):
            self.mapping()

    def test_missing_geometry_is_reported_separately(self):
        ref = np.asarray([[0., 0., 0.], [5., 0., 0.]], np.float32)
        mapping = f.fixed_mapping(self.output / 'map.npz', self.xyz, ref, .01, workers=1)
        rows = f.target_diagnostics(mapping, self.surface['instance_id'], np.asarray([4001, 4001]), np.ones(2, bool), {52: 4001})
        self.assertEqual(rows['targets'][0]['no_geometric_correspondence_reference_points'], 1)
        self.assertEqual(rows['targets'][0]['expected_identity_full_GT_coverage'], .5)

    def test_local_commit_restores_outside_labels_and_preserves_inside(self):
        candidate = {**self.surface, 'instance_id': np.asarray([355, -1, 99], np.int32)}
        labels, proof, blocked = f.bounded_final_labels(self.surface, candidate, np.asarray([0, 1], np.int32))
        np.testing.assert_array_equal(labels, [355, -1, 8])
        np.testing.assert_array_equal(blocked, [2])
        self.assertEqual(proof['final_outside_changes'], 0)
        self.assertEqual(proof['candidate_outside_changes_blocked'], 1)

    def test_empty_allowed_set_is_noop(self):
        candidate = {**self.surface, 'instance_id': np.asarray([355, -1, 99], np.int32)}
        labels, _, _ = f.bounded_final_labels(self.surface, candidate, np.asarray([], np.int32))
        np.testing.assert_array_equal(labels, self.surface['instance_id'])

    def test_changed_geometry_or_rgb_rejected(self):
        for field in ['xyz_m', 'rgb']:
            candidate = {**self.surface, field: self.surface[field] + 1}
            with self.assertRaises(ValueError):
                f.bounded_final_labels(self.surface, candidate, np.asarray([0], np.int32))

    def test_invalid_or_duplicate_allowed_ids_rejected(self):
        for ids in [[0, 0], [-1], [3]]:
            with self.assertRaises(ValueError):
                f.allowed_mask(np.asarray(ids, np.int32), 3)
        with self.assertRaises(ValueError):
            f.allowed_mask(np.asarray([0.5]), 3)

    def test_label_only_changes_do_not_hide_errors_as_missing_geometry(self):
        mapping = self.mapping()
        truth = np.asarray([4001, 4002, 4002])
        report = f.target_diagnostics(mapping, np.asarray([52, -1, 355], np.int32), truth, np.ones(3, bool), {52: 4001, 355: 4002})
        target = report['targets'][1]
        self.assertEqual(target['GT_reference_points'], 2)
        self.assertEqual(target['reachable_fixed_surface_reference_points'], 2)
        self.assertEqual(target['unassigned_reachable_reference_points'], 1)
        self.assertEqual(target['expected_identity_full_GT_coverage'], .5)

    def test_wrong_to_correct_and_correct_to_unassigned_transitions(self):
        mapping = self.mapping()
        report = f.target_transitions(mapping, np.asarray([52, 52, 355], np.int32),
                                     np.asarray([52, 355, -1], np.int32), np.asarray([4001, 4002, 4002]),
                                     np.ones(3, bool), {52: 4001, 355: 4002})
        self.assertEqual(report[1]['wrong_positive_to_correct'], 1)
        self.assertEqual(report[1]['correct_to_unassigned'], 1)

    def test_raw_vote_and_state_escape_rejected(self):
        evidence = {'state': np.asarray([2, 2, 1], np.uint8)}
        pairs = {'surface_point_index': np.asarray([0, 1, 2]), 'instance_id': np.asarray([52, 52, 8]),
                 'frame_votes': np.asarray([2, 2, 1])}
        ids = np.asarray([0], np.int32)
        changed = {'state': np.asarray([2, 2, 2], np.uint8)}
        with self.assertRaises(ValueError):
            f.verify_raw_outside(evidence, changed, pairs, pairs, ids)
        changed_pairs = {**pairs, 'frame_votes': np.asarray([2, 2, 9])}
        with self.assertRaises(ValueError):
            f.verify_raw_outside(evidence, evidence, pairs, changed_pairs, ids)
        reordered = {key: value[::-1] for key, value in pairs.items()}
        self.assertTrue(f.verify_raw_outside(evidence, evidence, pairs, reordered, ids)['all_vote_counts_unchanged_outside'])

    def test_known_whole_map_metadata_is_separate_from_point_evidence(self):
        evidence = {'state': np.asarray([2, 2, 1], np.uint8)}
        before = {**evidence, 'map_version': np.asarray([400]),
                  'association_decisions_sha256': np.asarray(['frozen-association-hash'])}
        pairs = {'surface_point_index': np.asarray([0, 1, 2]), 'instance_id': np.asarray([52, 52, 8]),
                 'frame_votes': np.asarray([2, 2, 1])}
        pairs_with_metadata = {**pairs, 'map_version': before['map_version'],
                               'association_decisions_sha256': before['association_decisions_sha256']}
        proof = f.verify_raw_outside(before, evidence, pairs_with_metadata, pairs, np.asarray([0], np.int32))
        self.assertEqual(proof['storage_metadata_difference']['baseline_only'], ['association_decisions_sha256', 'map_version'])
        self.assertEqual(proof['verified_per_point_fields'], ['state'])
        self.assertEqual(proof['storage_metadata_difference']['vote_ledger_baseline_only'], ['association_decisions_sha256', 'map_version'])

    def test_missing_real_evidence_field_and_changed_metadata_rejected(self):
        evidence = {'state': np.asarray([2, 2, 1], np.uint8), 'top1_votes': np.asarray([2, 2, 1])}
        pairs = {'surface_point_index': np.asarray([0, 1, 2]), 'instance_id': np.asarray([52, 52, 8]),
                 'frame_votes': np.asarray([2, 2, 1])}
        with self.assertRaises(ValueError):
            f.verify_raw_outside(evidence, {'state': evidence['state']}, pairs, pairs, np.asarray([0], np.int32))
        with self.assertRaises(ValueError):
            f.verify_raw_outside({**evidence, 'map_version': np.asarray([400])},
                                 {**evidence, 'map_version': np.asarray([399])}, pairs, pairs, np.asarray([0], np.int32))
        with self.assertRaises(ValueError):
            f.verify_raw_outside(evidence, evidence, {**pairs, 'map_version': np.asarray([400])},
                                 {**pairs, 'map_version': np.asarray([399])}, np.asarray([0], np.int32))


if __name__ == '__main__':
    unittest.main()

import copy
import json
import unittest
from pathlib import Path
from pipeline import (containment_small, fixed_primary_groups, iou_duplicate, pr_excluded,
                      run_filters, semantic_excluded, skip_repair)
from run import validate_reference


class BoundaryTests(unittest.TestCase):
    def test_semantic_ninety_percent_inclusive(self):
        self.assertTrue(semantic_excluded(90, 100))
        self.assertFalse(semantic_excluded(89, 100))

    def test_iou_only_accepts_tiny_evidence(self):
        self.assertTrue(iou_duplicate(4, 5))
        self.assertFalse(iou_duplicate(3, 4))
        self.assertFalse(iou_duplicate(0, 0))

    def test_fixed_representative_does_not_transitively_merge(self):
        edges = {frozenset(('A', 'B')), frozenset(('B', 'C'))}
        self.assertEqual(fixed_primary_groups(['A', 'B', 'C'],
            lambda a, b: frozenset((a, b)) in edges), [['A', 'B'], ['C']])

    def test_B_requires_both_coverages_and_includes_boundary(self):
        self.assertTrue(skip_repair(19, 20, 20))
        self.assertTrue(skip_repair(1, 1, 1))
        self.assertFalse(skip_repair(19, 20, 21))
        self.assertFalse(skip_repair(0, 1, 0))

    def test_full_containment_uses_strict_boundary(self):
        self.assertIsNone(containment_small('A', 10, 'B', 100, 7))
        self.assertEqual(containment_small('A', 10, 'B', 100, 8), ('A', 'B'))
        self.assertEqual(containment_small('B', 100, 'A', 10, 8), ('A', 'B'))
        with self.assertRaisesRegex(ValueError, 'user choice'):
            containment_small('A', 10, 'B', 10, 8)

    def test_pr_strict_AND_and_zero_instance(self):
        self.assertFalse(pr_excluded(10, 100, 100))
        self.assertTrue(pr_excluded(11, 100, 100))
        self.assertFalse(pr_excluded(20, 100, 100))
        self.assertFalse(pr_excluded(1, 100, 100))
        self.assertFalse(pr_excluded(0, 100, 0))
        self.assertTrue(pr_excluded(10001, 100000, 100000))


class FrozenRoom2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).parent / 'fixtures' / 'room2_20261019'
        cls.measured = json.loads((folder / 'measurements.json').read_text(encoding='utf-8'))
        cls.overrides = json.loads((folder / 'manual_overrides.json').read_text(encoding='utf-8'))
        cls.reference = json.loads((folder / 'reference_decisions.json').read_text(encoding='utf-8'))

    def test_all_historical_stage_decisions_and_final_35(self):
        before = copy.deepcopy(self.measured)
        result = run_filters(self.measured, self.overrides)
        validate_reference(result, self.reference)
        self.assertEqual([s['retained'] for s in result['stages']], [72, 56, 45, 42, 33])
        self.assertEqual(len(result['retained_group_ids']), 35)
        self.assertEqual(result['manual_restored_group_ids'], ['G017', 'G025'])
        self.assertEqual(result['final_pr_excluded_group_ids'],
                         ['G069', 'G071', 'G074', 'G075', 'G077', 'G082', 'G086'])
        self.assertEqual(self.measured, before)

    def test_automatic_mode_keeps_manual_restores_separate(self):
        result = run_filters(self.measured)
        validate_reference(result, self.reference, automatic_only=True)
        self.assertEqual(len(result['retained_group_ids']), 33)
        self.assertEqual(result['manual_restored_group_ids'], [])
        self.assertNotIn('G017', result['retained_group_ids'])

    def test_missing_pair_fails_instead_of_assuming_disjoint(self):
        measured = copy.deepcopy(self.measured)
        measured['pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing complete pair'):
            run_filters(measured, self.overrides)

    def test_coverage_identity_mismatch_fails(self):
        measured = copy.deepcopy(self.measured)
        measured['coverage'][0]['frame'] += 1
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            run_filters(measured, self.overrides)

    def test_manual_restore_cannot_reintroduce_prior_exclusions(self):
        overrides = copy.deepcopy(self.overrides)
        overrides['retain'][0]['group_id'] = 'G080'
        with self.assertRaisesRegex(ValueError, 'manual restore'):
            run_filters(self.measured, overrides)

    def test_manual_restore_requires_original_pixels_and_scope(self):
        overrides = copy.deepcopy(self.overrides)
        overrides['retain'][0]['mask_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            run_filters(self.measured, overrides)
        overrides = copy.deepcopy(self.overrides)
        overrides['input_group_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'input scope changed'):
            run_filters(self.measured, overrides)


if __name__ == '__main__':
    unittest.main()

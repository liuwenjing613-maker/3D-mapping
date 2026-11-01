import copy
import importlib.util
import json
from pathlib import Path
import unittest
from instance_repair.seed_filter_rules import run_filters, skip_repair

ROOT = Path(__file__).resolve().parents[2]


def measurements(c=90, n=100, i=100):
    seed = dict(group_id='G001', original_track_id=1, mask_id='M001', frame=0,
                mask_sha256='a' * 64, mask_digest='a' * 64, source_model=0,
                support_count=n, area=100, high_excluded_pixels=0)
    return dict(schema='auto-v2-count-measurements-v1', case_id='threshold-test',
                seeds=[seed], pairs=[], coverage=[dict(seed, intersection=c,
                instance_visible_count=i, instance_id=4 if i else 0)])


class SkipRepairTests(unittest.TestCase):
    def test_inclusive_both_directions(self):
        self.assertTrue(skip_repair(90, 100, 100))
        self.assertTrue(skip_repair(90, 90, 100))
        self.assertTrue(skip_repair(90, 100, 90))

    def test_either_direction_below_threshold_keeps_repair(self):
        self.assertFalse(skip_repair(89, 100, 89))
        self.assertFalse(skip_repair(89, 89, 100))
        self.assertFalse(skip_repair(89, 100, 100))

    def test_unmatched_seed_keeps_repair(self):
        self.assertFalse(skip_repair(0, 100, 0))

    def test_office0_reported_mask(self):
        self.assertTrue(skip_repair(11424, 12692, 11448))

    def test_invalid_counts_are_rejected(self):
        for values in [(1, 0, 1), (-1, 100, 100), (101, 100, 100),
                       (1, 100, 0), (90.0, 100, 100)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                skip_repair(*values)

    def test_pipeline_uses_new_threshold_without_changing_identity(self):
        data = measurements()
        original = copy.deepcopy(data)
        selected = run_filters(data)
        self.assertEqual(selected['retained_group_ids'], [])
        self.assertEqual(selected['records'][0]['terminal_stage'], 'skip_repair_B')
        self.assertEqual(selected['records'][0]['original_track_id'], 1)
        self.assertEqual(data, original)
        self.assertEqual(run_filters(measurements(89))['retained_group_ids'], ['G001'])

    def test_semantic_exemption_cannot_restore_skipped_mask(self):
        config = json.loads((ROOT/'experiments/seed_skip90_20261101/config.json').read_text())
        path = ROOT/config['semantic_exemption_rules']
        spec = importlib.util.spec_from_file_location('skip90_semantic', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for class_name in ['windowpane', 'cabinet']:
            selected = module.apply_semantic_exemption(run_filters(measurements()),
                dict(records=[dict(group_id='G001', top_class=class_name)]))
            self.assertEqual(selected['retained_group_ids'], [])

    def test_current_registry_points_to_new_rules(self):
        registry = json.loads((ROOT/'instance_repair/configs/current_auto_pipeline.json').read_text())
        config = json.loads((ROOT/registry['seed_pipeline_config']).read_text())
        self.assertEqual(config['skip_repair']['precision_min'], 0.9)
        self.assertEqual(config['skip_repair']['recall_min'], 0.9)
        self.assertEqual(registry['seed_filter_module'], 'instance_repair.seed_filter_rules')
        self.assertEqual(registry['seed_boundary_policy'], 'full-frame-and-crop-edge-20261101')
        self.assertEqual(registry['final_publication']['policy'], 'scene-final-B-20261101')


if __name__ == '__main__':
    unittest.main()

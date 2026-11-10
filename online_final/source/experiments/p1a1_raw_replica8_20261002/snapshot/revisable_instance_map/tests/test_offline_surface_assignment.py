"""Behavioral safety checks on geometries with known surface ownership."""
import sys
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/revisable_instance_map'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_surface_assignment import AssignmentSettings, assign_surface, validate_inputs


def evidence_for(xyz, candidates):
    n = len(xyz)
    e = {'state': np.zeros(n, np.uint8), 'top1_instance_id': np.full(n, -1, np.int32),
         'top1_votes': np.zeros(n, np.int32), 'top2_instance_id': np.full(n, -1, np.int32),
         'top2_votes': np.zeros(n, np.int32), 'total_frame_votes': np.zeros(n, np.int32),
         'confidence': np.zeros(n, np.float32)}
    pp, ii, vv = [], [], []
    for point, observed in enumerate(candidates):
        if not observed:
            continue
        ranked = sorted(observed, key=lambda row: (-row[1], row[0]))
        top, count = ranked[0]
        total = sum(v for _, v in ranked)
        e['top1_instance_id'][point], e['top1_votes'][point] = top, count
        e['total_frame_votes'][point], e['confidence'][point] = total, count / total
        if len(ranked) > 1:
            e['top2_instance_id'][point], e['top2_votes'][point] = ranked[1]
        e['state'][point] = 3 if len(ranked) > 1 and count / total < .67 else (2 if count >= 2 and count / total >= .67 else 1)
        for instance, votes in sorted(observed):
            pp.append(point)
            ii.append(instance)
            vv.append(votes)
    pair = {k: np.array(v, np.int32) for k, v in
            [('surface_point_index', pp), ('instance_id', ii), ('frame_votes', vv)]}
    return e, pair, np.where(e['state'] == 2, e['top1_instance_id'], -1).astype(np.int32)


def grid(width=13, height=13):
    xyz = np.array([[x * .01, y * .01, 0] for y in range(height) for x in range(width)], float)
    return xyz, np.tile([0., 0., 1.], (len(xyz), 1)), np.full_like(xyz, .5)


def run(xyz, normals, rgb, candidates, settings=None):
    e, p, base = evidence_for(xyz, candidates)
    out, detail, stats = assign_surface(xyz, normals, rgb, e, p, base, settings)
    return out, detail, stats, e, p, base


class AssignmentGeometryTests(unittest.TestCase):
    def test_enclosed_unknown_hole_and_immutable_evidence(self):
        xyz, normals, rgb = grid()
        center = 6 * 13 + 6
        candidates = [[(1, 10)] for _ in xyz]
        candidates[center] = []
        e, pair, base = evidence_for(xyz, candidates)
        snapshots = [a.copy() for a in [xyz, normals, rgb, base, *e.values(), *pair.values()]]
        out, d, s = assign_surface(xyz, normals, rgb, e, pair, base)
        self.assertEqual(out['holes_only'][center], 1)
        self.assertEqual(d['assignment_route'][0], 2)
        for original, copy in zip([xyz, normals, rgb, base, *e.values(), *pair.values()], snapshots):
            np.testing.assert_array_equal(original, copy)

    def test_mixed_three_state_component(self):
        xyz, normals, rgb = grid()
        points = [6 * 13 + 6, 6 * 13 + 7, 7 * 13 + 6]
        candidates = [[(1, 10)] for _ in xyz]
        candidates[points[0]] = []
        candidates[points[1]] = [(1, 1)]
        candidates[points[2]] = [(1, 2), (2, 2)]
        out, d, s, *_ = run(xyz, normals, rgb, candidates)
        np.testing.assert_array_equal(out['holes_only'][points], [1, 1, 1])
        self.assertEqual(s['holes']['accepted_components'], 1)
        self.assertEqual(len(np.unique(d['hole_component_id'])), 1)

    def test_open_surface_edge_is_not_a_hole(self):
        xyz, normals, rgb = grid()
        p = 6 * 13
        candidates = [[(1, 10)] for _ in xyz]
        candidates[p] = []
        out, d, s, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(out['holes_only'][p], -1)

    def test_opposite_faces_do_not_connect(self):
        xyz, normals, rgb = grid(9, 9)
        second = xyz.copy()
        second[:, 2] += .001
        all_xyz = np.r_[xyz, second]
        all_normals = np.r_[normals, -normals]
        all_rgb = np.r_[rgb, rgb]
        candidates = [[(1, 10)] for _ in xyz] + [[] for _ in second]
        out, d, s, *_ = run(all_xyz, all_normals, all_rgb, candidates)
        self.assertTrue((out['holes_geodesic'][len(xyz):] < 0).all())
        self.assertGreater(s['normal_orientation_uncertain_points'], 0)

    def test_parallel_disjoint_faces_do_not_connect(self):
        xyz, normals, rgb = grid(9, 9)
        second = xyz.copy()
        second[:, 2] += .004
        out, *_ = run(np.r_[xyz, second], np.r_[normals, normals], np.r_[rgb, rgb],
                     [[(1, 10)] for _ in xyz] + [[] for _ in second])
        self.assertTrue((out['holes_geodesic'][len(xyz):] < 0).all())

    def test_candidate_outside_top_two_can_be_accepted(self):
        xyz, normals, rgb = grid()
        p = 6 * 13 + 6
        candidates = [[(12, 10)] for _ in xyz]
        candidates[p] = [(12, 1), (13, 2), (14, 2)]
        out, d, s, e, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(e['top1_instance_id'][p], 13)
        self.assertEqual(e['top2_instance_id'][p], 14)
        self.assertEqual(out['holes_geodesic'][p], 12)
        self.assertEqual(d['selected_direct_frame_votes'][0], 1)

    def test_tentative_contradiction_is_retained(self):
        xyz, normals, rgb = grid()
        p = 6 * 13 + 6
        candidates = [[(1, 10)] for _ in xyz]
        candidates[p] = [(2, 1)]
        out, d, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(out['holes_geodesic'][p], -1)
        self.assertEqual(d['rejection_reason'][0], 3)

    def test_weak_coherent_instance_without_confirmed_core_is_protected(self):
        xyz, normals, rgb = grid()
        weak = [y * 13 + x for y in (5, 6, 7) for x in (5, 6, 7)]
        candidates = [[(1, 10)] for _ in xyz]
        for p in weak:
            candidates[p] = [(1, 2), (2, 2)]
        out, d, s, *_ = run(xyz, normals, rgb, candidates)
        self.assertTrue((out['holes_geodesic'][weak] < 0).all())
        self.assertGreater(s['by_state']['CONFLICT']['rejections'].get('coherent_competing_evidence', 0), 0)

    def test_single_isolated_confirmed_point_cannot_seed(self):
        xyz, normals, rgb = grid()
        candidates = [[] for _ in xyz]
        candidates[6 * 13 + 6] = [(1, 10)]
        out, _, s, *rest = run(xyz, normals, rgb, candidates)
        np.testing.assert_array_equal(out['holes_geodesic'], rest[-1])
        self.assertEqual(s['reliable_original_boundary_seeds'], 0)

    def test_original_path_budget_does_not_renew(self):
        xyz, normals, rgb = grid(25, 9)
        candidates = [[(1, 10)] if p[0] <= .04 + 1e-10 else [] for p in xyz]
        out, d, s, *_ = run(xyz, normals, rgb, candidates)
        self.assertGreater(np.sum(out['holes_geodesic'] > 0), sum(bool(x) for x in candidates))
        self.assertTrue((out['holes_geodesic'][xyz[:, 0] > .08] < 0).all())
        accepted = d['assignment_route'] > 0
        self.assertTrue((d['geometric_path_length_m'][accepted] <= s['path_budget_m'] + 1e-8).all())

    def test_two_different_instances_tied_are_retained(self):
        xyz, normals, rgb = grid()
        candidates = [[(1 if p[0] < .06 else 2, 10)] for p in xyz]
        center_column = np.flatnonzero(np.abs(xyz[:, 0] - .06) < 1e-10)
        for p in center_column:
            candidates[p] = []
        out, d, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(out['holes_geodesic'][6 * 13 + 6], -1)
        middle = np.flatnonzero(d['surface_point_index'] == 6 * 13 + 6)[0]
        self.assertEqual(d['rejection_reason'][middle], 6)
        self.assertNotEqual(d['proposed_instance_id'][middle], d['second_instance_id'][middle])

    def test_complete_path_records_end_at_original_seed(self):
        xyz, normals, rgb = grid(19, 9)
        candidates = [[(1, 10)] if p[0] <= .04 + 1e-10 else [] for p in xyz]
        out, d, _, _, _, base = run(xyz, normals, rgb, candidates)
        for row, lo, hi in zip(d['accepted_path_target_rows'], d['accepted_path_offsets'][:-1], d['accepted_path_offsets'][1:]):
            path = d['accepted_path_surface_indices'][lo:hi]
            self.assertEqual(path[0], d['surface_point_index'][row])
            self.assertEqual(path[-1], d['source_seed_index'][row])
            self.assertEqual(base[path[-1]], d['proposed_instance_id'][row])
            self.assertTrue((base[path[:-1]] < 0).all())
            length = np.linalg.norm(np.diff(xyz[path], axis=0), axis=1).sum()
            self.assertAlmostEqual(length, d['geometric_path_length_m'][row], places=9)

    def test_pattern_color_changes_cost_without_cutting_surface(self):
        xyz, normals, rgb = grid()
        p = 6 * 13 + 6
        candidates = [[(1, 10)] for _ in xyz]
        candidates[p] = []
        rgb[p] = [1, 0, 0]
        out, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(out['holes_geodesic'][p], 1)

    def test_published_competitor_is_not_removed_for_hole_check(self):
        xyz, normals, rgb = grid()
        p = 6 * 13 + 6
        candidates = [[(1, 10)] for _ in xyz]
        candidates[p] = []
        candidates[p + 1] = [(2, 2)]  # Weak confirmed B still counts at boundary.
        out, _, s, *_ = run(xyz, normals, rgb, candidates)
        self.assertEqual(out['holes_only'][p], -1)
        self.assertEqual(out['holes_geodesic'][p], -1)
        self.assertGreater(s['holes'].get('competing_boundary_labels', 0), 0)

    def test_duplicate_candidate_is_rejected(self):
        xyz, normals, rgb = grid(5, 5)
        e, pair, base = evidence_for(xyz, [[(1, 10)] for _ in xyz])
        for k in pair:
            pair[k] = np.r_[pair[k], pair[k][:1]]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_inputs(xyz, normals, rgb, e, pair, base)

    def test_wrong_full_votes_are_rejected(self):
        xyz, normals, rgb = grid(5, 5)
        candidates = [[(1, 10)] for _ in xyz]
        candidates[12] = []
        e, pair, base = evidence_for(xyz, candidates)
        pair['frame_votes'][0] += 1
        with self.assertRaisesRegex(ValueError, 'votes differ'):
            validate_inputs(xyz, normals, rgb, e, pair, base)


if __name__ == '__main__':
    unittest.main(verbosity=2)

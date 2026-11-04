"""Meaningful set-math, ambiguity, causality and view-independence checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from instance_repair.two_audit_matching import (
    Observation, TrackBuilder, TrackSettings, intersection_count, pair_evidence)
from instance_repair.two_audit_tracks import build_tracks, validate_tracks
from instance_repair.two_audit_cache import build_cache
from instance_repair.two_audit_sources import sha256_file
from instance_repair.tests.test_two_audit_cache import fixture, certify_geometry


def points(start, stop):
    return np.arange(start, stop, dtype=np.int32)


def pose(x=0., yaw=0., roll=0.):
    y, r = np.radians(yaw), np.radians(roll)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    value = np.eye(4)
    value[:3, :3] = ry @ rz
    value[0, 3] = x
    return value


def add(builder, fid, supports, *, visible=None, camera=None, evidence=None):
    rows = [Observation(f"f{fid}/m{i+1}", i+1, s) for i, s in enumerate(supports)]
    return builder.process_frame(fid, points(0, 500) if visible is None else visible, rows,
                                 pose() if camera is None else camera, geometry_id="geometry",
                                 evidence_sink=None if evidence is None else evidence.append)


class MatchingTests(unittest.TestCase):
    def engine(self):
        return TrackBuilder("geometry", 500)

    def test_exact_intersection_matches_independent_python_sets(self):
        rng = np.random.default_rng(44)
        for _ in range(100):
            a = np.sort(rng.choice(500, size=int(rng.integers(0, 200)), replace=False))
            b = np.sort(rng.choice(500, size=int(rng.integers(0, 200)), replace=False))
            self.assertEqual(intersection_count(a, b), len(set(a) & set(b)))

    def test_common_visibility_not_full_support_iou(self):
        a, b = points(0, 100), points(75, 175)
        result = pair_evidence(a, a, b, b, TrackSettings())
        self.assertEqual(result["common_a"], 25)
        self.assertEqual(result["common_b"], 25)
        self.assertEqual(result["covisible_iou"], 1.)
        self.assertEqual(result["state"], "MATCH")
        self.assertLess(len(set(a) & set(b)) / len(set(a) | set(b)), .55)

    def test_insufficient_common_support_is_unknown_even_for_perfect_overlap(self):
        a, b = points(0, 100), points(76, 176)
        result = pair_evidence(a, a, b, b, TrackSettings())
        self.assertEqual(result["intersection"], 24)
        self.assertEqual(result["state"], "INSUFFICIENT_COMMON_SUPPORT")
        self.assertIsNone(result["covisible_iou"])

    def test_iou_threshold_is_inclusive(self):
        visible = points(0, 200)
        at = pair_evidence(points(0, 80), visible, points(25, 100), visible, TrackSettings())
        below = pair_evidence(points(0, 80), visible, points(26, 101), visible, TrackSettings())
        self.assertEqual(at["covisible_iou"], .55)
        self.assertEqual(at["state"], "MATCH")
        self.assertEqual(below["state"], "BELOW_IOU")

    def test_fixed_representative_blocks_transitive_chain(self):
        b = self.engine()
        add(b, 0, [points(0, 100)])
        add(b, 5, [points(20, 120)])
        result = add(b, 10, [points(40, 140)])
        self.assertEqual(len(b.tracks), 2)
        self.assertEqual(b.tracks[0].anchor.key, "f0/m1")
        self.assertEqual(len(b.tracks[0].members), 2)
        self.assertEqual(result[0]["status"], "NEW_TRACK")

    def test_multiple_matches_are_deferred_without_new_track_or_vote(self):
        b = self.engine()
        add(b, 0, [points(0, 70)])
        add(b, 5, [points(30, 100)])
        result = add(b, 10, [points(0, 100)])
        self.assertEqual(result[0]["status"], "DEFERRED_AMBIGUOUS")
        self.assertEqual(len(result[0]["matched_track_ids"]), 2)
        self.assertIsNone(result[0]["track_id"])
        self.assertEqual(len(b.tracks), 2)
        self.assertEqual([len(t.members) for t in b.tracks], [1, 1])

    def test_whole_frame_conflict_does_not_choose_first_mask(self):
        b = self.engine()
        add(b, 0, [points(0, 100)])
        result = add(b, 5, [points(0, 90), points(10, 100)])
        self.assertEqual([x["status"] for x in result],
                         ["DEFERRED_SAME_FRAME_CONFLICT"] * 2)
        self.assertEqual(len(b.tracks[0].members), 1)

    def test_same_frame_cannot_match_a_new_same_frame_representative(self):
        b = self.engine()
        result = add(b, 0, [points(0, 100), points(0, 100)])
        self.assertEqual([x["status"] for x in result], ["NEW_TRACK"] * 2)
        self.assertEqual(b.stats["comparisons"], 0)
        self.assertEqual(len(b.tracks), 2)

    def test_three_observations_and_two_translation_views_make_stable(self):
        b = self.engine()
        for fid, x in [(0, 0.), (5, .05), (10, .1)]:
            add(b, fid, [points(0, 50)], camera=pose(x=x))
        t = b.tracks[0]
        self.assertTrue(t.stable)
        self.assertEqual(len(t.view_groups), 2)
        self.assertEqual(t.first_stable_frame_id, 10)

    def test_repeated_static_camera_does_not_count_as_independent_views(self):
        b = self.engine()
        for fid in (0, 5, 10, 15):
            add(b, fid, [points(0, 50)])
        self.assertEqual(len(b.tracks[0].members), 4)
        self.assertEqual(len(b.tracks[0].view_groups), 1)
        self.assertFalse(b.tracks[0].stable)

    def test_roll_is_not_optical_axis_change_but_ten_degree_yaw_is(self):
        b = self.engine()
        add(b, 0, [points(0, 50)])
        add(b, 5, [points(0, 50)], camera=pose(roll=60.))
        self.assertEqual(len(b.tracks[0].view_groups), 1)
        add(b, 10, [points(0, 50)], camera=pose(yaw=10.))
        self.assertEqual(len(b.tracks[0].view_groups), 2)
        self.assertTrue(b.tracks[0].stable)

    def test_view_groups_have_fixed_representatives_not_chained_centres(self):
        b = self.engine()
        for fid, x in [(0, 0.), (5, .09), (10, .18)]:
            add(b, fid, [points(0, 50)], camera=pose(x=x))
        groups = b.tracks[0].view_groups
        self.assertEqual([g["representative_frame_id"] for g in groups], [0, 10])
        self.assertEqual(groups[0]["member_frame_ids"], [0, 5])

    def test_small_and_zero_support_remain_pending(self):
        b = self.engine()
        result = add(b, 0, [points(0, 0), points(0, 24), points(30, 55)])
        self.assertEqual([x["status"] for x in result],
                         ["DEFERRED_INSUFFICIENT_SUPPORT",
                          "DEFERRED_INSUFFICIENT_SUPPORT", "NEW_TRACK"])
        self.assertEqual(b.stats["observations"], 3)
        self.assertEqual(len(b.tracks), 1)

    def test_shared_support_shortage_is_not_recorded_as_disagreement(self):
        b, evidence = self.engine(), []
        add(b, 0, [points(0, 100)], visible=points(0, 100))
        result = add(b, 5, [points(76, 176)], visible=points(76, 176), evidence=evidence)
        self.assertEqual(evidence[0]["state"], "INSUFFICIENT_COMMON_SUPPORT")
        self.assertEqual(result[0]["evidence_state"], "INSUFFICIENT_COMMON_SUPPORT")
        self.assertEqual(result[0]["status"], "NEW_TRACK")
        self.assertFalse(any(t.stable for t in b.tracks))

    def test_exact_candidate_index_matches_all_positive_set_intersections(self):
        b, rng = self.engine(), np.random.default_rng(20)
        anchors = [np.sort(rng.choice(500, 70, replace=False)).astype(np.int32) for _ in range(12)]
        add(b, 0, anchors)
        for _ in range(20):
            query = np.sort(rng.choice(500, 90, replace=False)).astype(np.int32)
            expected = {t.index: len(set(t.anchor.support) & set(query)) for t in b.tracks}
            expected = {k: v for k, v in expected.items() if v}
            self.assertEqual(dict(b.candidate_intersections(query)), expected)

    def test_geometry_and_temporal_boundaries(self):
        b = self.engine()
        with self.assertRaisesRegex(ValueError, "geometry"):
            b.process_frame(0, points(0, 50), [], pose(), geometry_id="other")
        add(b, 0, [points(0, 50)])
        with self.assertRaisesRegex(ValueError, "increasing"):
            add(b, 0, [points(0, 50)])
        with self.assertRaisesRegex(ValueError, "only once"):
            b.process_frame(5, points(0, 100), [Observation("f0/m1", 1, points(0, 50))],
                            pose(), geometry_id="geometry")

    def test_support_must_be_visible_unique_and_in_range(self):
        for support, visibility in [(points(0, 50), points(0, 40)),
                                    (np.array([1, 1], np.int32), points(0, 50)),
                                    (np.array([-1, 0], np.int32), points(0, 50))]:
            with self.assertRaises(ValueError):
                add(self.engine(), 0, [support], visible=visibility)

    def test_anchor_and_visibility_are_immutable_copies(self):
        b = self.engine()
        support, visible = points(0, 50), points(0, 100)
        add(b, 0, [support], visible=visible)
        support[:] += 100
        visible[:] = visible[::-1]
        result = add(b, 5, [points(0, 50)], visible=points(0, 100))
        self.assertEqual(result[0]["status"], "MATCHED_TRACK")
        np.testing.assert_array_equal(b.tracks[0].anchor.support, points(0, 50))

    def test_future_observations_do_not_rewrite_prefix_decisions(self):
        a, b = self.engine(), self.engine()
        da, db = [], []
        for fid, x in [(0, 0.), (5, .05), (10, .1)]:
            da.extend(add(a, fid, [points(0, 50)], camera=pose(x=x)))
            db.extend(add(b, fid, [points(0, 50)], camera=pose(x=x)))
        snapshot = json.loads(json.dumps(a.tracks[0].members))
        add(b, 15, [points(0, 50)], camera=pose(x=.2))
        self.assertEqual(da, db)
        self.assertEqual(snapshot, b.tracks[0].members[:3])
        self.assertEqual(a.tracks[0].track_key, b.tracks[0].track_key)

    def test_mask_input_order_does_not_change_frame_decisions(self):
        a, b = self.engine(), self.engine()
        rows = [Observation("second", 2, points(100, 150)),
                Observation("first", 1, points(0, 50))]
        da = a.process_frame(0, points(0, 200), rows, pose(), geometry_id="geometry")
        db = b.process_frame(0, points(0, 200), list(reversed(rows)), pose(), geometry_id="geometry")
        self.assertEqual(da, db)
        self.assertEqual(a.records(), b.records())

    def test_invalid_count_or_zero_iou_threshold_is_rejected(self):
        for settings in [dict(min_common_points=0), dict(min_common_points=25.5),
                         dict(min_covisible_iou=0), dict(view_axis_degrees=181)]:
            with self.assertRaises(ValueError):
                TrackSettings(**settings)


class TrackArtifactTests(unittest.TestCase):
    def test_real_cache_adapter_complete_accounting_determinism_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="two-audit-tracks-") as temporary:
            root = Path(temporary)
            mapping, _ = fixture(root)
            xyz = np.column_stack((np.linspace(.001, .03, 30),
                                   np.zeros(30), np.ones(30))).astype(np.float32)
            np.savez_compressed(mapping / "baseline/native/instance_surface.npz",
                                xyz_m=xyz, instance_id=np.full(30, 999, np.int32))
            certify_geometry(mapping, xyz)
            cache_root = root / "cache"
            # Only this two-frame IO fixture uses min_observations=2.
            settings = TrackSettings(min_observations=2)
            with redirect_stdout(io.StringIO()):
                build_cache(mapping, cache_root, mode="offline", checkpoint=2)
                first = build_tracks(cache_root, root / "first", settings=settings)
                second = build_tracks(cache_root, root / "second", settings=settings)
            result = validate_tracks(root / "first", verify_inputs=True)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(first["totals"]["stable_tracks"], 1)
            self.assertEqual(first["totals"]["assigned_observations"], 2)
            self.assertEqual(first["totals"]["deferred_observations"], 6)
            self.assertEqual(first["run_id"], second["run_id"])
            for name in ["tracks.json", "stable_tracks.json", "pair_evidence.jsonl.gz",
                         "observation_decisions.jsonl"]:
                self.assertEqual(sha256_file(root / "first" / name),
                                 sha256_file(root / "second" / name))
            with self.assertRaises(FileExistsError):
                build_tracks(cache_root, root / "first", settings=settings)
            summary = root / "first/track_summary.csv"
            summary.write_text(summary.read_text() + "corruption\n")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_tracks(root / "first")


if __name__ == "__main__":
    unittest.main()

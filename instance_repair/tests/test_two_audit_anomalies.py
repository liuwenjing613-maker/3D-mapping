"""Numerical, causal, provenance and complete-artifact tests for phase 3."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from instance_repair.two_audit_anomaly_core import (
    AnomalySettings, StableMapAudit, add_joint_view, add_roi_comparison)
from instance_repair.two_audit_anomalies import build_anomalies, validate_anomalies
from instance_repair.two_audit_cache import ObservationCache, build_cache
from instance_repair.two_audit_map_inputs import AuditMapInputs, read_ids
from instance_repair.two_audit_sources import file_record, sha256_file
from instance_repair.two_audit_tracks import build_tracks
from instance_repair.tests.test_two_audit_cache import fixture, certify_geometry, dump


def pose(x=0., yaw=0., roll=0.):
    r, a = np.radians(roll), np.radians(yaw)
    p = np.eye(4)
    p[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                          [-np.sin(a), 0, np.cos(a)]]) @ np.array([
                              [np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    p[0, 3] = x
    return p


def spec(tid, fid, mid, support):
    return (tid, f"{tid}-f{fid}", mid, np.asarray(support, np.int32))


def engine(ids, frames):
    grouped = {}
    for fid, rows in frames.items():
        for tid, key, mid, support in rows:
            grouped.setdefault(tid, []).append(dict(observation_key=key, frame_id=fid,
                 mask_local_id=mid, surface_support_count=len(support), view_group_id=1))
    tracks = [dict(track_id=tid, track_key="key-"+tid, status="STABLE", geometry_id="g",
                   first_stable_frame_id=max(m["frame_id"] for m in members),
                   representative_observation_key=members[0]["observation_key"],
                   representative_frame_id=members[0]["frame_id"], members=members)
              for tid, members in grouped.items()]
    return StableMapAudit(ids, tracks, "g", "native-hash"), tracks


def run_case(ids, frames, poses=None):
    audit, tracks = engine(ids, frames)
    for fid, rows in sorted(frames.items()):
        audit.consume_frame(fid, np.arange(len(ids), dtype=np.int32), rows,
                            (poses or {}).get(fid, pose(x=fid/100)),
                            geometry_id="g")
    return audit.finish(), tracks


class AnomalyMathTests(unittest.TestCase):
    def simple(self):
        ids = np.full(60, 7, np.int32)
        frames = {fid:[spec("T1", fid, 1, range(25)), spec("T2", fid, 2, range(25, 50))]
                  for fid in (0, 5, 10)}
        return ids, frames

    def test_same_id_separation_at_exact_25_points_and_two_views(self):
        result, _ = run_case(*self.simple())
        pair = result["candidate_pairs"][0]
        self.assertEqual(pair["status"], "POTENTIAL_OVERMERGE")
        self.assertEqual(pair["joint_frame_count"], 3)
        self.assertEqual(pair["independent_joint_view_groups"], 2)
        self.assertTrue(all(e["target_support_points"] == [25, 25] for e in pair["joint_observations"]))

    def test_individually_stable_but_shared_views_static_is_insufficient(self):
        ids, _ = self.simple()
        frames = {fid:[] for fid in (0, 5, 10, 15)}
        for fid in (0, 5, 10):
            frames[fid].append(spec("T1", fid, 1, range(25)))
        for fid in (0, 5, 15):
            frames[fid].append(spec("T2", fid, 2, range(25, 50)))
        result, _ = run_case(ids, frames, {0:pose(), 5:pose(), 10:pose(.2), 15:pose(.3)})
        self.assertEqual(result["candidate_pairs"][0]["status"], "INSUFFICIENT_INDEPENDENT_VIEWS")
        self.assertEqual(result["candidate_pairs"][0]["independent_joint_view_groups"], 1)

    def test_separate_frames_do_not_prove_joint_visibility(self):
        ids, _ = self.simple()
        frames = {fid:[spec("T1", fid, 1, range(25))] for fid in (0, 5, 10)}
        frames.update({fid:[spec("T2", fid, 2, range(25, 50))] for fid in (15, 20, 25)})
        result, _ = run_case(ids, frames)
        self.assertEqual(result["candidate_pairs"], [])
        self.assertEqual(result["pair_checks"], [])

    def test_different_old_ids_are_not_overmerge(self):
        ids, frames = self.simple()
        ids[25:50] = 8
        result, _ = run_case(ids, frames)
        self.assertEqual(result["candidate_pairs"], [])

    def test_one_side_24_target_points_is_unknown_not_candidate(self):
        ids, frames = self.simple()
        ids[49] = 8
        result, _ = run_case(ids, frames)
        self.assertEqual(result["candidate_pairs"], [])
        self.assertEqual(len(result["pair_checks"]), 3)
        self.assertTrue(all(e["state"] == "INSUFFICIENT_TARGET_SUPPORT" for e in result["pair_checks"]))

    def test_no_extra_ratio_gate_even_at_five_percent(self):
        ids = np.full(1000, -1, np.int32)
        ids[:25] = ids[500:525] = 7
        frames = {fid:[spec("T1", fid, 1, range(500)), spec("T2", fid, 2, range(500, 1000))]
                  for fid in (0, 5, 10)}
        result, _ = run_case(ids, frames)
        pair = result["candidate_pairs"][0]
        self.assertEqual(pair["status"], "POTENTIAL_OVERMERGE")
        self.assertEqual(pair["joint_observations"][0]["target_support_fractions"], [.05, .05])
        self.assertEqual(result["observation_coverage"][0]["unassigned_points"], 475)

    def test_unassigned_and_zero_ids_are_recorded_but_never_targets(self):
        ids, frames = self.simple()
        ids[:25] = -1
        ids[25:] = 0
        result, _ = run_case(ids, frames)
        self.assertEqual(result["candidate_pairs"], [])
        self.assertEqual(result["observation_coverage"][0]["instance_coverage"][0]["instance_id"], -1)

    def test_union_coverage_counts_points_once_across_repeated_frames(self):
        result, _ = run_case(*self.simple())
        self.assertEqual(result["track_coverage"][0]["unique_support_points"], 25)
        self.assertEqual(result["track_coverage"][0]["instance_coverage"][0]["observation_count"], 3)
        self.assertEqual(result["track_coverage"][0]["instance_coverage"][0]["fraction_of_instance_surface"], 25/60)

    def test_independent_python_set_reference_for_joint_counts(self):
        rng = np.random.default_rng(314)
        for _ in range(20):
            ids = rng.choice([-1, 2, 7], 200).astype(np.int32)
            a = np.sort(rng.choice(100, 70, replace=False))
            b = np.sort(rng.choice(np.arange(100, 200), 70, replace=False))
            frames = {fid:[spec("A", fid, 1, a), spec("B", fid, 2, b)] for fid in (0, 5, 10)}
            result, _ = run_case(ids, frames)
            for event in result["pair_checks"]:
                iid = event["target_instance_id"]
                target = {i for i, value in enumerate(ids) if value == iid}
                expected = [len(set(a)&target), len(set(b)&target)]
                self.assertEqual(event["target_support_points"], expected)
                self.assertEqual(event["state"] == "JOINT_SEPARATION_SUPPORT", min(expected)>=25)

    def test_same_raw_mask_or_overlapping_support_is_rejected(self):
        ids, frames = self.simple()
        audit, _ = engine(ids, frames)
        bad = [frames[0][0], spec("T2", 0, 1, range(25, 50))]
        with self.assertRaisesRegex(ValueError, "Same original mask"):
            audit.consume_frame(0, np.arange(60), bad, pose(), geometry_id="g")
        frames = {fid:[spec("T1", fid, 1, range(25)), spec("T2", fid, 2, range(20, 45))]
                  for fid in (0, 5, 10)}
        with self.assertRaisesRegex(ValueError, "overlapping surface"):
            run_case(ids, frames)

    def test_deferred_or_unregistered_member_is_rejected(self):
        ids, frames = self.simple()
        audit, _ = engine(ids, frames)
        bad = [spec("T1", 0, 1, range(25))]
        bad[0] = ("T1", "pending-key", 1, bad[0][3])
        with self.assertRaisesRegex(ValueError, "deferred"):
            audit.consume_frame(0, np.arange(60), bad, pose(), geometry_id="g")

    def test_geometry_future_order_and_lost_members_are_rejected(self):
        ids, frames = self.simple()
        audit, _ = engine(ids, frames)
        with self.assertRaisesRegex(ValueError, "Geometry"):
            audit.consume_frame(0, np.arange(60), frames[0], pose(), geometry_id="wrong")
        audit.consume_frame(0, np.arange(60), frames[0], pose(), geometry_id="g")
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            audit.consume_frame(0, np.arange(60), frames[0], pose(), geometry_id="g")
        with self.assertRaisesRegex(ValueError, "lost"):
            audit.finish()

    def test_joint_view_axis_not_roll_and_fixed_representatives(self):
        groups, settings = [], AnomalySettings()
        self.assertEqual(add_joint_view(groups, 0, ["a", "b"], pose(), settings), 1)
        self.assertEqual(add_joint_view(groups, 5, ["c", "d"], pose(roll=60), settings), 1)
        self.assertEqual(add_joint_view(groups, 10, ["e", "f"], pose(yaw=10), settings), 2)
        groups = []
        for fid, x in [(0, 0), (5, .09), (10, .18)]:
            add_joint_view(groups, fid, [str(fid)], pose(x), settings)
        self.assertEqual([g["representative_frame_id"] for g in groups], [0, 10])

    def test_empty_stable_registry_yields_empty_review(self):
        audit = StableMapAudit(np.full(10, -1), [], "g", "hash")
        audit.consume_frame(0, np.empty(0, np.int32), [], pose(), geometry_id="g")
        result = audit.finish()
        self.assertEqual(result["candidate_pairs"], [])
        self.assertEqual(result["track_coverage"], [])


def full_fixture(root):
    mapping, _ = fixture(root, per_frame=False)
    raw = root/"original"
    for kind, directory in [("frame", raw/"masks"), ("depth", raw/"scene/results")]:
        (directory/f"{kind}000010.png").write_bytes((directory/f"{kind}000000.png").read_bytes())
    metadata_path = raw/"masks/frames.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    metadata.append(dict(metadata[0], frame=10))
    metadata_path.write_text("".join(json.dumps(row)+"\n" for row in metadata))
    association = mapping/"association/observations.jsonl"
    records = [json.loads(line) for line in association.read_text().splitlines()]
    records += [dict(row, frame_id=10, observation_id=row["observation_id"].replace("f000000", "f000010"))
                for row in records if row["frame_id"]==0]
    association.write_text("".join(json.dumps(row)+"\n" for row in records))
    poses = np.tile(np.eye(4), (11, 1, 1))
    poses[5, 0, 3], poses[10, 0, 3] = .05, .1
    np.savetxt(raw/"scene/traj.txt", poses.reshape(11, 16))
    for p in [raw/"config.json", mapping/"config.json"]:
        d = json.loads(p.read_text())
        d["frame_selection"]["stop_exclusive"] = 11
        dump(p, d)
    d = json.loads((mapping/"mapping_plan.json").read_text())
    d.update(raw_cutoff=10, target_processed_frames=3)
    dump(mapping/"mapping_plan.json", d)
    d = json.loads((mapping/"mapping_complete.json").read_text())
    d.update(frame_ids=[0, 5, 10], frame_count=3, last_frame_id=10)
    dump(mapping/"mapping_complete.json", d)
    xyz = np.column_stack((np.r_[np.linspace(.3, .4, 25), np.linspace(4.1, 4.2, 25), np.linspace(.6, .7, 5)],
                           np.full(55, .2), np.ones(55))).astype(np.float32)
    ids = np.r_[np.full(50, 7, np.int32), np.full(5, -1, np.int32)]
    native = mapping/"baseline/native/instance_surface.npz"
    np.savez_compressed(native, xyz_m=xyz, instance_id=ids)
    certify_geometry(mapping, xyz)
    final = mapping/"baseline/final/instance_surface.npz"
    final.parent.mkdir()
    np.savez_compressed(final, xyz_m=xyz, instance_id=ids)
    evidence = mapping/"baseline/surface_evidence.npz"
    np.savez_compressed(evidence, xyz_m=xyz, state=np.r_[np.full(50, 2, np.uint8), np.zeros(5, np.uint8)])
    graph = mapping/"geometry/review_surface_graph_n02_p08.npz"
    np.savez_compressed(graph, fixture_no_graph_traversal=np.array([1]))
    candidate = mapping.parent/"candidates"
    catalog = candidate/"catalog/test"
    catalog.mkdir(parents=True)
    core = np.arange(50, 55, dtype=np.int32)
    context = np.r_[np.arange(5, dtype=np.int32), core]
    membership = catalog/"membership.npz"
    np.savez_compressed(membership, core_offsets=np.array([0, 5]), core_point_index=core,
                        context_offsets=np.array([0, 10]), context_point_index=context)
    digest = hashlib.sha256(core.astype("<i4").tobytes()).hexdigest()
    alias = dict(id="ROI-U0001", group="U")
    component = dict(uid="test-roi", component_index=0, core_points=5, context_points=10,
                     core_sha256=digest, counts=dict(CONFLICT=0, U=5, T=0), aliases=[alias])
    manifest = dict(scene="test", checkpoint=3, history_frames=3, no_GT_used=True,
                    current_checkpoint_repaired=False, repair_applied=False, previous_committed_checkpoint=0,
                    scoring_history_frame_ids=[0, 5, 10], candidate_search_frame_ids=[0, 5, 10],
                    unique_components=[component], entries=[dict(alias, uid="test-roi", component_index=0,
                        core_sha256=digest, views=[dict(frame=0)])], membership_sha256=sha256_file(membership))
    for field, path in [("surface", final), ("evidence", evidence), ("graph", graph), ("camera_config", mapping/"config.json")]:
        manifest[field+"_file"], manifest[field+"_sha256"] = str(path), sha256_file(path)
    dump(catalog/"candidate_manifest.json", manifest)
    dump(catalog/"verification.json", dict(status="PASS", scene="test", no_GT_used=True,
         no_map_change=True, unique_ROIs=1, baseline_sha256=sha256_file(final)))
    dump(candidate/"mapping_input_freeze.json", dict(status="FROZEN", checkpoint=3, map_is_this_branch=True,
         source_hashes={str(p):sha256_file(p) for p in [native, final, evidence, graph, mapping/"config.json"]}))
    with redirect_stdout(io.StringIO()):
        build_cache(mapping, root/"cache", mode="offline", checkpoint=3)
        build_tracks(root/"cache", root/"tracks")
    return mapping, catalog, xyz


class AnomalyArtifactTests(unittest.TestCase):
    def test_full_pipeline_replay_determinism_roi_scales_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="two-audit-anomaly-") as temp:
            root = Path(temp)
            mapping, catalog, xyz = full_fixture(root/"inputs")
            input_tracks = root/"inputs/tracks"
            before = sha256_file(mapping/"baseline/native/instance_surface.npz")
            with redirect_stdout(io.StringIO()):
                first = build_anomalies(input_tracks, root/"first")
                second = build_anomalies(input_tracks, root/"second")
                check = validate_anomalies(root/"first", verify_inputs=True)
            self.assertEqual(check["status"], "PASS")
            self.assertEqual(first["totals"]["potential_overmerge_pairs"], 1)
            self.assertEqual(first["totals"]["potential_overmerge_instances"], 1)
            pair = json.loads((root/"first/candidate_pairs.json").read_text())[0]
            self.assertEqual(pair["roi_comparison"]["full_mask_footprint"]["relation"], "CORE_OVERLAP")
            self.assertEqual(pair["roi_comparison"]["target_instance_footprint"]["relation"], "CONTEXT_ONLY")
            self.assertFalse(pair["repair_triggered"])
            self.assertEqual(first["run_id"], second["run_id"])
            for name in ["candidate_pairs.json", "same_frame_pair_checks.jsonl.gz", "instance_anomalies.json"]:
                self.assertEqual(sha256_file(root/"first"/name), sha256_file(root/"second"/name))
            self.assertEqual(before, sha256_file(mapping/"baseline/native/instance_surface.npz"))
            with self.assertRaises(FileExistsError):
                build_anomalies(input_tracks, root/"first")
            summary = root/"first/pair_summary.csv"
            summary.write_text(summary.read_text()+"corruption\n")
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                validate_anomalies(root/"first")

    def test_input_guards_wrong_geometry_changed_labels_future_roi_and_bad_membership(self):
        with tempfile.TemporaryDirectory(prefix="two-audit-inputs-") as temp:
            root = Path(temp)
            mapping, catalog, xyz = full_fixture(root)
            cache = ObservationCache(root/"cache")
            inputs = AuditMapInputs(cache)
            for points, expected in [(np.array([50], np.int32), "CORE_OVERLAP"),
                                     (np.array([0], np.int32), "CONTEXT_ONLY"),
                                     (np.array([25], np.int32), "NO_GEOMETRIC_OVERLAP")]:
                self.assertEqual(inputs.roi_overlap(points)["relation"], expected)
            wrong = root/"reordered.npz"
            np.savez_compressed(wrong, xyz_m=xyz[::-1], instance_id=np.full(55, 7, np.int32))
            with self.assertRaisesRegex(ValueError, "Geometry changed"):
                read_ids(wrong, cache)
            mp = catalog/"candidate_manifest.json"
            original = mp.read_text()
            m = json.loads(original)
            m["entries"][0]["views"][0]["frame"] = 15
            dump(mp, m)
            with self.assertRaisesRegex(ValueError, "future/unobserved"):
                AuditMapInputs(cache)
            mp.write_text(original)
            membership = catalog/"membership.npz"
            original_membership = membership.read_bytes()
            membership.write_bytes(b"not the frozen membership")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                AuditMapInputs(cache)
            membership.write_bytes(original_membership)
            native = mapping/"baseline/native/instance_surface.npz"
            np.savez_compressed(native, xyz_m=xyz, instance_id=np.full(55, 8, np.int32))
            with self.assertRaisesRegex(ValueError, "pre-repair snapshot"):
                AuditMapInputs(cache)


if __name__ == "__main__":
    unittest.main()

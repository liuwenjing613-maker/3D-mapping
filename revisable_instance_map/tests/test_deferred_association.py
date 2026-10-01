"""Behavioral contracts for isolation, independent witnesses, time and publication."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.identity_evidence import FrameIdentityEvidence, ObservationSupport
from revisable_instance_map.assignment_ledger import ACCEPTED, PENDING_BIND, PENDING_BIRTH, UNPROJECTABLE
from revisable_instance_map.deferred_association import DeferredAssociator
from revisable_instance_map.association_review import EvidenceView, reliable, score_complete
from revisable_instance_map.pending_observations import PendingPacket
from revisable_instance_map.frame_io import Frame, Camera
from revisable_instance_map.observations import extract_frame_observations
from revisable_instance_map.pending_publication import protect_publication, enforce_pending_protection
from revisable_instance_map.surface_evidence import reduce_surface_votes, UNOBSERVED


def make_frame(frame_id, translation=0., regions=((-0.15, .15, 1),), invalid=False):
    camera = Camera(40, 20, 40., 40., 20., 10., 1000.)
    rows, cols = np.indices((20, 40))
    world_x = (cols-camera.cx)/camera.fx + translation
    world_y = (rows-camera.cy)/camera.fy
    mask = np.zeros((20, 40), np.uint16)
    for lo, hi, label in regions:
        mask[(world_x >= lo) & (world_x < hi) & (world_y >= -.1) & (world_y < .1)] = label
    pose = np.eye(4)
    pose[0, 3] = translation
    depth = np.zeros((20, 40), np.float32) if invalid else np.ones((20, 40), np.float32)
    rgb = np.zeros((20, 40, 3), np.uint8)
    for array in (rgb, depth, mask, pose):
        array.setflags(write=False)
    return Frame(frame_id, rgb, depth, mask, pose, camera)


def observations(frame):
    return extract_frame_observations(frame, "test", "room", "0"*64)


def setup_model(frames, mode="B2", parameters=None):
    lookup = {f.frame_id: f for f in frames}
    return DeferredAssociator(lookup.__getitem__, mode, parameters)


def seed(model, frames):
    old_mode = model.mode
    model.mode = "B0"
    for frame in frames:
        model.process_frame(frame, observations(frame))
    model.mode = old_mode


class DeferredAssociationTest(unittest.TestCase):
    def test_b0_equals_a2_decisions_counts_and_ids(self):
        frames = [make_frame(f, f*.005) for f in (0, 5, 10, 15)]
        model = setup_model(frames, "B0")
        original = OnlineVoxelAssociator(association_mode="probabilistic", valid_first_sampling=False,
                                       allow_multiple_observations_per_instance_per_frame=True)
        for frame in frames:
            obs = observations(frame)
            self.assertEqual(original.process_frame(frame, obs), model.process_frame(frame, obs))
        for key, value in original.identity_evidence.export_arrays().items():
            np.testing.assert_array_equal(value, model.engine.identity_evidence.export_arrays()[key])
        model.validate()

    def test_pending_keeps_real_voxels_without_formal_votes_or_ids(self):
        frame = make_frame(0)
        model = setup_model([frame])
        model.process_frame(frame, observations(frame))
        key = observations(frame)[0].observation_id
        self.assertTrue(model.ledger.raw_support_store[key].voxels)
        self.assertEqual(model.ledger.assignment_store[key].status, PENDING_BIRTH)
        self.assertFalse(model.engine.identity_evidence.voxel_counts)
        self.assertFalse(model.engine.voxel_to_instances)
        self.assertEqual(model.engine.next_instance_id, 1)
        model.validate()

    def test_repeat_views_reuse_packet_but_do_not_confirm_birth(self):
        frames = [make_frame(f) for f in (0, 5, 10)]
        model = setup_model(frames)
        for frame in frames:
            model.process_frame(frame, observations(frame))
        self.assertEqual(len(model.packets), 1)
        self.assertFalse(model.engine.instances)
        self.assertEqual(len(model.ledger.raw_support_store), 3)

    def test_multiple_current_masks_near_one_old_pending_packet(self):
        frames = [make_frame(0, regions=((-0.15, 0., 1),)),
                  make_frame(5, regions=((-0.15, 0., 1), (0., .15, 2)))]
        model = setup_model(frames)
        for frame in frames:
            model.process_frame(frame, observations(frame))
        self.assertEqual(len(model.ledger.raw_support_store), 3)
        self.assertTrue(model.validate())

    def test_duplicate_frame_and_observation_leave_state_unchanged(self):
        frame = make_frame(0)
        model = setup_model([frame])
        model.process_frame(frame, observations(frame))
        baseline = model.assignment_rows()
        for obs in (observations(frame), observations(frame)*2):
            with self.assertRaises(ValueError):
                model.process_frame(frame, obs)
            self.assertEqual(model.assignment_rows(), baseline)
            self.assertEqual(model.map_version, 1)

    def test_same_frame_fragments_cannot_supply_multiple_view_groups(self):
        frame = make_frame(0, regions=((-0.15, 0., 1), (0., .15, 2)))
        model = setup_model([frame])
        model.process_frame(frame, observations(frame))
        self.assertFalse(model.engine.instances)
        records = list(model.ledger.raw_support_store.values())
        groups = model.verifier.information_groups(records)
        self.assertEqual(len(set(groups.values())), 1)

    def test_single_frame_support_with_high_q_is_not_reliable(self):
        frames = [make_frame(f) for f in (0, 5)]
        model = setup_model(frames)
        seed(model, frames[:1])
        scores = score_complete(model.engine, frames[1], observations(frames[1])[0],
                                model.engine._sample_observation_voxels(frames[1], observations(frames[1])[0]), model.parameters)
        self.assertGreater(scores["candidate_relative_weights"][0]["weight"], .98)
        self.assertEqual(reliable(model.engine, scores, model.parameters)[1], "single_frame_local_support")

    def test_visibility_fallback_is_deferred(self):
        frames = [make_frame(f) for f in (0, 5, 10)]
        model = setup_model(frames)
        seed(model, frames[:2])
        model.engine._visible_overlap = lambda *args: (None, 0)
        arrivals = model.process_frame(frames[2], observations(frames[2]))
        self.assertEqual(arrivals[0]["decision"], "visibility_unverified")
        self.assertIsNone(arrivals[0]["instance_id"])

    def test_candidate_budget_requires_expansion_or_pending(self):
        frames = [make_frame(0), make_frame(5)]
        model = setup_model(frames)
        voxels = model.engine._sample_observation_voxels(frames[0], observations(frames[0])[0])
        for instance in range(1, 10):
            for f in (0, 1):
                model.engine._commit(instance, replace(observations(frames[0])[0], observation_id=f"{instance}/{f}", frame_id=f), voxels)
        obs = observations(frames[1])[0]
        truncated = score_complete(model.engine, frames[1], obs, voxels, {**model.parameters, "expand_candidates": False})
        expanded = score_complete(model.engine, frames[1], obs, voxels, model.parameters)
        self.assertEqual(truncated["candidate_count_scored"], 8)
        self.assertFalse(truncated["candidate_complete"])
        self.assertEqual(expanded["candidate_count_scored"], 9)
        self.assertTrue(expanded["candidate_complete"])

    def test_cold_start_activates_only_after_two_distinct_views(self):
        frames = [make_frame(0), make_frame(5, .025)]
        model = setup_model(frames)
        model.process_frame(frames[0], observations(frames[0]))
        self.assertFalse(model.engine.instances)
        model.process_frame(frames[1], observations(frames[1]))
        self.assertEqual(len(model.engine.instances), 1)
        self.assertEqual({a.status for a in model.ledger.assignment_store.values()}, {ACCEPTED})
        self.assertEqual({e.frame_id for e in model.engine.observation_support.values()}, {0, 5})
        model.validate()

    def test_transient_mask_remains_pending_and_unprojectable_has_no_id(self):
        frames = [make_frame(0), make_frame(5, invalid=True)]
        model = setup_model(frames)
        for frame in frames:
            model.process_frame(frame, observations(frame))
        self.assertEqual(model.ledger.assignment_store[observations(frames[1])[0].observation_id].status, UNPROJECTABLE)
        self.assertFalse(model.engine.instances)
        self.assertEqual(len(model.ledger.raw_support_store), 2)

    def _binding_case(self, mode):
        frames = [make_frame(0), make_frame(5), make_frame(10), make_frame(15, .025)]
        model = setup_model(frames, mode)
        seed(model, frames[:2])
        target = observations(frames[2])[0].observation_id
        real_score = score_complete

        def risky(engine, frame, observation, voxels, parameters, excluded_ids=()):
            score = real_score(engine, frame, observation, voxels, parameters, excluded_ids)
            if observation.observation_id == target:
                best = {**score["candidates"][0], "score": .6}
                score["candidates"] = [best, {**best, "instance_id": 2, "score": .59}]
            return score

        with patch("revisable_instance_map.deferred_association.score_complete", risky):
            model.process_frame(frames[2], observations(frames[2]))
            before = model.assignment_rows(asof=10)
            model.process_frame(frames[3], observations(frames[3]))
        return model, target, before, frames

    def test_new_independent_anchor_releases_source_frame_not_decision_frame(self):
        model, key, _, _ = self._binding_case("B2")
        assignment = model.ledger.assignment_store[key]
        self.assertEqual(assignment.status, ACCEPTED)
        self.assertEqual(assignment.last_decision_frame_id, 15)
        self.assertEqual(model.engine.observation_support[key].frame_id, 10)
        self.assertEqual(len({w["group_id"] for w in assignment.witnesses}), 2)
        model.validate()

    def test_isolation_ablation_does_not_release_ambiguous_old_binding(self):
        model, key, _, _ = self._binding_case("B1")
        self.assertEqual(model.ledger.assignment_store[key].status, PENDING_BIND)
        self.assertNotIn(key, model.engine.observation_support)

    def test_same_witnesses_do_not_generate_more_votes(self):
        model, key, _, _ = self._binding_case("B2")
        count = len(model.engine.observation_support)
        packet = model.packets[model.ledger.assignment_store[key].packet_id]
        updates, *_ = model._review(packet, model.ledger.raw_support_store, model.ledger.assignment_store,
                                    set(), model.map_version, 15)
        self.assertFalse(updates)
        self.assertEqual(len(model.engine.observation_support), count)

    def test_excluding_packet_preserves_same_frame_independent_vote(self):
        evidence = FrameIdentityEvidence()
        evidence.add(ObservationSupport("q", 0, "0"*64, 1, ((0,0,0),)))
        evidence.add(ObservationSupport("outside", 0, "0"*64, 1, ((0,0,0),)))
        evidence.add(ObservationSupport("q2", 1, "0"*64, 1, ((0,0,0),)))
        view = EvidenceView(evidence, ("q", "q2"))
        self.assertEqual(view.voxel_counts[(0,0,0)], {1:1})
        self.assertEqual(view.voxel_totals[(0,0,0)], 1)
        self.assertEqual(len(view.instance_frame_references[1]), 1)
        self.assertEqual(evidence.voxel_counts[(0,0,0)], {1:2})

    def test_packet_members_cannot_self_certify_identity(self):
        frames = [make_frame(0), make_frame(5, .025), make_frame(10, .05)]
        model = setup_model(frames)
        for frame in frames[:2]:
            model.process_frame(frame, observations(frame))
        packet = next(iter(model.packets.values()))
        excluded = list(packet.observation_ids)
        obs = observations(frames[2])[0]
        scores = score_complete(model.engine, frames[2], obs,
                               model.engine._sample_observation_voxels(frames[2], obs), model.parameters, excluded)
        self.assertFalse(scores["candidates"])
        self.assertTrue(model.engine.instances)

    def test_nearby_separate_objects_do_not_merge_on_voxel_proximity(self):
        regions = ((-.15, -.025, 1), (.025, .15, 2))
        frames = [make_frame(0, regions=regions), make_frame(5, .025, regions)]
        model = setup_model(frames)
        for frame in frames:
            model.process_frame(frame, observations(frame))
        self.assertEqual(len(model.engine.instances), 2)
        ids = [model.ledger.assignment_store[o.observation_id].persistent_instance_id for o in observations(frames[0])]
        self.assertNotEqual(ids[0], ids[1])

    def test_same_frame_fragments_can_bind_one_existing_id(self):
        frames = [make_frame(0), make_frame(5), make_frame(10, regions=((-0.15,0.,1),(0.,.15,2)))]
        model = setup_model(frames)
        seed(model, frames[:2])
        arrivals = model.process_frame(frames[2], observations(frames[2]))
        self.assertEqual([r["instance_id"] for r in arrivals], [1,1])
        model.validate()

    def test_no_common_visible_region_is_unknown(self):
        frames = [make_frame(0), make_frame(5, 10.)]
        # Second frame retains an image mask but its camera is far away.
        frames[1] = replace(make_frame(5), camera_to_world=frames[1].camera_to_world)
        model = setup_model(frames)
        for frame in frames:
            model.verifier.receive(frame)
        raw = [model._raw(f, observations(f)[0]) for f in frames]
        self.assertEqual(model.verifier.compare(*raw)["status"], "UNKNOWN")

    def test_mixed_raw_mask_is_not_released_wholesale(self):
        frames = [make_frame(0), make_frame(5, .025, ((-.15,0.,1),(0.,.15,2)))]
        model = setup_model(frames)
        for frame in frames:
            model.verifier.receive(frame)
        raw = [model._raw(frames[0], observations(frames[0])[0]), model._raw(frames[1], observations(frames[1])[0])]
        self.assertEqual(model.verifier.compare(*raw)["status"], "MULTI_REGION_CONTRADICTION")

    def test_seven_good_and_three_mixed_members_are_reviewed_individually(self):
        split = ((-.15, 0., 1), (0., .15, 2))
        frames = [make_frame(0, regions=split), make_frame(5, regions=split)]
        frames += [make_frame(f, regions=((-.15, 0., 1),)) for f in range(10, 45, 5)]
        frames += [make_frame(f) for f in (45, 50, 55)]
        frames += [make_frame(60, .025, split)]
        model = setup_model(frames, "B1")
        seed(model, frames[:2])
        keys = [observations(f)[0].observation_id for f in frames[2:-1]]
        actual_score = score_complete
        def ambiguous(engine, frame, observation, voxels, parameters, excluded_ids=()):
            scores = actual_score(engine, frame, observation, voxels, parameters, excluded_ids)
            if observation.observation_id in keys and scores["candidates"]:
                best = {**scores["candidates"][0], "score": .6}
                scores["candidates"] = [best, {**best, "instance_id": 99, "score": .59}]
            return scores
        with patch("revisable_instance_map.deferred_association.score_complete", ambiguous):
            for frame in frames[2:]:
                model.process_frame(frame, observations(frame))
            # Deliberately coarse audit packet: grouping must never authorize a batch bind.
            packet = PendingPacket("Q-audit", tuple(keys), model.processing_step)
            model.mode = "B2"
            updates, _, _, _ = model._review(packet, model.ledger.raw_support_store,
                model.ledger.assignment_store, {o.observation_id for o in observations(frames[-1])},
                model.map_version, 60)
        reviewed = {k: updates.get(k, model.ledger.assignment_store[k]) for k in keys}
        self.assertEqual(sum(reviewed[k].status == ACCEPTED for k in keys), 7)
        self.assertTrue(all(reviewed[k].status == PENDING_BIND for k in keys[7:]))
        self.assertFalse(any(k in model.engine.observation_support for k in keys))

    def test_three_accepted_votes_plus_two_unknowns_cannot_publish_at_one(self):
        evidence = reduce_surface_votes(np.array([1]), np.array([3]), 1, 2)
        result = protect_publication(evidence, np.array([2]))
        self.assertAlmostEqual(result["worst_case_vote_share"][0], .6)
        self.assertEqual(result["instance_id"][0], -1)
        self.assertEqual(evidence["confidence"][0], 1.)

    def test_pending_only_differs_from_unobserved_and_blocks_diffusion(self):
        evidence = reduce_surface_votes(np.array([],np.int64), np.array([],np.int32), 2, 2)
        result = protect_publication(evidence, np.array([1,0]))
        self.assertEqual(result["publication_reason"].tolist(), [4,UNOBSERVED])
        inferred = enforce_pending_protection(result["instance_id"], np.array([7,7]), result["diffusion_protected_mask"])
        self.assertEqual(inferred.tolist(), [-1,7])

    def test_no_pending_publication_is_identical_to_original_p0(self):
        evidence = reduce_surface_votes(np.array([1,3]), np.array([3,1]), 2, 2)
        result = protect_publication(evidence, np.zeros(2,np.int32))
        np.testing.assert_array_equal(result["instance_id"], np.where(evidence["state"]==2, evidence["top1_instance_id"],-1))

    def test_late_release_is_invisible_in_old_prefix(self):
        model, key, before, _ = self._binding_case("B2")
        self.assertEqual(before, model.assignment_rows(asof=10))
        row = next(r for r in before if r["observation_id"] == key)
        self.assertIsNone(row["instance_id"])
        self.assertIsNotNone(next(r for r in model.assignment_rows(asof=15) if r["observation_id"]==key)["instance_id"])

    def test_checkpoint_resume_preserves_pending_and_highwaters(self):
        frames = [make_frame(0), make_frame(5), make_frame(10,.025)]
        model = setup_model(frames)
        for frame in frames[:2]:
            model.process_frame(frame, observations(frame))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.npz"
            model.save_checkpoint(path)
            resumed = DeferredAssociator.from_checkpoint(path, model.verifier.loader)
        self.assertEqual(json.loads(json.dumps(model.assignment_rows())),
                         json.loads(json.dumps(resumed.assignment_rows())))
        self.assertEqual(model.assignment_rows(), resumed.assignment_rows())
        self.assertEqual(model.packets, resumed.packets)
        self.assertEqual(model.next_packet_id, resumed.next_packet_id)
        model.process_frame(frames[2], observations(frames[2]))
        resumed.process_frame(frames[2], observations(frames[2]))
        self.assertEqual(model.assignment_rows(), resumed.assignment_rows())
        self.assertEqual(model.ledger.events, resumed.ledger.events)

    def test_import_a2_retains_retired_id_highwater_and_revision_history(self):
        frames = [make_frame(0), make_frame(5)]
        original = setup_model(frames, "B0")
        for frame in frames:
            original.process_frame(frame, observations(frame))
        key = observations(frames[0])[0].observation_id
        original.engine.reassign_observations({key: 9})
        original.engine.reassign_observations({key: 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"a2.npz"
            original.engine.save_checkpoint(path)
            imported = DeferredAssociator.from_a2_checkpoint(path, original.verifier.loader,
                [o for frame in frames for o in observations(frame)])
        self.assertEqual(imported.engine.next_instance_id, 10)
        self.assertEqual(imported.imported_a2_revision_events, original.engine.revision_events)
        self.assertEqual(imported.ledger.assignment_store[key].assignment_version, 2)
        self.assertFalse(imported.assignment_rows(asof=0))
        self.assertTrue(imported.validate())

    def test_import_a2_counts_empty_processed_frames(self):
        frames = [make_frame(0,regions=()), make_frame(5), make_frame(10,regions=())]
        original = setup_model(frames,"B0")
        for frame in frames:
            original.process_frame(frame, observations(frame))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"a2.npz"
            original.engine.save_checkpoint(path)
            imported = DeferredAssociator.from_a2_checkpoint(path, original.verifier.loader,
                [o for frame in frames for o in observations(frame)])
        self.assertEqual(imported.processing_step,3)

    def test_full_source_rebuild_equals_incremental_active_and_pending(self):
        frames = [make_frame(0), make_frame(5,.025), make_frame(10,invalid=True)]
        model = setup_model(frames)
        for frame in frames:
            model.process_frame(frame, observations(frame))
        self.assertTrue(model.validate())

    def test_fault_after_activation_or_ledger_rolls_back_whole_batch(self):
        for phase in ("after_activation", "after_ledger", "after_spatial_index", "after_bookkeeping"):
            frames = [make_frame(0), make_frame(5,.025)]
            model = setup_model(frames)
            model.process_frame(frames[0], observations(frames[0]))
            baseline = model.assignment_rows()
            events = list(model.ledger.events)
            def fail(stage):
                if stage == phase:
                    raise RuntimeError("injected failure")
            model.fault_injector = fail
            with self.assertRaises(RuntimeError):
                model.process_frame(frames[1], observations(frames[1]))
            self.assertEqual(model.assignment_rows(), baseline)
            self.assertEqual(model.ledger.events, events)
            self.assertFalse(model.engine.instances)
            self.assertEqual(model.map_version, 1)
            self.assertEqual(model.engine.next_instance_id, 1)
            self.assertEqual(model.processing_step, 1)
            self.assertEqual(len(model.arrivals), 1)
            self.assertTrue(model.validate())

    def test_dormant_packet_keeps_sources_and_can_wake(self):
        frames = [make_frame(0), make_frame(5, regions=()), make_frame(10, regions=()), make_frame(15)]
        model = setup_model(frames, parameters={"active_window_steps":1})
        for frame in frames[:3]:
            model.process_frame(frame, observations(frame))
        self.assertEqual(next(iter(model.packets.values())).lifecycle,"DORMANT")
        self.assertEqual(len(model.ledger.raw_support_store),1)
        model.process_frame(frames[3], observations(frames[3]))
        self.assertEqual(len(model.packets),1)
        self.assertEqual(next(iter(model.packets.values())).lifecycle,"ACTIVE")

    def test_observation_budget_carries_unreviewed_members_without_new_frames(self):
        frames = [make_frame(f) for f in (0,5,10)]
        frames += [make_frame(f, regions=()) for f in (15,20,25)]
        model = setup_model(frames, parameters={"observation_review_budget":1})
        for frame in frames:
            model.process_frame(frame, observations(frame))
        reviewed_later = {d["observation_id"] for d in model.review_log if d["decision_frame_id"]>=15}
        self.assertEqual(reviewed_later, {o.observation_id for f in frames[:3] for o in observations(f)})
        self.assertTrue(model.review_queue)
        self.assertFalse(model.engine.instances)

    def test_source_hash_detects_changed_intrinsics_with_same_pixel_arrays(self):
        frame = make_frame(0)
        model = setup_model([frame])
        original = model.verifier.source_hash(frame)
        changed = replace(frame, camera=replace(frame.camera, fx=80.))
        self.assertNotEqual(original, model.verifier.source_hash(changed))

    def test_persistent_wrong_hypothesis_does_not_invent_a_correct_id(self):
        frames = [make_frame(f) for f in (0,5,10,15)]
        model = setup_model(frames)
        seed(model, frames[:2])
        for frame in frames[2:]:
            model.process_frame(frame, observations(frame))
        self.assertEqual(set(model.engine.instances),{1})

    def test_accepted_observation_is_not_reassigned_by_probability_changes(self):
        model, key, _, frames = self._binding_case("B2")
        before = model.ledger.assignment_store[key]
        frame = make_frame(20)
        model.verifier.loader = {f.frame_id:f for f in [*frames,frame]}.__getitem__
        model.process_frame(frame, observations(frame))
        self.assertEqual(model.ledger.assignment_store[key], before)


if __name__ == "__main__":
    unittest.main()

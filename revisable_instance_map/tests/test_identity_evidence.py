"""P1-A invariants: frame votes, causality, revisions, and real ranking changes."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.association import OnlineVoxelAssociator, _NEIGHBORS
from revisable_instance_map.identity_evidence import FrameIdentityEvidence, ObservationSupport
from revisable_instance_map.observations import RawInstanceObservation


def obs(frame_id, local_id=1, key=None):
    return RawInstanceObservation(key or f"f{frame_id}/m{local_id}", frame_id, local_id, "0" * 64,
                                  1, 1, (0, 0, 1, 1), None, None, None)


def entry(key, frame, instance, voxels=((0, 0, 0),)):
    return ObservationSupport(key, frame, "0" * 64, instance, voxels)


def associator(mode="probabilistic", improved=False):
    result = OnlineVoxelAssociator(association_mode=mode, improved_association=improved,
        allow_multiple_observations_per_instance_per_frame=True, valid_first_sampling=False)
    result._sample_observation_voxels = lambda frame, observation: ((0, 0, 0),)
    result._visible_overlap = lambda frame, observation, support: (None, 0)
    return result


class IdentityEvidenceTest(unittest.TestCase):
    def test_empty_support_is_recorded_but_has_no_identity_vote_or_support_frame(self):
        ledger = FrameIdentityEvidence()
        ledger.add(entry("empty", 0, 1, ()))
        ledger.add(entry("valid", 1, 1))
        self.assertEqual(len(ledger.instance_frame_references[1]), 1)
        ledger.reassign({"empty": 2})
        self.assertNotIn(2, ledger.instance_frame_references)
        ledger.validate()

    def test_same_frame_fragments_and_duplicate_voxels_count_once(self):
        ledger = FrameIdentityEvidence()
        ledger.add(entry("a", 0, 1, ((0, 0, 0), (0, 0, 0))))
        ledger.add(entry("b", 0, 1))
        self.assertEqual(ledger.voxel_counts[(0, 0, 0)], {1: 1})
        self.assertEqual(ledger.frame_voxel_references[(0, (0, 0, 0), 1)], 2)
        self.assertEqual(ledger.entries["a"].voxels, ((0, 0, 0),))
        ledger.validate()

    def test_different_instances_compete_even_in_same_frame(self):
        ledger = FrameIdentityEvidence()
        ledger.add(entry("a", 0, 1))
        ledger.add(entry("b", 0, 2))
        self.assertEqual(ledger.probability((0, 0, 0), 1), .5)
        self.assertEqual(ledger.voxel_totals[(0, 0, 0)], 2)
        self.assertEqual(len(ledger.voxel_frame_references[(0, 0, 0)]), 1)

    def test_reassign_one_fragment_preserves_other_fragment_vote(self):
        ledger = FrameIdentityEvidence()
        ledger.add(entry("a", 0, 1))
        ledger.add(entry("b", 0, 1))
        ledger.reassign({"a": 2})
        self.assertEqual(ledger.voxel_counts[(0, 0, 0)], {1: 1, 2: 1})
        ledger.reassign({"b": 2})
        self.assertEqual(ledger.voxel_counts[(0, 0, 0)], {2: 1})
        self.assertNotIn((0, (0, 0, 0), 1), ledger.frame_voxel_references)
        ledger.validate()

    def test_round_trip_revision_restores_statistics_and_sources(self):
        ledger = FrameIdentityEvidence()
        for f in range(4):
            ledger.add(entry(str(f), f, 1))
        baseline = ledger.export_arrays()
        ledger.reassign({"0": 2})
        ledger.reassign({"0": 1})
        for key, value in baseline.items():
            np.testing.assert_array_equal(value, ledger.export_arrays()[key])
        self.assertEqual(ledger.entries["0"].assignment_version, 2)
        self.assertEqual(ledger.entries["0"].source_mask_sha256, "0" * 64)
        ledger.validate()

    def test_duplicate_and_invalid_transactions_leave_all_counts_unchanged(self):
        ledger = FrameIdentityEvidence()
        ledger.add(entry("a", 0, 1))
        for updates in ({"a": 2, "missing": 3}, {"a": 2.5}, {"a": True}, {"a": -1}):
            with self.assertRaises(ValueError):
                ledger.reassign(updates)
            self.assertEqual(ledger.entries["a"].instance_id, 1)
            ledger.validate()
        with self.assertRaises(ValueError):
            ledger.add(entry("a", 1, 2))
        self.assertEqual(ledger.voxel_counts[(0, 0, 0)], {1: 1})

    def test_probability_ranking_uses_competing_frame_evidence(self):
        binary, probability = associator("binary-ledger"), associator()
        for result in (binary, probability):
            for f in range(10):
                result._commit(2 if f < 9 else 1, obs(f), ((0, 0, 0),))
            # Many extra fragments do not give ID1 extra identity frame votes.
            for i in range(2, 15):
                result._commit(1, obs(9, i), ((0, 0, 0),))
        frame = SimpleNamespace(frame_id=10)
        self.assertEqual(binary._score_candidates(frame, obs(10), ((0, 0, 0),))[0]["instance_id"], 1)
        scores = probability._score_candidates(frame, obs(10), ((0, 0, 0),))
        self.assertEqual(scores[0]["instance_id"], 2)
        self.assertEqual(scores[0]["geometric_coverage"], 1)
        self.assertEqual(scores[0]["probabilistic_geometric_consistency"], .9)
        self.assertEqual(scores[1]["probabilistic_geometric_consistency"], .1)
        self.assertEqual(scores[1]["prior_support_frame_count"], 1)

    def test_neighborhood_reads_max_but_never_writes_neighbors(self):
        result = associator()
        result._commit(1, obs(0), ((1, 0, 0),))
        result._commit(2, obs(1), ((0, 0, 0),))
        scores = result._score_candidates(SimpleNamespace(frame_id=2), obs(2), ((0, 0, 0),))
        self.assertEqual({s["probabilistic_geometric_consistency"] for s in scores}, {1.0})
        self.assertEqual(len(_NEIGHBORS), 19)
        self.assertEqual(set(result.identity_evidence.voxel_counts), {(0, 0, 0), (1, 0, 0)})
        self.assertEqual(result.candidate_support_traces["f2/m1"][0][2][0][1], (1, 0, 0))

    def test_partial_coverage_zero_fills_unmatched_query_voxels(self):
        result = associator()
        result._commit(1, obs(0), ((0, 0, 0),))
        score = result._score_candidates(SimpleNamespace(frame_id=1), obs(1), ((0, 0, 0), (10, 10, 10)))[0]
        self.assertEqual(score["probabilistic_geometric_consistency"], .5)
        self.assertEqual(score["geometric_coverage"], .5)

    def test_visible_overlap_and_fallback_preserve_the_original_formula(self):
        result = associator()
        result._commit(1, obs(0), ((0, 0, 0),))
        result._commit(2, obs(1), ((0, 0, 0),))
        result._visible_overlap = lambda *args: (.8, 20)
        score = result._score_candidates(SimpleNamespace(frame_id=2), obs(2), ((0, 0, 0),))[0]
        self.assertEqual(score["score"], .65)
        result._visible_overlap = lambda *args: (None, 0)
        score = result._score_candidates(SimpleNamespace(frame_id=2), obs(2), ((0, 0, 0),))[0]
        self.assertEqual(score["score"], .5)
        self.assertTrue(score["visibility_fallback"])

    def test_binary_ledger_preserves_legacy_decisions(self):
        legacy, counted = associator("legacy"), associator("binary-ledger")
        for f in range(5):
            observations = (obs(f, 1), obs(f, 2))
            expected = legacy.process_frame(SimpleNamespace(frame_id=f), observations)
            actual = counted.process_frame(SimpleNamespace(frame_id=f), observations)
            for old, new in zip(expected, actual):
                for key, value in old.items():
                    if key == "candidates":
                        for old_candidate, new_candidate in zip(value, new[key]):
                            for field, item in old_candidate.items():
                                self.assertEqual(item, new_candidate[field])
                    else:
                        self.assertEqual(value, new[key])

    def test_frame_is_frozen_and_repeat_frame_is_rejected_without_votes(self):
        for improved in (False, True):
            result = associator(improved=improved)
            rows = result.process_frame(SimpleNamespace(frame_id=0), (obs(0, 1), obs(0, 2)))
            self.assertEqual([row["candidates"] for row in rows], [[], []])
            self.assertEqual([row["instance_id"] for row in rows], [1, 2])
            baseline = result.identity_evidence.export_arrays()
            with self.assertRaises(ValueError):
                result.process_frame(SimpleNamespace(frame_id=0), (obs(0, 1),))
            for key, value in baseline.items():
                np.testing.assert_array_equal(value, result.identity_evidence.export_arrays()[key])

    def test_bad_observation_ids_fail_before_any_partial_frame_commit(self):
        result = associator()
        observations = (obs(0, 1, "same"), obs(0, 2, "same"))
        with self.assertRaises(ValueError):
            result.process_frame(SimpleNamespace(frame_id=0), observations)
        self.assertEqual(result.instances, {})
        self.assertEqual(result.next_instance_id, 1)

    def test_noop_reassignment_does_not_create_an_event_or_a_vote(self):
        result = associator()
        result.process_frame(SimpleNamespace(frame_id=0), (obs(0),))
        result.reassign_observations({"f0/m1": 1})
        self.assertEqual(result.map_version, 1)
        self.assertEqual(result.revision_events, [])

    def test_retired_maximum_id_is_not_reused_after_save_and_restore(self):
        result = associator()
        result.process_frame(SimpleNamespace(frame_id=0), (obs(0), obs(0, 2)))
        result.reassign_observations({"f0/m2": 1})
        self.assertEqual(result.next_instance_id, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.npz"
            result.save_checkpoint(path)
            resumed = OnlineVoxelAssociator.from_checkpoint(path)
        self.assertEqual(resumed.next_instance_id, 3)
        self.assertEqual(resumed.revision_events, result.revision_events)
        self.assertEqual(resumed.identity_evidence.voxel_counts, result.identity_evidence.voxel_counts)
        resumed._sample_observation_voxels = lambda *args: ((99, 99, 99),)
        row = resumed.process_frame(SimpleNamespace(frame_id=1), (obs(1),))[0]
        self.assertEqual(row["instance_id"], 3)

    def test_saved_resume_and_independent_prefix_are_causal(self):
        full, prefix = associator(), associator()
        expected = []
        for f in range(3):
            observations = (obs(f),)
            expected.append(full.process_frame(SimpleNamespace(frame_id=f), observations))
            self.assertEqual(prefix.process_frame(SimpleNamespace(frame_id=f), observations), expected[-1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.npz"
            prefix.save_checkpoint(path)
            resumed = OnlineVoxelAssociator.from_checkpoint(path)
            resumed._sample_observation_voxels = full._sample_observation_voxels
            resumed._visible_overlap = full._visible_overlap
            for f in range(3, 6):
                frame = SimpleNamespace(frame_id=f)
                self.assertEqual(resumed.process_frame(frame, (obs(f),)), full.process_frame(frame, (obs(f),)))
        self.assertEqual(len(prefix.observation_support), 3)
        self.assertEqual(prefix.identity_evidence.voxel_counts[(0, 0, 0)], {1: 3})

    def test_local_probability_reversal_never_renames_previous_observations(self):
        result = associator()
        result._commit(1, obs(0), ((0, 0, 0),))
        result._commit(1, obs(1), ((0, 0, 0),))
        for f in range(2, 6):
            result._commit(2, obs(f), ((0, 0, 0),))
        self.assertGreater(result.identity_evidence.probability((0, 0, 0), 2), .5)
        self.assertEqual(result.observation_support["f0/m1"].instance_id, 1)
        self.assertEqual(result.revision_events, [])

    def test_consistently_wrong_single_hypothesis_does_not_magically_split(self):
        result = associator()
        for f in range(12):
            result._commit(1, obs(f), ((0, 0, 0),))
        row = result.process_frame(SimpleNamespace(frame_id=12), (obs(12),))[0]
        self.assertEqual(row["instance_id"], 1)
        self.assertEqual(row["candidates"][0]["probabilistic_geometric_consistency"], 1)

    def test_relative_weights_include_null_and_do_not_count_as_evidence(self):
        result = associator()
        result.process_frame(SimpleNamespace(frame_id=0), (obs(0),))
        row = result.process_frame(SimpleNamespace(frame_id=1), (obs(1),))[0]
        self.assertAlmostEqual(sum(item["weight"] for item in row["candidate_relative_weights"])
                               + row["null_relative_weight"], 1)
        self.assertGreater(row["null_relative_weight"], 0)
        self.assertEqual(result.identity_evidence.voxel_counts[(0, 0, 0)], {1: 2})


if __name__ == "__main__":
    unittest.main()

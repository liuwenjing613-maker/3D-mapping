"""Synthetic checks for causal parent proposal matching and valid sampling."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.frame_io import Camera, Frame
from revisable_instance_map.observations import RawInstanceObservation


def candidate(instance_id, score, geometric):
    return dict(instance_id=instance_id, score=score, geometric_coverage=geometric,
                visible_overlap=None, visible_support_points=0,
                prior_observation_count=1)


def obs(frame_id, local_id, pixels):
    return RawInstanceObservation(
        observation_id=f'f{frame_id}/m{local_id}', frame_id=frame_id,
        mask_local_id=local_id, source_mask_sha256='0' * 64,
        pixel_count=pixels, projectable_pixel_count=pixels,
        bbox_xyxy_exclusive=(0, 0, pixels, 1), world_centroid_m=None,
        world_aabb_min_m=None, world_aabb_max_m=None,
    )


class ParentProposalAssociationTest(unittest.TestCase):
    def test_second_candidate_used_when_best_fails_gate(self):
        associator = OnlineVoxelAssociator(improved_association=True)
        associator.next_instance_id = 3
        associator._sample_observation_voxels = lambda frame, observation: ((1, 2, 3),)
        associator._score_candidates = lambda frame, observation, voxels: [
            candidate(1, .8, .1), candidate(2, .6, .6)]
        decisions = associator.process_frame(SimpleNamespace(frame_id=1), (obs(1, 1, 1),))
        self.assertEqual(decisions[0]['instance_id'], 2)
        self.assertEqual(decisions[0]['decision'], 'matched')

    def test_claimed_candidate_falls_back_without_same_frame_evidence(self):
        associator = OnlineVoxelAssociator(improved_association=True)
        associator.next_instance_id = 3
        associator._sample_observation_voxels = lambda frame, observation: ((observation.mask_local_id, 0, 0),)
        seen_prior_counts = []
        def score(frame, observation, voxels):
            seen_prior_counts.append(len(associator.instances))
            return [candidate(1, .8, .8), candidate(2, .7, .7)]
        associator._score_candidates = score
        decisions = associator.process_frame(SimpleNamespace(frame_id=1), (obs(1, 1, 1), obs(1, 2, 1)))
        self.assertEqual([d['instance_id'] for d in decisions], [1, 2])
        self.assertEqual(seen_prior_counts, [0, 0])

    def test_valid_depth_is_filtered_before_budget(self):
        camera = Camera(6, 1, 1., 1., 0., 0., 1.)
        frame = Frame(0, np.zeros((1, 6, 3), np.uint8),
                      np.array([[0, 0, 1, 1, 1, 1]], np.float32),
                      np.ones((1, 6), np.uint8), np.eye(4), camera)
        observation = obs(0, 1, 6)
        improved = OnlineVoxelAssociator(max_points_per_observation=2, improved_association=True)
        legacy = OnlineVoxelAssociator(max_points_per_observation=2)
        self.assertEqual(len(improved._sample_observation_voxels(frame, observation)), 2)
        self.assertEqual(len(legacy._sample_observation_voxels(frame, observation)), 1)

    def test_reassignment_rebuilds_index_without_stale_support(self):
        associator = OnlineVoxelAssociator(improved_association=True)
        associator._commit(1, obs(0, 1, 1), ((1, 0, 0),))
        associator._commit(2, obs(0, 2, 1), ((2, 0, 0),))
        associator.reassign_observations({'f0/m1': 2})
        self.assertNotIn(1, associator.instances)
        self.assertEqual(associator.voxel_to_instances[(1, 0, 0)], {2})
        self.assertEqual(associator.instances[2].observation_ids, ['f0/m1', 'f0/m2'])
        with self.assertRaises(ValueError):
            associator.reassign_observations({'missing': 3})
        self.assertEqual(associator.voxel_to_instances[(1, 0, 0)], {2})


if __name__ == '__main__':
    unittest.main()

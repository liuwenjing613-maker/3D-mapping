"""Correctness gates before any ablation outputs or GT evaluation."""
from pathlib import Path
import sys
import unittest
import numpy as np
from one_vote import pair_keys, ambiguous_points, effective_keys, supported_depth_preferences

TASK = Path(__file__).resolve().parent
LOCAL = TASK.parents[1]
legacy = LOCAL/'outputs/pilot10_repair_20261007'
sys.path.insert(0, str(legacy if legacy.exists() else TASK.parent/'pilot10_repair_20261007'))
from evidence_replacement import EvidenceReplacement
source = LOCAL/'worktrees/ROI-repair/revisable_instance_map/src'
if not source.exists():source = TASK.parent/'p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src'
sys.path.insert(0,str(source))
from revisable_instance_map.surface_projective_evidence import project_surface_regions
from types import SimpleNamespace


class OneVoteTests(unittest.TestCase):
    def test_shared_id_is_retained_when_one_raw_mask_retires(self):
        for policy in ('original', 'abstain', 'depth'):
            before = effective_keys([0, 0, 1], [10, 10, 20], 100, policy)
            after = effective_keys([0, 1], [10, 20], 100, policy)
            np.testing.assert_array_equal(before, after)

    def test_unresolved_collision_abstains_and_clear_point_survives(self):
        np.testing.assert_array_equal(effective_keys([0, 0, 0, 1], [10, 10, 20, 30], 100, 'abstain'), [130])
        np.testing.assert_array_equal(effective_keys([0, 0, 0], [10, 20, 30], 100, 'abstain'), [])

    def test_depth_can_only_resolve_an_existing_collision(self):
        points, ids = [0, 0, 1], [10, 20, 30]
        np.testing.assert_array_equal(effective_keys(points, ids, 100, 'depth', [20]), [20, 130])
        np.testing.assert_array_equal(effective_keys(points, ids, 100, 'depth', [40, 240]), [130])
        with self.assertRaises(ValueError):
            effective_keys(points, ids, 100, 'depth', [10, 20])

    def test_central_pixel_must_have_saved_raw_mask_support(self):
        table = np.array([-1, 10, 20, 30])
        keys = supported_depth_preferences([0, 0, 1], [1, 2, 3], [0, 1, 2], [2, 1, 3], table, 100)
        np.testing.assert_array_equal(keys, [20])

    def test_raw_replacement_recomputes_frame_before_ledger_delta(self):
        base = 100
        before = [effective_keys([0, 0], [10, 20], base, 'abstain'),
                  effective_keys([0, 1], [10, 20], base, 'abstain')]
        keys, votes = np.unique(np.concatenate(before), return_counts=True)
        ledger = EvidenceReplacement(keys, votes)
        original_digest = ledger.digest()
        after = effective_keys([0], [10], base, 'abstain')
        self.assertTrue(ledger.replace('raw-mask-retire', before[0], after))
        full_keys, full_votes = np.unique(np.concatenate([after, before[1]]), return_counts=True)
        np.testing.assert_array_equal(ledger.keys, full_keys)
        np.testing.assert_array_equal(ledger.votes, full_votes)
        changed_digest = ledger.digest()
        self.assertFalse(ledger.replace('raw-mask-retire', before[0], after))
        self.assertEqual(ledger.digest(), changed_digest)
        ledger.undo('raw-mask-retire')
        self.assertEqual(ledger.digest(), original_digest)

    def test_incremental_equals_independent_full_recompute_randomized(self):
        rng = np.random.default_rng(20261009)
        for policy in ('original', 'abstain', 'depth'):
            for trial in range(30):
                raw = [(rng.integers(0, 15, 60), rng.integers(1, 6, 60)) for _ in range(4)]
                preferred = [np.arange(15, dtype=np.int64)*100+rng.integers(1,6,15) for _ in raw]
                frames = [effective_keys(p, ids, 100, policy, pref) for (p,ids),pref in zip(raw,preferred)]
                keys, votes = np.unique(np.concatenate(frames), return_counts=True)
                ledger = EvidenceReplacement(keys, votes)
                p, ids = raw[trial%4]
                keep = rng.random(len(p)) > 0.35
                changed = effective_keys(p[keep], ids[keep], 100, policy, preferred[trial%4])
                ledger.replace(str(trial), frames[trial%4], changed)
                frames[trial%4] = changed
                full_keys, full_votes = np.unique(np.concatenate(frames), return_counts=True)
                np.testing.assert_array_equal(ledger.keys, full_keys)
                np.testing.assert_array_equal(ledger.votes, full_votes)
                for frame in frames:
                    if policy != 'original':self.assertEqual(len(frame),len(np.unique(frame//100)))

    def test_order_does_not_choose_a_winner(self):
        points, ids = np.array([0, 0, 1, 1, 2]), np.array([10, 20, 30, 30, 40])
        order = [4, 2, 0, 3, 1]
        for policy in ('original', 'abstain', 'depth'):
            first = effective_keys(points, ids, 100, policy, [20])
            second = effective_keys(points[order], ids[order], 100, policy, [20])
            np.testing.assert_array_equal(first, second)

    def test_empty_and_invalid_inputs(self):
        for policy in ('original', 'abstain', 'depth'):
            self.assertEqual(len(effective_keys([], [], 100, policy)),0)
        for points,ids,base in (([-1],[1],100),([0],[0],100),([0],[100],100),([0,1],[1],100)):
            with self.assertRaises(ValueError):pair_keys(points, ids, base)

    def test_frozen_depth_projection_rejects_occlusion_invalid_depth_and_outside(self):
        camera=SimpleNamespace(width=3,height=3,fx=1.,fy=1.,cx=1.,cy=1.)
        frame=SimpleNamespace(camera=camera,camera_to_world=np.eye(4),
            depth_m=np.ones((3,3),np.float32),mask_local=np.full((3,3),2,np.int32))
        xyz=np.array([[0,0,1],[0,0,2],[4,0,1],[0,0,-1],[0,0,1.01],[0,0,1.03]])
        points,local,_=project_surface_regions(frame,xyz,.015)
        np.testing.assert_array_equal(points,[0,4])
        np.testing.assert_array_equal(local,[2,2])
        frame.depth_m[1,1]=0
        self.assertEqual(len(project_surface_regions(frame,xyz,.015)[0]),0)

    def test_frozen_depth_projection_handles_rotation_and_translation(self):
        rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=[3,4,5]
        camera=SimpleNamespace(width=3,height=3,fx=1.,fy=1.,cx=1.,cy=1.)
        frame=SimpleNamespace(camera=camera,camera_to_world=pose,
            depth_m=np.ones((3,3),np.float32),mask_local=np.full((3,3),2,np.int32))
        xyz=np.array([[0,0,1],[0,0,2]])@rotation.T+pose[:3,3]
        np.testing.assert_array_equal(project_surface_regions(frame,xyz,.015)[0],[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from revisable_instance_map.frame_io import Camera, Frame
from revisable_instance_map.surface_evidence import (
    UNOBSERVED, TENTATIVE, CONFIRMED, CONFLICT,
    project_frame_regions, frame_instance_keys, reduce_surface_votes,
)


class SurfaceEvidenceTest(unittest.TestCase):
    def test_frame_vote_is_not_pixel_or_fragment_count(self):
        camera = Camera(3, 1, 1000., 1000., 0., 0., 1.)
        frame = Frame(0, np.zeros((1, 3, 3), np.uint8),
                      np.ones((1, 3), np.float32),
                      np.array([[1, 1, 2]], np.uint8), np.eye(4), camera)
        points, regions, stats = project_frame_regions(
            frame, cKDTree(np.array([[0., 0., 1.]])), 1, .015, 1)
        self.assertEqual(stats['matched_pixels'], 3)
        self.assertEqual(list(zip(points, regions)), [(0, 1), (0, 2)])
        keys = frame_instance_keys(points, regions,
                                   np.array([-1, 5, 5], np.int32), 10)
        np.testing.assert_array_equal(keys, np.array([5], np.int64))

    def test_surface_states_keep_singleton_and_competition_separate(self):
        base = 10
        keys = np.array([0*base+1, 0*base+2, 1*base+3,
                         2*base+4, 3*base+5, 3*base+6], np.int64)
        votes = np.array([2, 1, 1, 2, 2, 2], np.int32)
        result = reduce_surface_votes(keys, votes, 5, base)
        self.assertEqual(result['state'].tolist(),
                         [CONFLICT, TENTATIVE, CONFIRMED, CONFLICT, UNOBSERVED])
        self.assertEqual(result['top1_votes'].tolist(), [2, 1, 2, 2, 0])
        self.assertEqual(result['top2_votes'].tolist(), [1, 0, 0, 2, 0])
        self.assertEqual(result['total_frame_votes'].tolist(), [3, 1, 2, 4, 0])
        self.assertEqual(result['top2_instance_id'].tolist(), [2, -1, -1, 6, -1])

    def test_duplicate_pair_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            reduce_surface_votes(np.array([11, 11]), np.array([1, 1]), 2, 10)


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from revisable_instance_map.frame_io import Camera, Frame
from revisable_instance_map.surface_projective_evidence import (
    project_surface_regions,
)


def make_frame(depth, mask, pose=None):
    height, width = depth.shape
    camera = Camera(width, height, 1.0, 1.0, 0.0, 0.0, 1.0)
    if pose is None:
        pose = np.eye(4)
    return Frame(
        0, np.zeros((height, width, 3), np.uint8),
        np.asarray(depth, np.float32), np.asarray(mask, np.uint8),
        np.asarray(pose, np.float64), camera,
    )


class SurfaceProjectiveEvidenceTest(unittest.TestCase):
    def test_visibility_and_positive_mask_gate(self):
        frame = make_frame(
            np.array([[1.0, 1.03, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[1, 2, 0], [0, 0, 0]]),
        )
        xyz = np.array([
            [0.0, 0.0, 1.0],   # positive mask and depth consistent
            [1.0, 0.0, 1.0],   # depth mismatch
            [0.0, 1.0, 1.0],   # visible background mask
            [-1.0, 0.0, 1.0],  # outside image
            [0.0, 0.0, -1.0],  # behind camera
        ])
        points, local_ids, stats = project_surface_regions(
            frame, xyz, depth_tolerance_m=0.015, point_chunk_size=2)
        np.testing.assert_array_equal(points, np.array([0], np.int32))
        np.testing.assert_array_equal(local_ids, np.array([1], np.int32))
        self.assertEqual(stats['surface_points_tested'], 5)
        self.assertEqual(stats['visible_depth_consistent_points'], 2)
        self.assertEqual(stats['positive_mask_evidence_points'], 1)
        self.assertEqual(stats['background_visible_points'], 1)

    def test_camera_to_world_pose_is_inverted_for_projection(self):
        pose = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0,  0.0, 0.0, 2.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ])
        frame = make_frame(np.array([[2.0, 2.0]]), np.array([[0, 3]]), pose)
        # Camera [2, 0, 2] maps to world [1, 4, 2].
        points, local_ids, _ = project_surface_regions(
            frame, np.array([[1.0, 4.0, 2.0]]), point_chunk_size=1)
        np.testing.assert_array_equal(points, np.array([0], np.int32))
        np.testing.assert_array_equal(local_ids, np.array([3], np.int32))

    def test_each_surface_point_has_at_most_one_frame_label(self):
        frame = make_frame(
            np.ones((2, 3), np.float32),
            np.array([[1, 2, 3], [4, 5, 6]], np.uint8),
        )
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        points, local_ids, _ = project_surface_regions(
            frame, xyz, point_chunk_size=1)
        self.assertEqual(len(points), len(np.unique(points)))
        np.testing.assert_array_equal(points, np.arange(4, dtype=np.int32))
        np.testing.assert_array_equal(local_ids, np.array([1, 2, 4, 5]))

    def test_invalid_arguments_are_rejected(self):
        frame = make_frame(np.ones((1, 1)), np.ones((1, 1), np.uint8))
        with self.assertRaises(ValueError):
            project_surface_regions(frame, np.empty((0, 3)))
        with self.assertRaises(ValueError):
            project_surface_regions(frame, np.zeros((1, 3)), depth_tolerance_m=0)


if __name__ == '__main__':
    unittest.main()

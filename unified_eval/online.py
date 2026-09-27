from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .schema import EvaluationError


def observed_vertices_from_depth(depth_m: np.ndarray, intrinsics: np.ndarray,
                                 world_from_camera: np.ndarray, ref_xyz: np.ndarray,
                                 max_distance_m: float) -> np.ndarray:
    """Return reference vertices supported by one RGB-D frame, without future views."""
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    ref = np.asarray(ref_xyz, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or pose.shape != (4, 4) or ref.ndim != 2 or ref.shape[1] != 3:
        raise EvaluationError("Depth, intrinsics, pose or reference shape is invalid")
    if not np.isfinite(k).all() or not np.isfinite(pose).all() or not np.isfinite(ref).all():
        raise EvaluationError("Camera or reference values are not finite")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError("Observation mapping distance must be finite and positive")
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(rows) or not len(ref):
        return np.zeros(len(ref), dtype=bool)
    homogeneous = np.stack((cols, rows, np.ones(len(rows))), axis=0)
    camera = np.linalg.solve(k, homogeneous).T * depth[rows, cols, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    distance, nearest = cKDTree(world).query(ref, k=1, workers=-1)
    return distance < max_distance_m


def update_observation_count(count: np.ndarray, observed_this_frame: np.ndarray) -> np.ndarray:
    """Count each reference vertex once per frame; saturate uint16 safely."""
    value = np.asarray(count)
    observed = np.asarray(observed_this_frame, dtype=bool)
    if value.dtype != np.uint16 or value.shape != observed.shape:
        raise EvaluationError("Observation count must be uint16 and match observed mask")
    result = value.copy()
    active = observed & (result < np.iinfo(np.uint16).max)
    result[active] += 1
    return result

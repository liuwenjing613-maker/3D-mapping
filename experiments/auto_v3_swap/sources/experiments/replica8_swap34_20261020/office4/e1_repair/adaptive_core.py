"""Seed-only adaptive erosion. Never inspect votes, labels, tracking or GT."""
import math
import numpy as np
from scipy.ndimage import binary_erosion


def choose_core(mask, project_points, min_points=20, retained_fraction=0.20):
    mask = np.asarray(mask, dtype=bool)
    seed_points = np.unique(project_points(mask))
    required = max(min_points, math.ceil(retained_fraction * len(seed_points)))
    profile = []
    for radius in range(10, 0, -1):
        interior = binary_erosion(mask, iterations=radius)
        points = np.unique(project_points(interior))
        assert np.all(np.isin(points, seed_points))
        profile.append(dict(radius=radius, pixels=int(interior.sum()), points=len(points)))
        if len(points) >= required:
            return interior, dict(radius=radius, seed_points=len(seed_points),
                required_points=required, retained_points=len(points),
                adaptive=radius != 10, status='PASS', profile=profile)
    return np.zeros(mask.shape, bool), dict(radius=None, seed_points=len(seed_points),
        required_points=required, retained_points=0, adaptive=True,
        status='INSUFFICIENT_SAFE_INTERIOR', profile=profile)

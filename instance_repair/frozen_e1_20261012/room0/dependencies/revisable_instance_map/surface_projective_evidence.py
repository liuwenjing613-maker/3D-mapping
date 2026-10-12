"""Project shared TSDF surface points into one RGB-D frame.

This is the P0.1 evidence direction: every surface point queries exactly one
image pixel per frame.  A depth-consistency test determines visibility before
the frame-local mask label is read.
"""

import numpy as np


def project_surface_regions(frame, surface_xyz, depth_tolerance_m=0.015,
                            point_chunk_size=250_000, max_depth_m=10.0):
    """Return visible positive-mask (surface index, local mask ID) pairs.

    The returned surface indices are unique.  Therefore a frame contributes at
    most one local region label, and later at most one persistent instance
    label, to each surface point.
    """
    xyz = np.asarray(surface_xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError('surface_xyz must be a non-empty N x 3 array')
    if depth_tolerance_m <= 0 or point_chunk_size < 1 or max_depth_m <= 0:
        raise ValueError('Invalid projective evidence settings')
    if not np.all(np.isfinite(xyz)):
        raise ValueError('surface_xyz contains non-finite coordinates')

    camera = frame.camera
    rotation = frame.camera_to_world[:3, :3]
    translation = frame.camera_to_world[:3, 3]
    point_parts = []
    local_parts = []
    stats = {
        'surface_points_tested': len(xyz),
        'in_front_points': 0,
        'inside_image_points': 0,
        'valid_depth_points': 0,
        'visible_depth_consistent_points': 0,
        'positive_mask_evidence_points': 0,
        'background_visible_points': 0,
    }

    for start in range(0, len(xyz), point_chunk_size):
        stop = min(start + point_chunk_size, len(xyz))
        world = xyz[start:stop].astype(np.float64, copy=False)
        # Row-vector inverse of: world = camera_xyz @ R.T + translation.
        camera_xyz = (world - translation) @ rotation
        z = camera_xyz[:, 2]
        front = np.isfinite(z) & (z > 0) & (z < max_depth_m)
        stats['in_front_points'] += int(np.count_nonzero(front))
        if not np.any(front):
            continue

        relative = np.flatnonzero(front)
        camera_front = camera_xyz[front]
        z_front = camera_front[:, 2]
        u_float = camera.fx * camera_front[:, 0] / z_front + camera.cx
        v_float = camera.fy * camera_front[:, 1] / z_front + camera.cy
        finite_projection = (np.isfinite(u_float) & np.isfinite(v_float) &
                             (np.abs(u_float) < np.iinfo(np.int32).max) &
                             (np.abs(v_float) < np.iinfo(np.int32).max))
        relative = relative[finite_projection]
        z_front = z_front[finite_projection]
        u = np.rint(u_float[finite_projection]).astype(np.int32)
        v = np.rint(v_float[finite_projection]).astype(np.int32)
        inside = ((u >= 0) & (u < camera.width) &
                  (v >= 0) & (v < camera.height))
        stats['inside_image_points'] += int(np.count_nonzero(inside))
        if not np.any(inside):
            continue

        relative = relative[inside]
        u = u[inside]
        v = v[inside]
        z_inside = z_front[inside]
        observed_depth = frame.depth_m[v, u]
        valid_depth = (np.isfinite(observed_depth) & (observed_depth > 0) &
                       (observed_depth < max_depth_m))
        stats['valid_depth_points'] += int(np.count_nonzero(valid_depth))
        if not np.any(valid_depth):
            continue

        relative = relative[valid_depth]
        u = u[valid_depth]
        v = v[valid_depth]
        z_inside = z_inside[valid_depth]
        observed_depth = observed_depth[valid_depth]
        visible = np.abs(observed_depth - z_inside) <= depth_tolerance_m
        stats['visible_depth_consistent_points'] += int(np.count_nonzero(visible))
        if not np.any(visible):
            continue

        relative = relative[visible]
        u = u[visible]
        v = v[visible]
        local_ids = frame.mask_local[v, u]
        positive = local_ids > 0
        stats['positive_mask_evidence_points'] += int(np.count_nonzero(positive))
        stats['background_visible_points'] += int(np.count_nonzero(~positive))
        if np.any(positive):
            point_parts.append((start + relative[positive]).astype(np.int32))
            local_parts.append(local_ids[positive].astype(np.int32))

    if not point_parts:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, stats
    points = np.concatenate(point_parts)
    local_ids = np.concatenate(local_parts)
    if len(points) != len(np.unique(points)):
        raise AssertionError('Surface-to-frame projection produced duplicate points')
    if len(points) != stats['positive_mask_evidence_points']:
        raise AssertionError('Projection statistics disagree with returned evidence')
    return points, local_ids, stats

"""Direct RGB-D evidence on shared TSDF surface points.

The immutable frame-region support is independent of current persistent IDs.
One frame contributes at most one vote per (surface point, instance ID).
"""
import numpy as np

UNOBSERVED = 0
TENTATIVE = 1
CONFIRMED = 2
CONFLICT = 3
STATE_NAMES = ('unobserved', 'tentative', 'confirmed', 'conflict')


def project_frame_regions(frame, surface_tree, pixel_stride=2,
                          max_distance_m=0.015, workers=8, max_depth_m=10.0):
    """Return unique (surface index, local mask ID) pairs for one frame."""
    if pixel_stride < 1 or max_distance_m <= 0 or workers < 1:
        raise ValueError('Invalid surface projection settings')
    local = frame.mask_local[::pixel_stride, ::pixel_stride]
    depth = frame.depth_m[::pixel_stride, ::pixel_stride]
    valid = (local > 0) & np.isfinite(depth) & (depth > 0) & (depth < max_depth_m)
    sample_rows, sample_cols = np.nonzero(valid)
    if not len(sample_rows):
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, {'eligible_pixels': 0, 'matched_pixels': 0,
                              'unique_region_surface_pairs': 0}
    rows = sample_rows.astype(np.int32) * pixel_stride
    cols = sample_cols.astype(np.int32) * pixel_stride
    z = depth[valid].astype(np.float64)
    camera_xyz = np.column_stack((
        (cols - frame.camera.cx) * z / frame.camera.fx,
        (rows - frame.camera.cy) * z / frame.camera.fy,
        z,
    ))
    world_xyz = camera_xyz @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
    distances, surface_idx = surface_tree.query(world_xyz, k=1, workers=workers)
    accepted = np.isfinite(distances) & (distances < max_distance_m)
    if not np.any(accepted):
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, {'eligible_pixels': len(z), 'matched_pixels': 0,
                              'unique_region_surface_pairs': 0}
    local_ids = local[valid][accepted].astype(np.int64)
    surface_idx = surface_idx[accepted].astype(np.int64)
    base = int(frame.mask_local.max()) + 1
    unique = np.unique(surface_idx * base + local_ids)
    points = (unique // base).astype(np.int32)
    region_ids = (unique % base).astype(np.int32)
    return points, region_ids, {
        'eligible_pixels': len(z), 'matched_pixels': int(np.count_nonzero(accepted)),
        'unique_region_surface_pairs': len(points),
        'nearest_distance_median_m': float(np.median(distances[accepted])),
    }


def frame_instance_keys(surface_points, local_ids, local_to_instance, instance_base):
    """Deduplicate all regions of one frame by (surface point, persistent ID)."""
    if len(surface_points) != len(local_ids):
        raise ValueError('Surface and region arrays differ in length')
    if not len(surface_points):
        return np.empty(0, dtype=np.int64)
    if np.max(local_ids) >= len(local_to_instance):
        raise ValueError('Unmapped local mask ID')
    instance_ids = local_to_instance[local_ids].astype(np.int64)
    if np.any(instance_ids <= 0) or np.any(instance_ids >= instance_base):
        raise ValueError('A sampled region lacks a valid persistent ID')
    return np.unique(surface_points.astype(np.int64) * instance_base + instance_ids)


def reduce_surface_votes(keys, votes, surface_count, instance_base,
                         min_confirmed_votes=2, min_confirmed_ratio=0.67):
    """Aggregate sorted or unsorted (point, ID) vote counts into surface state."""
    if len(keys) != len(votes) or surface_count < 1 or instance_base < 2:
        raise ValueError('Invalid vote arrays or surface dimensions')
    if not 0.5 < min_confirmed_ratio <= 1 or min_confirmed_votes < 1:
        raise ValueError('Invalid confidence thresholds')
    arrays = {
        'top1_instance_id': np.full(surface_count, -1, np.int32),
        'top1_votes': np.zeros(surface_count, np.int32),
        'top2_instance_id': np.full(surface_count, -1, np.int32),
        'top2_votes': np.zeros(surface_count, np.int32),
        'total_frame_votes': np.zeros(surface_count, np.int32),
        'confidence': np.zeros(surface_count, np.float32),
        'margin': np.zeros(surface_count, np.float32),
        'state': np.full(surface_count, UNOBSERVED, np.uint8),
    }
    if not len(keys):
        return arrays
    order = np.argsort(keys, kind='mergesort')
    keys = np.asarray(keys, dtype=np.int64)[order]
    votes = np.asarray(votes, dtype=np.int32)[order]
    if np.any(votes <= 0) or np.any(keys[1:] == keys[:-1]):
        raise ValueError('Vote keys must be unique with positive counts')
    point = keys // instance_base
    instance = keys % instance_base
    if np.any(point < 0) or np.any(point >= surface_count) or np.any(instance <= 0):
        raise ValueError('Vote key outside surface or instance range')
    starts = np.r_[0, np.flatnonzero(point[1:] != point[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(keys)])
    positions = np.arange(len(keys))
    group_index = np.repeat(np.arange(len(starts)), sizes)
    best_votes = np.maximum.reduceat(votes, starts)
    total_votes = np.add.reduceat(votes, starts)
    first = np.minimum.reduceat(
        np.where(votes == best_votes[group_index], positions, len(keys)), starts)
    remaining = positions != first[group_index]
    second_candidate_votes = np.where(remaining, votes, 0)
    second_votes = np.maximum.reduceat(second_candidate_votes, starts)
    second = np.minimum.reduceat(
        np.where(remaining & (votes == second_votes[group_index]) &
                 (second_votes[group_index] > 0), positions, len(keys)), starts)
    selected_points = point[starts].astype(np.intp)
    ratio = best_votes / total_votes
    margin = (best_votes - second_votes) / total_votes
    conflict = (second_votes > 0) & (ratio < min_confirmed_ratio)
    confirmed = ((best_votes >= min_confirmed_votes) &
                 (ratio >= min_confirmed_ratio))
    state = np.full(len(starts), TENTATIVE, np.uint8)
    state[confirmed] = CONFIRMED
    state[conflict] = CONFLICT
    arrays['top1_instance_id'][selected_points] = instance[first].astype(np.int32)
    arrays['top1_votes'][selected_points] = best_votes
    arrays['top2_instance_id'][selected_points] = np.where(second < len(keys),
        instance[np.minimum(second, len(keys) - 1)], -1).astype(np.int32)
    arrays['top2_votes'][selected_points] = second_votes
    arrays['total_frame_votes'][selected_points] = total_votes
    arrays['confidence'][selected_points] = ratio.astype(np.float32)
    arrays['margin'][selected_points] = margin.astype(np.float32)
    arrays['state'][selected_points] = state
    return arrays

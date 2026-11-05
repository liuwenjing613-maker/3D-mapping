"""GT-free seed modes, independent acceptance, and observation replacement."""
import numpy as np

SEED_MODES = {'original_control', 'original_reassociate', 'new_mask_reassociate'}


def seed_labels(original, candidate, objects, mode):
    if mode not in SEED_MODES:
        raise ValueError('Specify an explicit seed mode')
    track_ids = [int(o['track_id']) for o in objects]
    if not track_ids or len(set(track_ids)) != len(track_ids) or min(track_ids) <= 0 or max(track_ids) > 65535:
        raise ValueError('Track IDs must be distinct positive uint16 IDs')
    if mode.startswith('original_'):
        mask_ids = [int(o['original_mask_id']) for o in objects]
        if len(set(mask_ids)) != len(mask_ids) or any(mid <= 0 or not np.any(original == mid) for mid in mask_ids):
            raise ValueError('Select distinct existing original masks')
        result = np.zeros(original.shape, np.uint16)
        for oid, mid in zip(track_ids, mask_ids):
            result[original == mid] = oid
            np.testing.assert_array_equal(result == oid, original == mid)
        return result
    candidate = np.asarray(candidate)
    if candidate.shape != original.shape or candidate.dtype.kind not in 'iu':
        raise ValueError('New seed must use full original RGB coordinates')
    if any(not np.any(candidate == oid) for oid in track_ids):
        raise ValueError('A selected seed object is empty')
    return np.where(np.isin(candidate, track_ids), candidate, 0).astype(np.uint16)


def freeze_association(rows, objects, mode, original_lookup):
    """Keep many-to-one only in the explicit no-op control."""
    if mode not in SEED_MODES:
        raise ValueError('Unknown seed mode')
    by_track = {int(row['track_id']): dict(row) for row in rows}
    output = []
    for obj in objects:
        row = by_track[int(obj['track_id'])]
        if mode == 'original_control':
            row['persistent_id'] = int(original_lookup[int(obj['original_mask_id'])])
            row['association'] = 'exact_original_observation_control'
        row['seed_mode'] = mode
        row['original_mask_id'] = obj.get('original_mask_id')
        output.append(row)
    ids = [int(row['persistent_id']) for row in output]
    if any(pid <= 0 for pid in ids) or (mode != 'original_control' and len(set(ids)) != len(ids)):
        raise ValueError('Repair objects must have distinct positive identities')
    return output


def object_reliability(metrics, policy):
    reasons = []
    if metrics['visible_seed_samples'] < policy['minimum_visible_seed_surface_samples_each_target']:
        reasons.append('insufficient_visible_anchor')
    if metrics['tracked_coverage'] < policy['minimum_tracked_coverage_of_visible_seed_samples_each_target']:
        reasons.append('anchor_coverage')
    if metrics['new_surface_bbox_fraction'] < policy['minimum_new_surface_support_fraction_inside_target_bbox']:
        reasons.append('spatial_extent')
    if metrics['foreign_old_mask_pixel_fraction'] > policy['maximum_new_mask_pixel_overlap_fraction_with_other_old_instance_families']:
        reasons.append('foreign_old_family_overlap')
    for row in metrics['other_seed_overlaps']:
        if (row['visible_seed_samples'] >= policy['minimum_visible_protected_seed_samples'] and
                row['coverage'] > policy['maximum_new_union_coverage_of_other_visible_seed_samples']):
            reasons.append('other_seed_%s_overlap' % row['track_id'])
    return not reasons, reasons


def observation_plan(accepted_ids, target_ids, old_spatial_fraction, union_protected_ok, policy):
    """Reliable foreground replaces partially; fully explained observations retire."""
    accepted = set(accepted_ids)
    if not accepted:
        return 'retain'
    if (accepted == set(target_ids) and union_protected_ok and
            old_spatial_fraction >= policy['minimum_old_surface_support_fraction_inside_joint_bbox']):
        return 'whole'
    return 'partial'


def replace_observations(old_points, old_local, old_table, plans,
                         partial_points, partial_local, new_points, new_ids, base):
    """Rebuild a frame from observations before dedup; preserve shared old votes.

    partial_* must be a fresh projection of the original family pixels outside
    reliable foreground, not subtraction of point IDs. A retained pixel of the
    same old mask may still support a shared surface point.
    """
    old_points, old_local = np.asarray(old_points), np.asarray(old_local)
    partial_points, partial_local = np.asarray(partial_points), np.asarray(partial_local)
    if old_points.shape != old_local.shape or partial_points.shape != partial_local.shape:
        raise ValueError('Surface/local arrays differ')
    if any(action not in {'retain', 'partial', 'whole'} for action in plans.values()):
        raise ValueError('Invalid observation action')
    selected = [mid for mid, action in plans.items() if action != 'retain']
    partial = [mid for mid, action in plans.items() if action == 'partial']
    if not np.all(np.isin(partial_local, partial)):
        raise ValueError('Residual projection includes an unselected observation')
    original_pairs = np.c_[old_points, old_local]
    residual_pairs = np.c_[partial_points, partial_local]
    original_set = set(map(tuple, original_pairs[np.isin(old_local, partial)].tolist()))
    if any(tuple(pair) not in original_set for pair in residual_pairs):
        raise ValueError('Residual projection introduced old support')
    keep = ~np.isin(old_local, selected)
    retained_points = np.r_[old_points[keep], partial_points]
    retained_local = np.r_[old_local[keep], partial_local]
    from joint_vote_ops import keys_from_pairs
    retained = keys_from_pairs(retained_points, old_table[retained_local], base)
    new = keys_from_pairs(new_points, new_ids, base)
    revised = np.union1d(retained, new)
    # Select removed observation-region pairs, before persistent-ID dedup.
    remaining = set(map(tuple, np.c_[retained_points, retained_local].tolist()))
    retired = np.array([pair for pair in original_pairs if tuple(pair) not in remaining], dtype=np.int64).reshape(-1, 2)
    scope = np.unique(np.r_[retired[:, 0], new_points]).astype(np.int32)
    return revised, retained, retired, scope

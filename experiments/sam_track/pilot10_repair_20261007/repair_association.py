"""Freeze seed identities from visible predicted surfaces; never uses GT."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_seed(points, local_ids, old_visible_points, baseline_labels, object_ids,
                   maximum_original_id, threshold=0.5):
    visible = np.unique(np.concatenate([points, old_visible_points]))
    old_ids, old_counts = np.unique(baseline_labels[visible][baseline_labels[visible] > 0], return_counts=True)
    score = np.full((len(object_ids), len(old_ids) + len(object_ids)), -1., np.float64)
    dice = np.zeros((len(object_ids), len(old_ids)), np.float64)
    seed_sizes = []
    for row, oid in enumerate(object_ids):
        support = np.unique(points[local_ids == oid])
        seed_sizes.append(len(support))
        labels, counts = np.unique(baseline_labels[support], return_counts=True)
        overlap = dict(zip(labels.tolist(), counts.tolist()))
        for col, old_id in enumerate(old_ids):
            dice[row, col] = 2 * overlap.get(int(old_id), 0) / max(1, len(support) + int(old_counts[col]))
            if dice[row, col] > threshold:
                score[row, col] = dice[row, col]
        score[row, len(old_ids) + row] = 0
    rows, cols = linear_sum_assignment(-score)
    result = []
    for row, col in sorted(zip(rows.tolist(), cols.tolist())):
        oid = object_ids[row]
        matched = col < len(old_ids)
        result.append({'track_id': oid,
                       'persistent_id': int(old_ids[col]) if matched else maximum_original_id + oid,
                       'association': 'visible_surface_dice' if matched else 'distinct_new_identity',
                       'accepted_dice': float(dice[row, col]) if matched else None,
                       'best_visible_dice': float(dice[row].max(initial=0)),
                       'projected_seed_points': seed_sizes[row]})
    assert len(set(r['persistent_id'] for r in result)) == len(object_ids)
    return result


def compose_frame_mask(original, selected, old_local_to_persistent, frozen_track_to_persistent):
    """Only selected pixels replace their original labels; zero is outside scope."""
    if original.shape != selected.shape:
        raise ValueError('Mask dimensions differ')
    out = original.astype(np.uint16, copy=True)
    offset = int(original.max())
    largest = max(frozen_track_to_persistent, default=0)
    lookup = np.full(offset + largest + 1, -1, np.int32)
    lookup[:len(old_local_to_persistent)] = old_local_to_persistent
    for track_id, persistent_id in frozen_track_to_persistent.items():
        proxy = offset + track_id
        if proxy > np.iinfo(np.uint16).max:
            raise ValueError('Local proxy ID exceeds uint16')
        out[selected == track_id] = proxy
        lookup[proxy] = persistent_id
    if not set(np.unique(selected)).issubset({0, *frozen_track_to_persistent}):
        raise ValueError('Unexpected track ID')
    np.testing.assert_array_equal(out[selected == 0], original[selected == 0])
    return out, lookup

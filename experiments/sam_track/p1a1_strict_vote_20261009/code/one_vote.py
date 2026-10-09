"""Reduce immutable raw observations into effective per-frame identity votes."""
import numpy as np

POLICIES = ('original', 'abstain', 'depth')


def pair_keys(points, identities, base):
    points = np.asarray(points, dtype=np.int64)
    identities = np.asarray(identities, dtype=np.int64)
    if points.ndim != 1 or points.shape != identities.shape or base < 2:
        raise ValueError('Invalid point/identity vectors or instance base')
    if np.any(points < 0) or np.any(identities <= 0) or np.any(identities >= base):
        raise ValueError('Invalid point index or identity')
    if len(points) and int(points.max()) > (np.iinfo(np.int64).max - base) // base:
        raise ValueError('Vote encoding overflows int64')
    return np.unique(points * base + identities)


def ambiguous_points(keys, base):
    keys = np.asarray(keys, dtype=np.int64)
    points, counts = np.unique(keys // base, return_counts=True)
    return points[counts > 1]


def effective_keys(points, identities, base, policy, preferred_keys=None):
    """Shared masks of the same ID collapse once; unresolved multi-ID abstains."""
    if policy not in POLICIES:
        raise ValueError('Unknown vote policy')
    keys = pair_keys(points, identities, base)
    if policy == 'original' or not len(keys):
        return keys
    bad_points = ambiguous_points(keys, base)
    bad = np.isin(keys // base, bad_points)
    kept = keys[~bad]
    if policy == 'depth' and preferred_keys is not None:
        preferred = np.unique(np.asarray(preferred_keys, dtype=np.int64))
        if preferred.ndim != 1 or np.any(preferred < 0):
            raise ValueError('Invalid depth preferences')
        if len(preferred) != len(np.unique(preferred // base)):
            raise ValueError('Depth evidence proposes multiple IDs for one point')
        # A preference cannot invent support or affect a non-colliding point.
        supported = np.intersect1d(preferred, keys[bad], assume_unique=True)
        kept = np.union1d(kept, supported)
    if len(kept) != len(np.unique(kept // base)):
        raise AssertionError('More than one effective vote at a point in one frame')
    return kept


def supported_depth_preferences(raw_points, raw_local, projected_points,
                                projected_local, local_to_instance, base):
    """A depth-visible central pixel must belong to the saved raw region support."""
    raw_points, raw_local = np.asarray(raw_points, np.int64), np.asarray(raw_local, np.int64)
    points, local = np.asarray(projected_points, np.int64), np.asarray(projected_local, np.int64)
    table = np.asarray(local_to_instance, np.int64)
    if raw_points.shape != raw_local.shape or points.shape != local.shape:
        raise ValueError('Point/local vectors differ')
    if np.any(raw_local <= 0) or np.any(local <= 0) or np.any(local >= len(table)):
        raise ValueError('Unmapped or background local mask')
    if len(points) != len(np.unique(points)):
        raise ValueError('Surface projection repeats a point')
    region_base = max(int(raw_local.max()) if len(raw_local) else 0,
                      int(local.max()) if len(local) else 0) + 1
    raw = np.unique(raw_points * region_base + raw_local)
    keep = np.isin(points * region_base + local, raw, assume_unique=False)
    return pair_keys(points[keep], table[local[keep]], base)

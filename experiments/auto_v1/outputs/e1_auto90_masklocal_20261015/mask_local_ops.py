"""Authorize historical changes by the FINAL adopted mask at the same source pixel.

No GT, seed-only exemption, neighboring-frame permission, or TSDF-neighbor
permission. Outside source observations survive even when projection collisions
later cause strict per-frame vote abstention.
"""
import numpy as np


def accepted_owner(labels, accepted_tracks, canonical, fixed):
    labels = np.asarray(labels)
    allowed = {int(canonical[int(t)]) for t in accepted_tracks}
    present = set(map(int, np.unique(labels))) - {0}
    if present != allowed:
        raise ValueError(f"Adopted mask / final Gate mismatch: {present} != {allowed}")
    out = np.zeros(labels.shape, np.int64)
    for cid in sorted(allowed):
        pid = int(fixed[cid])
        if pid <= 0:
            raise ValueError("Nonpositive target identity")
        out[labels == cid] = pid
    return out


def authorized(target_ids, adopted_ids):
    target, adopted = np.asarray(target_ids), np.asarray(adopted_ids)
    if target.shape != adopted.shape:
        raise ValueError("Source sample arrays differ")
    return (target > 0) & (target == adopted)


def scoped_history(points, old_ids, available, prior_owner, owner, visible, adopted_ids):
    """E1 reassociation with the same pixel AND same accepted target required."""
    points = np.asarray(points)
    old_ids = np.asarray(old_ids)
    ids = old_ids.copy()
    prior = prior_owner[points]
    take = np.asarray(available, bool) & authorized(prior, adopted_ids)
    ids[take] = prior[take]
    target = owner[points]
    take = np.asarray(available, bool) & visible[points] & authorized(target, adopted_ids)
    ids[take] = target[take]
    changed = ids != old_ids
    assert np.all(authorized(ids[changed], np.asarray(adopted_ids)[changed]))
    return ids, changed


def assert_source_scope(before, after, adopted):
    before, after, adopted = map(np.asarray, (before, after, adopted))
    if before.shape != after.shape or before.shape != adopted.shape:
        raise ValueError("Source sample arrays differ")
    changed = before != after
    assert np.all(adopted[changed] > 0), "Changed an original observation outside adopted mask"
    relabeled = changed & (after > 0)
    assert np.all(after[relabeled] == adopted[relabeled]), "Changed to a different accepted target"
    return changed

"""Containment uses full unique fixed-surface indices, without cross-view clipping."""
import numpy as np


def compare(a_id, a, b_id, b):
    for points in (a, b):
        if points.ndim != 1 or not len(points) or not np.all(points[1:] > points[:-1]):
            raise ValueError('Support must be nonempty, unique and sorted.')
    na, nb = len(a), len(b)
    common = int(np.intersect1d(a, b, assume_unique=True).size)
    passes = 10 * common > 7 * min(na, nb)
    if passes and na == nb:
        raise ValueError('Equal-size supports pass containment; ask which mask to keep.')
    small, large = (a_id, b_id) if na < nb else (b_id, a_id)
    return dict(pair_id=f'{min(a_id,b_id)}_{max(a_id,b_id)}', a=a_id, b=b_id,
                count_a=na, count_b=nb, intersection=common,
                coverage_a=common / na, coverage_b=common / nb,
                small=small, large=large, equal_size=na == nb,
                small_count=min(na, nb), large_count=max(na, nb),
                containment=common / min(na, nb),
                large_coverage=common / max(na, nb),
                original_iou=common / (na + nb - common), passes=bool(passes))


def decide(ids, pairs):
    """Only direct qualifying evidence; no inferred transitive containment."""
    excluded = {p['small'] for p in pairs if p['passes']}
    if not excluded <= set(ids):
        raise ValueError('Pair refers to a seed outside input scope.')
    return sorted(excluded), sorted(set(ids) - excluded)

"""Two-source-view surface comparison; no semantic or instance-label decisions."""
import numpy as np


def pair_metrics(a, b, a_visible_from_b, b_visible_from_a):
    """Inputs are unique, sorted fixed-surface vertex IDs, not 2D pixels."""
    na, nb = len(a), len(b)
    va, vb = len(a_visible_from_b), len(b_visible_from_a)
    common = len(np.intersect1d(a_visible_from_b, b_visible_from_a, assume_unique=True))
    original_common = len(np.intersect1d(a, b, assume_unique=True))
    union = va + vb - common
    original_union = na + nb - original_common
    guards = va >= 20 and vb >= 20 and 10 * va >= 3 * na and 10 * vb >= 3 * nb
    passes = guards and union > 0 and 5 * common >= 4 * union
    return dict(count_a=na, count_b=nb, visible_a=va, visible_b=vb,
                fraction_a=va / na if na else 0., fraction_b=vb / nb if nb else 0.,
                intersection=common, union=union,
                original_intersection=original_common,
                original_iou=original_common / original_union if original_union else 0.,
                score=common / union if union else 0., evidence_sufficient=bool(guards),
                duplicate=bool(passes), reason='duplicate' if passes else
                ('insufficient_visibility' if not guards else 'overlap_below_threshold'))


def order_key(seed):
    return (-seed['support_count'], seed['frame'], seed['representative']['model'], seed['digest'])


def fixed_primary_groups(ordered_ids, passes):
    remaining = list(ordered_ids)
    groups = []
    while remaining:
        primary = remaining[0]
        members, rest = [primary], []
        for candidate in remaining[1:]:
            (members if passes(primary, candidate) else rest).append(candidate)
        groups.append(members)
        remaining = rest
    return groups

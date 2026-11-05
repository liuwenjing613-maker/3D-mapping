"""Ordinary 3D IoU > 0.25 with fixed-primary grouping; no extra gates."""
def same_instance(intersection, count_a, count_b):
    union = count_a + count_b - intersection
    return union > 0 and 4 * intersection > union

def group_seeds(ordered_ids,passes):
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

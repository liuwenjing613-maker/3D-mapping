"""User rule, applied to frozen B coverage counts with strict boundaries."""


def excluded(intersection, support_count, instance_visible_count):
    if min(intersection, support_count, instance_visible_count) < 0:
        raise ValueError('Negative point count.')
    if intersection > min(support_count, instance_visible_count):
        raise ValueError('Intersection exceeds denominator.')
    return bool(support_count > 0 and instance_visible_count > 0
                and 10 * intersection > support_count
                and 5 * intersection < instance_visible_count)

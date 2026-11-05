"""Anchor Gate V1 plus user-requested removal of other-seed acceptance veto."""

def visibility_metrics(sample_count, visible_count):
    if not 0 <= visible_count <= sample_count:
        raise ValueError('Visible anchors must be a subset of sampled seed anchors')
    return {'seed_anchor_count':int(sample_count),
            'visible_seed_fraction':visible_count/sample_count if sample_count else 0.0}

def object_reliability(metrics, policy):
    reasons=[]
    if (metrics['visible_seed_samples'] < policy['minimum_visible_seed_surface_samples_each_target'] or
        metrics['visible_seed_fraction'] < policy['minimum_visible_seed_fraction_each_target']):
        reasons.append('insufficient_seed_visibility')
    if metrics['tracked_coverage'] < policy['minimum_tracked_coverage_of_visible_seed_samples_each_target']:
        reasons.append('anchor_coverage')
    if metrics['new_surface_bbox_fraction'] < policy['minimum_new_surface_support_fraction_inside_target_bbox']:
        reasons.append('extreme_spatial_drift')
    # Foreign-old-family overlap remains diagnostic only; it no longer vetoes acceptance.
    # Other-seed overlap is recorded by metrics_for, but does not veto acceptance.
    # Pixel conflict abstention and old-observation retirement rules are unchanged.
    return not reasons,reasons

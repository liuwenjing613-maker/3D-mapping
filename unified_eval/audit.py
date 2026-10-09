"""Secondary audit tables; these functions never change the primary scoring policy."""
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree

from .evaluate import AP_THRESHOLDS
from .metrics import build_overlap, average_precision, instance_precision_recall_f1, panoptic_quality
from .schema import EvaluationError


def native_nearest_reference_region_support(native_xyz, ref_xyz, regions, max_distance_m):
    """Alternative boundary audit: classify the closest surface without TARGET priority."""
    native, ref, regions = np.asarray(native_xyz), np.asarray(ref_xyz), np.asarray(regions)
    if any(x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() for x in (native, ref)):
        raise EvaluationError('Boundary support requires finite [N,3] geometry')
    if regions.shape != (len(ref),) or not np.isin(regions, [0, 1, 2]).all():
        raise EvaluationError('Boundary support requires explicit reference regions')
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise EvaluationError('Boundary support distance must be finite and positive')
    support = np.full(len(native), -1, dtype=np.int8)
    if not len(native) or not len(ref):
        return support
    unique, first, inverse = np.unique(ref, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    low, high = np.full(len(unique), 3, dtype=np.int8), np.full(len(unique), -1, dtype=np.int8)
    np.minimum.at(low, inverse, regions)
    np.maximum.at(high, inverse, regions)
    unique_regions = np.where(low == high, low, 0)
    distance, index = cKDTree(unique).query(native, k=1, workers=-1)
    valid = distance < max_distance_m
    support[valid] = unique_regions[index[valid]]
    return support


def _quality(overlaps, predictions, protocol):
    pr = [instance_precision_recall_f1(x, protocol.ignore_unmatched_pred_void_fraction_gt) for x in overlaps]
    pq = [panoptic_quality(x, p.is_partition, protocol.ignore_unmatched_pred_void_fraction_gt)
          for x, p in zip(overlaps, predictions)]
    tp, fp, fn = (sum(x[key] for x in pr) for key in ('TP', 'FP', 'FN'))
    denominator = sum(x['TP'] + .5 * x['FP'] + .5 * x['FN'] for x in pq)
    ap = {f'{t:.2f}': average_precision(overlaps, predictions, protocol.confidence_mode, t,
        protocol.ignore_unmatched_pred_void_fraction_gt)['ap'] for t in [.25] + AP_THRESHOLDS}
    defined = [ap[f'{t:.2f}'] for t in AP_THRESHOLDS if ap[f'{t:.2f}'] is not None]
    return {'CA_PQ': sum((x['SQ'] or 0) * x['TP'] for x in pq) / denominator if denominator else None,
        'CA_F1_0_5': 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        'CA_AP50_uniform': ap['0.50'], 'CA_AP_uniform': sum(defined) / len(defined) if defined else None,
        'TP': tp, 'FP': fp, 'FN': fn,
        'ignored_prediction_count': sum(x['ignored_prediction_count'] for x in pr)}


def background_only_fp_sensitivity(scenes, protocol):
    """Score both declared audit policies without altering predictions or the config."""
    overlaps = [build_overlap(gt, pred, protocol) for gt, pred in scenes]
    changed, counts = [], []
    for overlap in overlaps:
        mask = np.asarray([x == 'IGNORE_BACKGROUND_ONLY_REPORT' for x in overlap.unmatched_policy])
        clone = replace(overlap, unmatched_ignore_mask=overlap.unmatched_ignore_mask.copy(),
                        unmatched_policy=list(overlap.unmatched_policy))
        clone.unmatched_ignore_mask[mask] = False
        for index in np.flatnonzero(mask):
            clone.unmatched_policy[index] = 'FP_IF_UNMATCHED_SECONDARY_BACKGROUND_SENSITIVITY'
        changed.append(clone)
        counts.append(int(mask.sum()))
    predictions = [pred for _, pred in scenes]
    return {'status': 'SECONDARY_SENSITIVITY_NOT_PRIMARY_PROTOCOL', 'background_only_candidate_count': sum(counts),
        'per_scene_background_only_count': counts,
        'ignore_background_only': _quality(overlaps, predictions, protocol),
        'count_all_background_only_as_FP': _quality(changed, predictions, protocol),
        'prediction_type_source': 'native_declaration_only_otherwise_unknown; no GT completion'}


def distance_distribution(distances):
    values = np.asarray(distances)
    finite = values[np.isfinite(values)]
    return {'count': len(values), 'nonfinite_count': int(len(values) - len(finite)),
        **{name: float(np.percentile(finite, percentile)) if len(finite) else None
           for name, percentile in [('min_m', 0), ('p25_m', 25), ('median_m', 50), ('p90_m', 90),
                                    ('p95_m', 95), ('p99_m', 99), ('max_m', 100)]}}

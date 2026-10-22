"""Whole-mask semantic decisions, with no human selection or GT input."""
import copy
import numpy as np


def resolve_labels(id2label, config):
    labels = {int(k): str(v).strip() for k, v in id2label.items()}
    if sorted(labels) != list(range(len(labels))) or len(set(labels.values())) != len(labels):
        raise ValueError('Model label mapping must be contiguous and unique')
    if set(config['exclude_classes']) != {'wall', 'floor', 'ceiling', 'rug'}:
        raise ValueError('Excluded classes do not match the authorized scope')
    by_name = {v: k for k, v in labels.items()}
    missing = set(config['exclude_classes']) - set(by_name)
    if missing:
        raise ValueError('Missing model classes: ' + repr(missing))
    for key in ['pixel_confidence', 'mask_fraction']:
        if not 0.5 < config[key] <= 1:
            raise ValueError('Threshold must be in (0.5, 1]')
    return labels, [by_name[n] for n in config['exclude_classes']]


def summarize_mask(prediction, confidence, mask, id2label, config):
    labels, excluded_ids = resolve_labels(id2label, config)
    if mask.dtype != np.bool_ or mask.ndim != 2 or prediction.shape != mask.shape or confidence.shape != mask.shape:
        raise ValueError('Prediction, confidence and original boolean mask must align exactly')
    area = int(mask.sum())
    if area == 0:
        raise ValueError('Empty seed mask')
    if not np.issubdtype(prediction.dtype, np.integer) or prediction.min() < 0 or prediction.max() >= len(labels):
        raise ValueError('Prediction has invalid label IDs')
    if not np.all(np.isfinite(confidence)) or confidence.min() < 0 or confidence.max() > 1:
        raise ValueError('Invalid pixel confidence')
    ids, conf = prediction[mask], confidence[mask]
    high = conf >= config['pixel_confidence']
    counts = np.bincount(ids, minlength=len(labels))
    high_counts = np.bincount(ids[high], minlength=len(labels))
    excluded_count = int(counts[excluded_ids].sum())
    high_excluded = int(high_counts[excluded_ids].sum())
    high_other = int(high_counts.sum()) - high_excluded
    low = area - int(high_counts.sum())
    assert high_excluded + high_other + low == area
    threshold = config['mask_fraction']
    decision = ('suggest_exclude' if high_excluded / area >= threshold else
                'retain' if high_other / area >= threshold else 'review')
    categories = [{'id': k, 'name': labels[k], 'pixels': int(counts[k]),
                   'fraction': float(counts[k]/area), 'high_confidence_pixels': int(high_counts[k]),
                   'high_confidence_fraction': float(high_counts[k]/area)} for k in range(len(labels))]
    top = sorted(categories, key=lambda x: (-x['pixels'], x['id']))
    sensitivity = {str(t): int(((conf >= t) & np.isin(ids, excluded_ids)).sum()) / area
                   for t in config.get('sensitivity_pixel_thresholds', [])}
    return {'area': area, 'exclude': decision == 'suggest_exclude', 'decision': decision,
            'top_class': top[0]['name'], 'top_class_fraction': top[0]['fraction'],
            'excluded_fraction': excluded_count / area,
            'high_excluded_fraction': high_excluded / area,
            'high_retained_fraction': high_other / area, 'low_confidence_fraction': low / area,
            'high_excluded_pixels': high_excluded, 'high_retained_pixels': high_other,
            'low_confidence_pixels': low, 'mean_top_probability': float(conf.mean()),
            'categories': categories, 'sensitivity_high_excluded_fraction': sensitivity}


def validate_records(source, records):
    expected = [g['id'] for g in source['groups']]
    if len(set(expected)) != len(expected) or [r['group_id'] for r in records] != expected:
        raise ValueError('Missing, reordered or duplicate seed groups')
    for original_id, (group, rec) in enumerate(zip(source['groups'], records), 1):
        p = group['primary']
        if (rec['original_track_id'], rec['mask_id'], rec['frame'], rec['mask_sha256'], rec['area']) != (
                original_id, p['id'], p['frame'], p['mask_sha256'], p['area']):
            raise ValueError('Final seed identity or pixels changed: ' + group['id'])


def filtered_manifest(source, records, source_sha, classification_sha, config):
    validate_records(source, records)
    result = copy.deepcopy(source)
    result['groups'] = []
    for group, rec in zip(source['groups'], records):
        if not rec['exclude']:
            group = copy.deepcopy(group)
            group['original_track_id'] = rec['original_track_id']
            result['groups'].append(group)
    result.update(status='semantic_screening_only', tracking_requested=False, map_write=False,
                  source_manifest_sha256=source_sha, classification_sha256=classification_sha,
                  filter_config=copy.deepcopy(config), original_track_ids_preserved=True,
                  excluded_group_ids=[r['group_id'] for r in records if r['exclude']],
                  uncertain_group_ids=[r['group_id'] for r in records if r['decision']=='review'],
                  scope='Suggested whole-mask exclusions; uncertain retained; no deduplication or E1 run')
    return result

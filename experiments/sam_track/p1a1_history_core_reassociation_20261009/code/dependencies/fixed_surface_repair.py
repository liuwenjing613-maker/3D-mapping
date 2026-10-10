"""Fixed-geometry diagnostics and bounded final-label commits.

These utilities do not change masks, votes, confirmation rules or v3 scoring.
The full TSDF supplies the nearest-neighbour candidates, including unlabeled points.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def allowed_mask(indices, point_count):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('Allowed surface IDs must be a one-dimensional integer array')
    if len(indices) and (indices.min() < 0 or indices.max() >= point_count):
        raise ValueError('Allowed surface ID outside the frozen TSDF')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Allowed surface IDs must be unique')
    mask = np.zeros(point_count, dtype=bool)
    mask[indices] = True
    return mask


def require_same_geometry(baseline, candidate):
    for field in ('xyz_m', 'rgb'):
        if field not in baseline or field not in candidate:
            raise ValueError('Missing frozen surface field: ' + field)
        if array_sha256(baseline[field]) != array_sha256(candidate[field]):
            raise ValueError('Frozen surface changed: ' + field)
    n = len(baseline['xyz_m'])
    for surface in (baseline, candidate):
        labels = surface['instance_id']
        if labels.shape != (n,) or labels.dtype.kind != 'i':
            raise ValueError('Instance labels must be one signed integer per frozen vertex')
    if baseline['instance_id'].dtype != candidate['instance_id'].dtype:
        raise ValueError('Instance-label dtype changed')


def bounded_final_labels(baseline, candidate, indices):
    """Keep candidate labels inside the declared set and baseline labels outside."""
    require_same_geometry(baseline, candidate)
    mask = allowed_mask(indices, len(baseline['xyz_m']))
    before = baseline['instance_id']
    proposed = candidate['instance_id']
    result = before.copy()
    result[mask] = proposed[mask]
    if not np.array_equal(result[~mask], before[~mask]):
        raise AssertionError('Final labels escaped the allowed set')
    if not np.array_equal(result[mask], proposed[mask]):
        raise AssertionError('An allowed candidate label was modified')
    blocked = np.flatnonzero(~mask & (proposed != before)).astype(np.int32)
    return result, {
        'allowed_surface_points': int(mask.sum()),
        'candidate_outside_changes_blocked': len(blocked),
        'final_outside_changes': int(np.sum(~mask & (result != before))),
        'final_inside_changes': int(np.sum(mask & (result != before))),
        'inside_labels_identical_to_unrestricted_candidate': True,
        'outside_labels_bit_identical_to_baseline': True,
    }, blocked


def verify_raw_outside(baseline_evidence, repaired_evidence, baseline_pairs, repaired_pairs, indices):
    """Audit evidence fields and the complete vote ledger outside the allowed set."""
    n = len(baseline_evidence['state'])
    mask = allowed_mask(indices, n)
    metadata_fields = {'map_version', 'association_decisions_sha256'}
    before_fields = set(baseline_evidence) - metadata_fields
    after_fields = set(repaired_evidence) - metadata_fields
    if before_fields != after_fields:
        raise ValueError('Per-point surface evidence fields changed')
    for field in before_fields:
        if baseline_evidence[field].shape[:1] != (n,) or repaired_evidence[field].shape[:1] != (n,):
            raise ValueError('Invalid per-point evidence shape: ' + field)
        if array_sha256(baseline_evidence[field][~mask]) != array_sha256(repaired_evidence[field][~mask]):
            raise ValueError('Raw evidence changed outside the allowed set: ' + field)
    for field in metadata_fields & (set(baseline_evidence) | set(repaired_evidence)):
        for evidence in (baseline_evidence, repaired_evidence):
            if field in evidence and evidence[field].shape != (1,):
                raise ValueError('Invalid whole-map metadata shape: ' + field)
        if field in baseline_evidence and field in repaired_evidence and array_sha256(baseline_evidence[field]) != array_sha256(repaired_evidence[field]):
            raise ValueError('Whole-map metadata differs: ' + field)
    for pairs in (baseline_pairs, repaired_pairs):
        if set(pairs) - metadata_fields != {'surface_point_index', 'instance_id', 'frame_votes'}:
            raise ValueError('Unexpected vote ledger fields')
        for field in metadata_fields & set(pairs):
            if pairs[field].shape != (1,):
                raise ValueError('Invalid vote-ledger metadata shape: ' + field)
    for field in metadata_fields & set(baseline_pairs) & set(repaired_pairs):
        if array_sha256(baseline_pairs[field]) != array_sha256(repaired_pairs[field]):
            raise ValueError('Vote-ledger metadata differs: ' + field)
    selected = [~mask[pairs['surface_point_index']] for pairs in (baseline_pairs, repaired_pairs)]
    # A canonical ordering allows storage order to differ while preserving every vote.
    canonical = []
    for pairs, keep in zip((baseline_pairs, repaired_pairs), selected):
        rows = np.c_[pairs['surface_point_index'][keep], pairs['instance_id'][keep], pairs['frame_votes'][keep]]
        canonical.append(rows[np.lexsort((rows[:, 1], rows[:, 0]))])
    if not np.array_equal(*canonical):
        raise ValueError('Vote ledger changed outside the allowed set')
    return {'all_per_point_raw_evidence_fields_unchanged_outside': True,
            'verified_per_point_fields': sorted(before_fields),
            'storage_metadata_difference': {
                'baseline_only': sorted(set(baseline_evidence) - set(repaired_evidence)),
                'repaired_only': sorted(set(repaired_evidence) - set(baseline_evidence)),
                'vote_ledger_baseline_only': sorted(set(baseline_pairs) - set(repaired_pairs)),
                'vote_ledger_repaired_only': sorted(set(repaired_pairs) - set(baseline_pairs)),
                'baseline_metadata_reference': {field: baseline_evidence[field].tolist()
                                                for field in metadata_fields if field in baseline_evidence}},
            'all_vote_counts_unchanged_outside': True,
            'outside_vote_pairs_checked': len(canonical[0])}


def fixed_mapping(path, native_xyz, reference_xyz, max_distance_m, workers=8):
    """Create once or reuse a geometry-only correspondence; labels are not inputs."""
    native_xyz, reference_xyz = np.asarray(native_xyz), np.asarray(reference_xyz)
    for name, xyz in [('native', native_xyz), ('reference', reference_xyz)]:
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.all(np.isfinite(xyz)):
            raise ValueError('Invalid ' + name + ' geometry')
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError('Invalid diagnostic distance')
    expected = {'schema_version': 1, 'native_geometry_sha256': array_sha256(native_xyz),
                'reference_geometry_sha256': array_sha256(reference_xyz),
                'native_point_count': len(native_xyz), 'reference_point_count': len(reference_xyz),
                'max_distance_m': float(max_distance_m), 'distance_comparison': 'strict_less_than',
                'mapping_direction': 'reference_to_nearest_full_TSDF_vertex',
                'unassigned_native_vertices_remain_candidates': True}
    path = Path(path)
    if path.exists():
        archive = load_arrays(path)
        metadata = json.loads(str(archive['metadata_json'].item()))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('Cached geometry or diagnostic settings changed; create a separate cache')
        nearest, distance = archive['nearest_surface_index'], archive['distance_m']
        if nearest.shape != (len(reference_xyz),) or distance.shape != nearest.shape:
            raise ValueError('Invalid cached mapping shape')
        if nearest.dtype.kind not in 'iu' or np.any(nearest < 0) or np.any(nearest >= len(native_xyz)):
            raise ValueError('Invalid cached surface index')
        if not np.all(np.isfinite(distance)) or np.any(distance < 0):
            raise ValueError('Invalid cached mapping distance')
        if metadata.get('nearest_sha256') != array_sha256(nearest) or metadata.get('distance_sha256') != array_sha256(distance):
            raise ValueError('Cached correspondence arrays failed integrity verification')
    else:
        distance, nearest = cKDTree(native_xyz.astype(np.float64)).query(reference_xyz.astype(np.float64), workers=workers)
        nearest = nearest.astype(np.int32)
        metadata = {**expected, 'nearest_sha256': array_sha256(nearest), 'distance_sha256': array_sha256(distance)}
        atomic_npz(path, nearest_surface_index=nearest, distance_m=distance,
                   metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return {'nearest_surface_index': nearest, 'distance_m': distance,
            'reachable': distance < max_distance_m, 'metadata': metadata, 'cache_sha256': file_sha256(path)}


def mapped_labels(mapping, native_labels):
    native_labels = np.asarray(native_labels)
    if native_labels.shape != (mapping['metadata']['native_point_count'],) or native_labels.dtype.kind != 'i':
        raise ValueError('Labels do not match the frozen native surface')
    labels = native_labels[mapping['nearest_surface_index']].copy()
    labels[(labels <= 0) | ~mapping['reachable']] = -1
    return labels


def target_diagnostics(mapping, native_labels, truth, valid, target_identity):
    """Supplementary reference-point diagnostics, not a replacement v3 protocol."""
    labels = mapped_labels(mapping, native_labels)
    truth, valid = np.asarray(truth), np.asarray(valid, dtype=bool)
    if truth.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError('GT fields do not match the frozen reference')
    identity = {int(pid): int(gid) for pid, gid in target_identity.items()}
    if any(pid <= 0 for pid in identity) or len(set(identity.values())) != len(identity):
        raise ValueError('Target identity must explicitly map distinct positive IDs to distinct GTs')
    all_ids, all_counts = np.unique(labels[valid & (labels > 0)], return_counts=True)
    pred_sizes = {int(pid): int(count) for pid, count in zip(all_ids, all_counts)}
    targets = []
    for pid, gid in sorted(identity.items()):
        selected = valid & (truth == gid)
        reached = selected & mapping['reachable']
        total, reachable = int(selected.sum()), int(reached.sum())
        if not total:
            raise ValueError('Target has no valid GT reference points')
        ids, counts = np.unique(labels[reached], return_counts=True)
        intersections = {int(i): int(c) for i, c in zip(ids, counts) if i > 0}
        correct = intersections.get(pid, 0)
        other_target = sum(c for i, c in intersections.items() if i in identity and i != pid)
        other_positive = sum(c for i, c in intersections.items() if i not in identity)
        best = max(intersections, key=lambda i: (intersections[i] / (total + pred_sizes[i] - intersections[i]), -i)) if intersections else None
        targets.append({'GT_id': gid, 'expected_persistent_id': pid, 'GT_reference_points': total,
                        'reachable_fixed_surface_reference_points': reachable,
                        'no_geometric_correspondence_reference_points': total - reachable,
                        'expected_identity_reference_points': correct,
                        'other_target_identity_reference_points': other_target,
                        'other_positive_identity_reference_points': other_positive,
                        'unassigned_reachable_reference_points': int(np.sum(reached & (labels <= 0))),
                        'reachable_label_counts': {str(int(i)): int(c) for i, c in zip(ids, counts)},
                        'expected_identity_full_GT_coverage': correct / total,
                        'expected_identity_reachable_GT_coverage': correct / reachable if reachable else 0.,
                        'expected_identity_reference_purity': correct / pred_sizes[pid] if pid in pred_sizes else 0.,
                        'expected_identity_fixed_IoU': correct / (total + pred_sizes.get(pid, 0) - correct),
                        'best_fixed_IoU_prediction_id': best,
                        'best_fixed_IoU': intersections[best] / (total + pred_sizes[best] - intersections[best]) if best else 0.,
                        'best_prediction_full_GT_coverage': intersections[best] / total if best else 0.,
                        'best_prediction_reference_purity': intersections[best] / pred_sizes[best] if best else 0.})
    return {'GT_valid_reference_points': int(valid.sum()),
            'fixed_reachable_valid_reference_points': int(np.sum(valid & mapping['reachable'])),
            'assigned_reachable_valid_reference_points': int(np.sum(valid & (labels > 0))),
            'unassigned_reachable_valid_reference_points': int(np.sum(valid & mapping['reachable'] & (labels <= 0))),
            'target_identity_used_only_for_posthoc_diagnostics': identity, 'targets': targets}


def target_transitions(mapping, before_labels, after_labels, truth, valid, target_identity):
    before, after = mapped_labels(mapping, before_labels), mapped_labels(mapping, after_labels)
    rows = []
    for pid, gid in sorted((int(p), int(g)) for p, g in target_identity.items()):
        selected = np.asarray(valid, dtype=bool) & (np.asarray(truth) == gid) & mapping['reachable']
        b, a = before[selected], after[selected]
        rows.append({'GT_id': gid, 'expected_persistent_id': pid,
                     'wrong_positive_to_correct': int(np.sum((b > 0) & (b != pid) & (a == pid))),
                     'unassigned_to_correct': int(np.sum((b <= 0) & (a == pid))),
                     'correct_to_unassigned': int(np.sum((b == pid) & (a <= 0))),
                     'correct_to_other_positive': int(np.sum((b == pid) & (a > 0) & (a != pid))),
                     'wrong_positive_to_unassigned': int(np.sum((b > 0) & (b != pid) & (a <= 0))),
                     'unassigned_to_other_positive': int(np.sum((b <= 0) & (a > 0) & (a != pid))),
                     'previously_assigned_now_unassigned': int(np.sum((b > 0) & (a <= 0))),
                     'previously_unassigned_now_assigned': int(np.sum((b <= 0) & (a > 0)))})
    return rows

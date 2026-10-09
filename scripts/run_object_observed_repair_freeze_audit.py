"""Rescore immutable exports under review revision 2; leave the 05f6bb0 baseline intact."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import copy
import csv
import json
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_eval.audit import background_only_fp_sensitivity, distance_distribution, native_nearest_reference_region_support
from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import (attach_reference_support, load_fixed_surface_correspondence,
    map_fixed_surface_labels, native_reference_region_support)
from unified_eval.io import load_gt, load_prediction, repair_evaluator_code_hashes, sha256_array, sha256_file
from unified_eval.metrics import build_overlap, significant_structure_diagnostics
from unified_eval.repair_profile import adapt_surface, load_surface_export, paired_revision_metrics, write_json
from unified_eval.schema import CanonicalInstance, CanonicalPrediction, EvaluationError, Protocol

SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
METHODS = ['P1-A1_native', 'P1-A1_holes_geodesic', 'P1-A1_auto_v2', 'OVI-MAP_full400']


def verify(path, digest):
    if sha256_file(path) != digest:
        raise EvaluationError('Locked source/artifact changed: ' + str(path))


def metadata(protocol, config):
    return {'profile_revision': protocol.profile_revision, 'profile_semantics_sha256': protocol.profile_semantics_sha256,
        'profile_config_sha256': sha256_file(config), 'evaluator_code_sha256': repair_evaluator_code_hashes()}


def oracle_audit(gt, xyz, cache, protocol, config):
    target = gt.evaluation_region == 1
    native_distance, native_ref = cKDTree(gt.xyz_ref).query(xyz, k=1, workers=4)
    labels = gt.instance_id[native_ref].copy()
    labels[native_distance >= protocol.geometry_mapping_max_distance_m] = -1
    mapped = map_fixed_surface_labels(xyz, labels, gt.xyz_ref, cache, scene_id=gt.scene_id,
        method_name='GEOMETRY_ORACLE_TEST_ONLY', method_commit='not_a_research_method',
        duplicate_coordinate_policy=protocol.duplicate_coordinate_policy,
        metadata={**metadata(protocol, config), 'oracle_uses_GT_to_construct_test_labels': True})
    support = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, protocol.geometry_mapping_max_distance_m)
    attach_reference_support(mapped, labels, support, gt, cache)
    summary = evaluate_scenes([(gt, mapped.prediction)], protocol)[0]
    nearest = cache.nearest_native_index
    read = np.full(len(nearest), -1, dtype=np.int64)
    valid = nearest >= 0
    read[valid] = labels[nearest[valid]]
    direct_correct = target & (read == gt.instance_id)
    no_geometry = target & ~valid
    unassigned = target & valid & (read <= 0)
    wrong = target & valid & (read > 0) & (read != gt.instance_id)
    assert np.array_equal(target, direct_correct | no_geometry | unassigned | wrong)
    rows = []
    for raw in np.unique(gt.instance_id[target]):
        selected = target & (gt.instance_id == raw)
        rows.append({'raw_gt_id': int(raw), 'observed_target_vertices': int(selected.sum()),
            'direct_same_GT_owner_vertices': int(np.count_nonzero(selected & direct_correct)),
            'no_geometry_vertices': int(np.count_nonzero(selected & no_geometry)),
            'unassigned_from_nearest_ignored_or_distant_GT_vertices': int(np.count_nonzero(selected & unassigned)),
            'wrong_GT_boundary_transfer_vertices': int(np.count_nonzero(selected & wrong)),
            'distance': distance_distribution(cache.distances_m[selected])})
    worst = sorted(rows, key=lambda row: (row['no_geometry_vertices'] + row['unassigned_from_nearest_ignored_or_distant_GT_vertices']
        + row['wrong_GT_boundary_transfer_vertices']) / row['observed_target_vertices'], reverse=True)[:10]
    unique_xyz = xyz[cache.unique_first_index]
    tree = cKDTree(unique_xyz)
    shifts = []
    for mm in (1, 3, 5, 10, 20):
        distance, _ = tree.query(gt.xyz_ref[target] - np.array([mm / 1000, 0, 0], dtype=xyz.dtype), k=1, workers=4)
        shifts.append({'translation_x_mm': mm, 'target_geometry_coverage': float(np.mean(distance < .01)),
                       'distance': distance_distribution(distance)})
    return {'status': 'SECONDARY_GEOMETRY_ORACLE_NOT_A_METHOD', 'GT_labels_are_never_written_to_method_exports': True,
        'geometry_xyz_sha256': cache.native_xyz_sha256, 'correspondence_sha256': cache.sha256,
        'summary': summary, 'direct_GT_id_residual_breakdown': {'correct': int(direct_correct.sum()),
            'no_geometry': int(no_geometry.sum()), 'unassigned_ignored_or_distant_GT': int(unassigned.sum()),
            'wrong_GT_boundary_transfer': int(wrong.sum()), 'target_vertices': int(target.sum()),
            'note': 'Raw GT-ID transfer diagnostic; owner coverage separately uses optimal one-to-one alignment.'},
        'target_distance': distance_distribution(cache.distances_m[target]),
        'known_non_target_distance': distance_distribution(cache.distances_m[gt.evaluation_region == 2]),
        'predeclared_geometry_gate_sensitivity': [{'gate_m': gate, 'target_geometry_coverage': float(np.mean(cache.distances_m[target] < gate))}
            for gate in (.005, .01, .02)], 'predeclared_translation_sensitivity': shifts, 'per_GT': rows, 'worst_GT': worst}


def boundary_audit(gt, xyz, labels, prediction, protocol):
    nearest = native_nearest_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, protocol.geometry_mapping_max_distance_m)
    alternative = copy.deepcopy(prediction)
    rows = []
    for instance, changed in zip(prediction.instances, alternative.instances):
        native_id = instance.metadata.get('normalized_positive_instance_id', instance.metadata['native_instance_id'])
        selected = labels == native_id
        counts = {name: int(np.count_nonzero(selected & (nearest == region)))
                  for region, name in [(1, 'target'), (2, 'known_non_target'), (0, 'ignore'), (-1, 'outside')]}
        for name, count in counts.items():
            changed.metadata['native_' + name + '_support_count'] = count
        regions = gt.evaluation_region[instance.vertex_indices]
        if not np.any(regions != 0) and instance.metadata.get('native_target_support_count', 0) and counts['target'] == 0:
            rows.append({'instance_uid': instance.instance_uid, 'projected_vertices': len(instance.vertex_indices),
                'priority_target_support': instance.metadata['native_target_support_count'], 'nearest_surface_support': counts,
                'native_prediction_type': instance.metadata.get('prediction_type', 'unknown')})
    primary = evaluate_scenes([(gt, prediction)], protocol)[0]
    secondary = evaluate_scenes([(gt, alternative)], protocol)[0]
    return {'status': 'SECONDARY_BOUNDARY_SENSITIVITY_PRIMARY_POLICY_UNCHANGED',
        'target_priority_without_closest_target_empty_projection_count': len(rows), 'cases': rows,
        'primary_FP': primary['CA_PRF1_0_5']['FP'], 'nearest_surface_alternative_FP': secondary['CA_PRF1_0_5']['FP'],
        'primary_PQ': primary['CA_PQ']['PQ'], 'nearest_surface_alternative_PQ': secondary['CA_PQ']['PQ']}


def structure_review_cases(gt, pred, diag, protocol):
    main, auxiliary = build_overlap(gt, pred, protocol), build_overlap(gt, diag, protocol)
    ms = (main.intersection >= 10) & (main.recall >= .05)
    ds = (auxiliary.intersection >= 10) & (auxiliary.recall >= .05)
    rows, used = [], set()
    def add(kind, ids, uid=None):
        key = (kind, tuple(ids))
        if key not in used:
            used.add(key)
            rows.append({'scene_id': gt.scene_id, 'candidate_type': kind, 'raw_gt_ids': ids, 'prediction_uid': uid,
                'review_status': 'PENDING_VISUAL_REVIEW', 'selection': 'deterministic first IDs, no manual cherry-picking'})
    for row in np.flatnonzero(ms.sum(axis=1) >= 2)[:3]:
        add('MAIN_MERGE_CANDIDATE', main.gt_ids[ms[row]].astype(int).tolist(), main.pred_uids[row])
    for col in np.flatnonzero(ms.sum(axis=0) >= 2)[:2]:
        add('MAIN_SPLIT_CANDIDATE', [int(main.gt_ids[col])])
    for row in np.flatnonzero((ds.sum(axis=1) >= 2) & (ms.sum(axis=1) < 2))[:2]:
        add('AUXILIARY_ONLY_MERGE_BOUNDARY', main.gt_ids[ds[row]].astype(int).tolist(), main.pred_uids[row])
    for col in np.flatnonzero((main.gt_size < 100))[:1]:
        add('SMALL_OBJECT_THRESHOLD_REVIEW', [int(main.gt_ids[col])])
    clean = np.flatnonzero((ms.sum(axis=0) == 1) & (ds.sum(axis=0) > 1))
    for col in clean[:2]:
        add('MAIN_SEPARATED_AUXILIARY_SPLIT', [int(main.gt_ids[col])])
    for col in np.argsort(main.gt_ids):
        if len(rows) >= 10:
            break
        add('PRIMARY_PARTITION_CONTROL', [int(main.gt_ids[col])])
    return rows[:10]


def audit_scene(arguments):
    scene, args = arguments
    out, old = args.output_root / scene, args.baseline_root / 'mesh' / scene
    out.mkdir(parents=True, exist_ok=True)
    config = args.config
    protocol = Protocol.from_dict(json.loads(config.read_text()))
    gt = load_gt(old / 'gt.npz')
    direct = CanonicalPrediction(scene, gt.vertex_count, [CanonicalInstance(str(int(raw)),
        np.flatnonzero(gt.instance_id == raw).astype(np.int32)) for raw in np.unique(gt.instance_id[gt.evaluation_region == 1])],
        'DISCRETE_GT_ORACLE_TEST_ONLY', 'not_a_research_method', 'direct_reference_oracle', protocol.name, True,
        {**gt.metadata, **metadata(protocol, config), 'evaluation_only_discrete_gt_oracle': True,
         'no_geometry_target_vertex_count': 0})
    direct_summary = evaluate_scenes([(gt, direct)], protocol)[0]
    if any(value != 1 for value in (direct_summary['CA_PQ']['PQ'], direct_summary['CA_PRF1_0_5']['F1'],
                                    direct_summary['CA_AP50_uniform'], direct_summary['CA_mCov'])):
        raise EvaluationError('Revision 2 discrete perfect GT oracle must remain ideal')
    write_json(out / 'discrete_GT_oracle.json', direct_summary)
    predictions, diagnostics, reports = {}, {}, {}
    for name in METHODS:
        manifest = json.loads((old / name / 'adapter_manifest.json').read_text())
        verify(manifest['source_surface'], manifest['source_surface_sha256'])
        verify(old / 'gt.npz', manifest['gt_file_sha256'])
        path = out / ('fixed_gt_to_ovimap.npz' if name.startswith('OVI') else 'fixed_gt_to_tsdf.npz')
        result = adapt_surface(manifest['source_surface'], old / 'gt.npz', config, path, out / name,
            method_name=name, method_commit=manifest['method_commit'], source_provenance=manifest['source_provenance'])
        predictions[name] = load_prediction(out / name / 'canonical_prediction.npz')
        diagnostics[name] = load_prediction(out / name / 'diagnostic_support_prediction.npz')
        reports[name] = result['summary']
        fp = background_only_fp_sensitivity([(gt, predictions[name])], protocol)
        write_json(out / name / 'background_FP_sensitivity.json', fp)
        main = build_overlap(gt, predictions[name], protocol)
        auxiliary = build_overlap(gt, diagnostics[name], protocol)
        sensitivity = {'status': 'SECONDARY_STRUCTURE_THRESHOLD_AUDIT', 'main_partition': {}, 'auxiliary_2cm': {}}
        for minimum in (1, 10):
            changed = replace(protocol, significant_min_intersection_vertices=minimum)
            sensitivity['main_partition'][str(minimum)] = significant_structure_diagnostics(main, changed)
            sensitivity['auxiliary_2cm'][str(minimum)] = significant_structure_diagnostics(auxiliary, changed)
        write_json(out / name / 'structure_sensitivity.json', sensitivity)
        print(json.dumps({'stage': 'method_scored', 'scene': scene, 'method': name, 'PQ': result['summary']['CA_PQ']['PQ']}), flush=True)
    for name in ('P1-A1_native', 'OVI-MAP_full400'):
        manifest = json.loads((out / name / 'adapter_manifest.json').read_text())
        xyz, labels, *_ = load_surface_export(manifest['source_surface'])
        cache = load_fixed_surface_correspondence(manifest['correspondence_cache'], xyz, gt.xyz_ref)
        oracle = oracle_audit(gt, xyz, cache, protocol, config)
        write_json(out / name / 'geometry_audit.json', oracle)
        write_json(out / name / 'boundary_support_sensitivity.json', boundary_audit(gt, xyz, labels, predictions[name], protocol))
        print(json.dumps({'stage': 'geometry_audited', 'scene': scene, 'method': name,
                          'oracle_PQ': oracle['summary']['CA_PQ']['PQ']}), flush=True)
    for before, after in zip(METHODS[:2], METHODS[1:3]):
        write_json(out / (before + '_to_' + after + '.paired.json'),
            paired_revision_metrics(gt, predictions[before], predictions[after], protocol))
    if scene in ('room0', 'room1'):
        write_json(out / 'structure_visual_review_cases.json', {'cases': structure_review_cases(gt,
            predictions['P1-A1_holes_geodesic'], diagnostics['P1-A1_holes_geodesic'], protocol)})
    if scene in ('room0', 'room2') and args.include_manual:
        for old_name in ('P1-A1_human_full_track', 'P1-A1_human_seed_only'):
            manifest = json.loads((args.manual_baseline_root / scene / old_name / 'adapter_manifest.json').read_text())
            verify(manifest['source_surface'], manifest['source_surface_sha256'])
            result = adapt_surface(manifest['source_surface'], old / 'gt.npz', config, out / 'fixed_gt_to_tsdf.npz', out / old_name,
                method_name=old_name, method_commit=manifest['method_commit'], source_provenance=manifest['source_provenance'])
            prediction = load_prediction(out / old_name / 'canonical_prediction.npz')
            write_json(out / (old_name + '.paired.json'), paired_revision_metrics(gt, predictions['P1-A1_holes_geodesic'], prediction, protocol))
    write_json(out / 'scene_complete.json', {'scene_id': scene, 'profile_revision': 2,
        'GT_scope_preserved': gt.metadata['gt_scope_sha256'], 'GT_observed_support_preserved': gt.metadata['gt_observed_support_sha256'],
        'GT_file_sha256': sha256_file(old / 'gt.npz'), 'methods': reports})
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009'))
    parser.add_argument('--manual-baseline-root', type=Path, default=Path('/data/chenkejun/CVPR/results/manual_v3_comparison_20261009'))
    parser.add_argument('--output-root', type=Path, default=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_audit_20261009'))
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json')
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--include-manual', action='store_true')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / 'development_baseline_lock.json'
    lock = json.loads(lock_path.read_text())
    if lock['code_commit'] != '05f6bb0d6c9a2491a51e990d217fd7f1397f5f6b' or lock['profile_revision'] != 1:
        raise EvaluationError('Freeze audit must refer to the exact locked development baseline')
    if Protocol.from_dict(json.loads(args.config.read_text())).profile_revision != 2:
        raise EvaluationError('Freeze audit requires the separately named review revision 2 config')
    progress = {'stage': 'running', 'completed_scenes': [], 'scene_ids': args.scenes, 'profile_revision': 2,
        'audit_code_commit': subprocess.check_output(['git', '-C', str(Path(__file__).parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
        'evaluator_code_sha256': repair_evaluator_code_hashes(), 'config_sha256': sha256_file(args.config)}
    progress['development_baseline_lock_sha256'] = sha256_file(lock_path)
    write_json(args.output_root / 'progress.json', progress)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(audit_scene, (scene, args)) for scene in args.scenes]
        for job in as_completed(jobs):
            progress['completed_scenes'].append(job.result())
            write_json(args.output_root / 'progress.json', progress)
    protocol = Protocol.from_dict(json.loads(args.config.read_text()))
    pooled, fp = {}, {}
    for name in METHODS:
        data = [(load_gt(args.baseline_root / 'mesh' / scene / 'gt.npz'),
                 load_prediction(args.output_root / scene / name / 'canonical_prediction.npz')) for scene in args.scenes]
        diag = [load_prediction(args.output_root / scene / name / 'diagnostic_support_prediction.npz') for scene in args.scenes]
        pooled[name] = evaluate_scenes(data, protocol, diagnostic_predictions=diag)[0]
        fp[name] = background_only_fp_sensitivity(data, protocol)
    write_json(args.output_root / 'pooled_metrics.review_r2.json', pooled)
    write_json(args.output_root / 'background_FP_sensitivity.review_r2.json', fp)
    if args.include_manual:
        manual_scenes = [scene for scene in args.scenes if scene in ('room0', 'room2')]
        comparison = {}
        for name in ('P1-A1_holes_geodesic', 'OVI-MAP_full400', 'P1-A1_human_full_track', 'P1-A1_human_seed_only'):
            data = [(load_gt(args.baseline_root / 'mesh' / scene / 'gt.npz'),
                     load_prediction(args.output_root / scene / name / 'canonical_prediction.npz')) for scene in manual_scenes]
            diag = [load_prediction(args.output_root / scene / name / 'diagnostic_support_prediction.npz') for scene in manual_scenes]
            comparison[name] = evaluate_scenes(data, protocol, diagnostic_predictions=diag)[0]
        write_json(args.output_root / 'manual_comparison.review_r2.json', {'scenes': manual_scenes, 'methods': comparison,
            'input_budget_note': 'Human repairs use additional masks and historical RGB; not equal supervision or input budget.'})
    progress['stage'] = 'complete'
    write_json(args.output_root / 'progress.json', progress)
    print(json.dumps({'stage': 'complete', 'scene_count': len(args.scenes)}), flush=True)


if __name__ == '__main__':
    main()

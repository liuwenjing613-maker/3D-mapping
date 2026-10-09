"""Audit fixed GT scope and existing maps without changing any map algorithm.

All outputs are development results. This script does not freeze thresholds or
silently update historical Replica-CA-v3 GT/results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import (attach_reference_support, build_fixed_surface_correspondence,
    load_fixed_surface_correspondence, map_fixed_surface_labels, native_reference_region_support)
from unified_eval.gt_scope import apply_observed_scope, build_gt_scope, coincident_label_ambiguity, scope_sha256
from unified_eval.io import load_gt, load_prediction, save_gt, sha256_array, sha256_file
from unified_eval.mesh_visibility import GTMeshVisibility
from unified_eval.metrics import build_overlap, significant_structure_diagnostics
from unified_eval.observable_surface import ObservationFrame, accumulate_observed_support, save_observed_support
from unified_eval.repair_profile import adapt_surface, paired_revision_metrics, write_json
from unified_eval.replica import SCENES, load_scope_source
from unified_eval.schema import CanonicalInstance, CanonicalPrediction, EvaluationError, Protocol


STATUS = 'DEVELOPMENT / NON_OFFICIAL / PROFILE_NOT_FROZEN'


def emit(stage, **values):
    print(json.dumps({'stage': stage, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def qualify(args, config):
    rows = []
    for scene in SCENES:
        source, manifest = load_scope_source(args.reference_root, scene)
        annotation = args.annotation_root / scene.replace('room', 'room_').replace('office', 'office_') / 'habitat/info_semantic.json'
        official = json.loads(annotation.read_text(encoding='utf-8'))
        objects = {int(x['id']): x for x in official['objects']}
        official_categories = {key: value['class_name'] for key, value in objects.items()
                               if value['class_id'] > 0}
        scope = build_gt_scope(source, manifest['semantic_classes'], manifest['instance_classes'],
            review_required_ids=tuple(config['review_required_gt_ids'].get(scene, [])),
            official_category_by_id=official_categories)
        scope['official_annotation_source'] = str(annotation)
        scope['official_annotation_sha256'] = sha256_file(annotation)
        for row in scope['objects']:
            obj = objects.get(row['raw_gt_id'])
            row['official_annotation'] = ({'raw_gt_id': int(obj['id']), 'class_id': int(obj['class_id']),
                'class_name': obj['class_name']} if obj else None)
            if row['object_status'] == 'UNKNOWN':
                row['reason'] = 'OFFICIAL_CLASS_UNDEFINED' if obj and obj['class_id'] <= 0 else 'UNRESOLVED_OFFICIAL_CATEGORY'
        directory = args.output_root / 'gt_scope_candidates' / scene
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'gt_scope.json', scope)
        counts = Counter(x['object_status'] for x in scope['objects'])
        candidate = [x for x in scope['objects'] if x['object_status'] == 'TARGET']
        row = {'scene_id': scene, 'status_counts': dict(counts), 'target_candidates': len(candidate),
            'quality_verified_candidates': sum(x['quality_verified'] for x in candidate),
            'pending_review_ids': [x['raw_gt_id'] for x in scope['objects'] if x['review_required']],
            'unknown_ids': [x['raw_gt_id'] for x in scope['objects'] if x['object_status'] == 'UNKNOWN'],
            'target_lt_100_vertices': [x['raw_gt_id'] for x in candidate if x['reference_vertex_count'] < 100],
            'scope_candidate_sha256': scope_sha256(scope)}
        rows.append(row)
        emit('qualified_candidates', **{k: v for k, v in row.items() if k not in ('unknown_ids', 'target_lt_100_vertices')})
    summary = {'status': STATUS, 'per_scene': rows,
        'target_candidates': sum(x['target_candidates'] for x in rows),
        'quality_verified_candidates': sum(x['quality_verified_candidates'] for x in rows),
        'pending_review_count': sum(len(x['pending_review_ids']) for x in rows),
        'unknown_count': sum(len(x['unknown_ids']) for x in rows),
        'qualification_uses_predictions': False}
    write_json(args.output_root / 'candidate_scope_summary.json', summary)
    return summary


def observation_frames(args, scene, source, config, progress_path):
    input_path = args.input_config_root / (scene + '.json')
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    if inputs['scene'] != scene or inputs['camera']['trajectory_matrix'] != 'camera_to_world_4x4_row_major':
        raise EvaluationError('Input scene/camera convention changed')
    selection = inputs['frame_selection']
    frame_ids = list(range(selection['start'], selection['stop_exclusive'], selection['stride']))
    if len(frame_ids) != 400:
        raise EvaluationError('Development experiment requires the same fixed 400 input frames')
    camera = inputs['camera']
    k = np.array([[camera['fx'], 0., camera['cx']], [0., camera['fy'], camera['cy']], [0., 0., 1.]])
    scene_root = Path(inputs['source']['scene_root'])
    alias = Path('/data/chenkejun/CVPR/datasets/Replica') / scene
    if alias.exists() and alias.resolve() == scene_root.resolve():
        scene_root = alias
    trajectory_path = scene_root / inputs['source']['trajectory']
    poses = np.loadtxt(trajectory_path).reshape(-1, 4, 4)
    if frame_ids[-1] >= len(poses):
        raise EvaluationError('Input trajectory is incomplete')
    common = {'input_config_sha256': sha256_file(input_path), 'trajectory_sha256': sha256_file(trajectory_path)}
    renderer, cache_hashes = None, None
    if args.visibility == 'mesh':
        from plyfile import PlyData
        mesh = PlyData.read(source.metadata['source_mesh'])
        vertices = mesh['vertex'].data
        xyz = np.column_stack([vertices[x] for x in ('x', 'y', 'z')])
        if sha256_array(xyz) != sha256_array(source.xyz_ref):
            raise EvaluationError('GT mesh vertex order/geometry differs from the reference')
        faces = np.asarray(list(mesh['face'].data['vertex_indices']), dtype=np.int64)
        renderer = GTMeshVisibility(xyz, faces, source.raw_instance_id)
        common.update(renderer.source_hashes)
        common['source_mesh_file_sha256'] = source.metadata['source_mesh_sha256']
        emit('mesh_loaded', scene_id=scene, vertices=len(xyz), triangles=len(faces),
             mixed_label_triangles=renderer.mixed_label_face_count)
    else:
        provenance_path = args.gt_cache_root / 'provenance.json'
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        cache_hashes = {int(x['frame_id']): x['sha256'] for x in provenance['frames'] if x['scene'] == scene}
        if set(cache_hashes) != set(frame_ids):
            raise EvaluationError('GT cache frame list differs from the fixed input sequence')
        common['gt_cache_provenance_sha256'] = sha256_file(provenance_path)
    started = time.monotonic()
    for ordinal, frame_id in enumerate(frame_ids, 1):
        depth_path = scene_root / inputs['source']['depth_pattern'].format(frame=frame_id)
        rgb_path = scene_root / inputs['source']['rgb_pattern'].format(frame=frame_id)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / float(camera['depth_png_units_per_meter'])
        if depth.shape != (camera['height'], camera['width']):
            raise EvaluationError('Depth shape differs from the frozen input camera')
        hashes = {**common, 'depth_file_sha256': sha256_file(depth_path), 'rgb_file_sha256': sha256_file(rgb_path)}
        if renderer is not None:
            ids, rendered_depth = renderer.render_reference_pixels(source.xyz_ref, k, poses[frame_id], depth.shape)
        else:
            cache_path = args.gt_cache_root / scene / f'frame{frame_id:06d}.png'
            digest = sha256_file(cache_path)
            if digest != cache_hashes[frame_id]:
                raise EvaluationError('GT cache checksum differs from its provenance')
            ids = np.asarray(Image.open(cache_path), dtype=np.int64)
            rendered_depth = None
            hashes['gt_id_cache_sha256'] = digest
        yield ObservationFrame(frame_id, depth, k, poses[frame_id], ids, rendered_depth, hashes)
        if ordinal % 20 == 0 or ordinal == len(frame_ids):
            progress = {'stage': 'observing_gt', 'scene_id': scene, 'completed_frames': ordinal,
                'total_frames': len(frame_ids), 'elapsed_seconds': round(time.monotonic() - started, 1)}
            write_json(progress_path, progress)
            emit('observing_gt', **{key: value for key, value in progress.items() if key != 'stage'})


def build_observed_gt(args, scene, config, directory):
    source, _ = load_scope_source(args.reference_root, scene)
    scope = json.loads((args.output_root / 'gt_scope_candidates' / scene / 'gt_scope.json').read_text(encoding='utf-8'))
    counts, metadata = accumulate_observed_support(source.xyz_ref, source.raw_instance_id,
        observation_frames(args, scene, source, config, args.output_root / 'progress.json'),
        config['observation']['depth_tolerance_m'])
    metadata['observer_code_sha256'] = {name: sha256_file(Path(__file__).parents[1] / 'unified_eval' / name)
        for name in ('observable_surface.py', 'mesh_visibility.py', 'gt_scope.py', 'replica.py')}
    ambiguous = coincident_label_ambiguity(source.xyz_ref, source.raw_instance_id)
    support_path = directory / 'gt_observed_support.npz'
    save_observed_support(support_path, source.xyz_ref, source.raw_instance_id, counts, metadata, ambiguous)
    metadata['support_file'] = str(support_path)
    metadata['support_file_sha256'] = sha256_file(support_path)
    gt, scope = apply_observed_scope(source, scope, counts, observation_metadata=metadata, ambiguity_mask=ambiguous)
    save_gt(directory / 'gt.npz', gt)
    write_json(directory / 'gt_scope.json', scope)
    report = {'scene_id': scene, 'status': STATUS, 'visibility_status': metadata['status'],
        'gt_scope_sha256': gt.metadata['gt_scope_sha256'], 'observed_support_sha256': gt.metadata['gt_observed_support_sha256'],
        'evaluable_gt_count': sum(x['evaluable'] for x in scope['objects']),
        'target_vertices': int(np.count_nonzero(gt.evaluation_region == 1)),
        'known_non_target_vertices': int(np.count_nonzero(gt.evaluation_region == 2)),
        'ignore_vertices': int(np.count_nonzero(gt.evaluation_region == 0)),
        'coincident_different_label_vertices': int(ambiguous.sum()),
        'unobserved_verified_target_ids': [x['raw_gt_id'] for x in scope['objects'] if x['object_status'] == 'TARGET'
            and x['quality_verified'] and not x['observable']],
        'small_target_audit': [x for x in scope['objects'] if x['object_status'] == 'TARGET' and x['reference_vertex_count'] < 100]}
    write_json(directory / 'observation_audit.json', report)
    emit('observed_gt_complete', **{key: value for key, value in report.items() if key not in ('small_target_audit',)})
    return gt, report


def reuse_checked_observed_gt(args, scene, config, directory):
    """Reuse evidence only after source, policy, code and every RGB/depth checksum pass."""
    gt = load_gt(directory / 'gt.npz')
    source, _ = load_scope_source(args.reference_root, scene)
    scope = json.loads((directory / 'gt_scope.json').read_text(encoding='utf-8'))
    if scope_sha256(scope) != gt.metadata['gt_scope_sha256'] or sha256_array(gt.xyz_ref) != sha256_array(source.xyz_ref):
        raise EvaluationError('Stored observed GT source/scope checksum changed')
    meta = gt.metadata['observation_metadata']
    for name, digest in meta['observer_code_sha256'].items():
        if sha256_file(Path(__file__).parents[1] / 'unified_eval' / name) != digest:
            raise EvaluationError('Observation code changed; regenerate support before reusing it')
    support_path = Path(meta['support_file'])
    if sha256_file(support_path) != meta['support_file_sha256']:
        raise EvaluationError('Observation support file changed')
    with np.load(support_path, allow_pickle=False) as stored:
        counts, ambiguous = stored['observation_count'].copy(), stored['ambiguity_mask'].copy()
        if not np.array_equal(counts, gt.observation_count):
            raise EvaluationError('Observation support count and GT count disagree')
    candidate = json.loads((args.output_root / 'gt_scope_candidates' / scene / 'gt_scope.json').read_text(encoding='utf-8'))
    regenerated, _ = apply_observed_scope(source, candidate, counts, observation_metadata=meta, ambiguity_mask=ambiguous)
    if regenerated.metadata['gt_scope_sha256'] != gt.metadata['gt_scope_sha256']:
        raise EvaluationError('Qualification policy changed; create a new named scope')
    input_path = args.input_config_root / (scene + '.json')
    inputs = json.loads(input_path.read_text(encoding='utf-8'))
    root = Path(inputs['source']['scene_root'])
    for evidence in meta['input_evidence']:
        frame = evidence['frame_id']
        hashes = evidence['source_hashes']
        for path, key in ((input_path, 'input_config_sha256'),
            (root / inputs['source']['trajectory'], 'trajectory_sha256'),
            (root / inputs['source']['depth_pattern'].format(frame=frame), 'depth_file_sha256'),
            (root / inputs['source']['rgb_pattern'].format(frame=frame), 'rgb_file_sha256')):
            if sha256_file(path) != hashes[key]:
                raise EvaluationError(f'Observation source file changed: {path}')
    report = json.loads((directory / 'observation_audit.json').read_text(encoding='utf-8'))
    emit('observed_gt_reused_after_checksum_verification', scene_id=scene)
    return gt, report


def p1_input_provenance(args, scene, gt):
    directory = args.baseline_root / 'native' / scene / 'P1-A1'
    association_path = directory / 'association/association_report.json'
    surface_path = directory / 'surface_p0/instance_surface.npz'
    report_path = directory / 'surface_p0/materialization_report.json'
    association = json.loads(association_path.read_text(encoding='utf-8'))
    material = json.loads(report_path.read_text(encoding='utf-8'))
    frames = [x['frame_id'] for x in association['frame_summaries']]
    config_path = args.input_config_root / (scene + '.json')
    if frames != gt.metadata['observation_metadata']['frame_ids'] or association['frame_count'] != len(frames) or material['frame_count'] != len(frames):
        raise EvaluationError('P1 map input frame sequence differs from the observed GT scope')
    if association['config_sha256'] != sha256_file(config_path) or material['config_sha256'] != sha256_file(config_path):
        raise EvaluationError('P1 map input configuration changed')
    if material['instance_surface_sha256'] != sha256_file(surface_path) or material['ground_truth_used']:
        raise EvaluationError('P1 source surface hash/GT isolation check failed')
    with np.load(surface_path, allow_pickle=False) as data:
        if str(data['association_decisions_sha256'].item()) != material['association_decisions_sha256']:
            raise EvaluationError('P1 surface association provenance disagrees')
    return {'input_scope_verified': True, 'frame_list_sha256': gt.metadata['observation_metadata']['frame_list_sha256'],
        'frame_count': len(frames), 'association_report_sha256': sha256_file(association_path),
        'materialization_report_sha256': sha256_file(report_path), 'input_config_sha256': sha256_file(config_path),
        'baseline_surface_sha256': sha256_file(surface_path),
        'revision_input_basis': 'same hash-verified full TSDF geometry and baseline 400-frame input contract'}


def ovi_input_provenance(args, scene, gt, surface_path):
    audit_path = args.ovi_input_audit_root / scene / 'input_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    evidence = audit['aligned_frames_audit']
    count = gt.metadata['observation_metadata']['frame_count']
    if audit['frame_count'] != count or audit['frame_list_sha256'] != gt.metadata['observation_metadata']['frame_list_sha256']:
        raise EvaluationError('OVI-MAP source input frames differ from observed GT scope')
    if not evidence['all_verified'] or any(evidence[key] != count for key in
        ('aligned_frames_record_count', 'frames_match', 'poses_match_atol_1e_6', 'intrinsics_match', 'decoded_rgb_hashes_match')):
        raise EvaluationError('OVI-MAP per-frame alignment audit is incomplete')
    manifest_path = surface_path.with_suffix('.json')
    if sha256_file(surface_path) != audit['ovi_export_sha256'] or sha256_file(manifest_path) != audit['ovi_export_manifest_sha256']:
        raise EvaluationError('OVI-MAP source export changed since its alignment audit')
    return {'input_scope_verified': True, 'frame_count': count, 'frame_list_sha256': audit['frame_list_sha256'],
        'alignment_audit_file': str(audit_path), 'alignment_audit_sha256': sha256_file(audit_path),
        'alignment_basis': 'existing per-frame pose/intrinsic/decoded RGB audit tied to immutable export hashes',
        'export_manifest_sha256': sha256_file(manifest_path)}


def direct_gt_oracle(gt, protocol):
    ids = np.unique(gt.instance_id[gt.evaluation_region == 1])
    pred = CanonicalPrediction(gt.scene_id, gt.vertex_count,
        [CanonicalInstance(str(int(raw)), np.flatnonzero(gt.instance_id == raw).astype(np.int32)) for raw in ids],
        'discrete_GT_oracle_evaluator_test_only', 'not_a_research_method', 'direct_reference_oracle', protocol.name, True,
        {**gt.metadata, 'no_geometry_target_vertex_count': 0, 'evaluation_only_discrete_gt_oracle': True})
    summary = evaluate_scenes([(gt, pred)], protocol)[0]
    for value in (summary['CA_PQ']['PQ'], summary['CA_PRF1_0_5']['F1'], summary['CA_AP50_uniform'],
                  summary['CA_mCov'], summary['owner_surface']['Correct_owner_Coverage']):
        if value != 1:
            raise EvaluationError('Discrete GT oracle did not achieve its ideal score')
    return summary


def geometry_oracle(gt, xyz, correspondence, protocol, support):
    """GT-derived labels are ONLY an oracle test; never overwrite method labels."""
    distance, nearest = cKDTree(gt.xyz_ref).query(xyz, k=1, workers=-1)
    labels = gt.instance_id[nearest].copy()
    labels[distance >= protocol.geometry_mapping_max_distance_m] = -1
    result = map_fixed_surface_labels(xyz, labels, gt.xyz_ref, correspondence,
        scene_id=gt.scene_id, method_name='GT_to_TSDF_geometry_oracle_test_only',
        method_commit='not_a_research_method',
        metadata={'oracle_uses_gt_to_construct_test_labels': True})
    attach_reference_support(result, labels, support, gt, correspondence)
    return evaluate_scenes([(gt, result.prediction)], protocol)[0]


def audit_scene(args, scene, config):
    directory = args.output_root / args.visibility / scene
    directory.mkdir(parents=True, exist_ok=True)
    gt, observation = (reuse_checked_observed_gt(args, scene, config, directory)
                       if args.reuse_observed and (directory / 'gt.npz').exists()
                       else build_observed_gt(args, scene, config, directory))
    protocol = Protocol.from_dict(config)
    discrete = direct_gt_oracle(gt, protocol)
    write_json(directory / 'discrete_gt_oracle.json', discrete)
    emit('discrete_gt_oracle_passed', scene_id=scene)
    surfaces = {
        'P1-A1_native': args.baseline_root / 'native' / scene / 'P1-A1/surface_p0/instance_surface.npz',
        'P1-A1_holes_geodesic': args.baseline_root / 'final' / scene / 'P1-A1/diffusion/holes_geodesic/instance_surface.npz',
        'P1-A1_auto_v2': args.auto_root / scene / 'final/diffusion/holes_geodesic/instance_surface.npz'}
    cache = directory / 'fixed_gt_to_tsdf.npz'
    p1_provenance = p1_input_provenance(args, scene, gt)
    results = {}
    for name, path in surfaces.items():
        emit('adapting_surface', scene_id=scene, method_name=name)
        results[name] = adapt_surface(path, directory / 'gt.npz', args.config, cache, directory / name,
            method_name=name, method_commit='existing_immutable_export_sha256_in_manifest', source_provenance=p1_provenance)
        summary = results[name]['summary']
        emit('surface_evaluated', scene_id=scene, method_name=name, gt_count=summary['CA_PRF1_0_5']['TP'] + summary['CA_PRF1_0_5']['FN'],
            PQ=summary['CA_PQ']['PQ'], F1=summary['CA_PRF1_0_5']['F1'], mCov=summary['CA_mCov'])
    names = list(surfaces)
    pairs = {}
    for before, after in zip(names, names[1:]):
        pair = paired_revision_metrics(gt, load_prediction(directory / before / 'canonical_prediction.npz'),
            load_prediction(directory / after / 'canonical_prediction.npz'), protocol)
        pairs[before + '_to_' + after] = pair
        write_json(directory / (before + '_to_' + after + '.paired.json'), pair)
    ovi_name = 'OVI-MAP_full400'
    ovi_path = args.ovimap_root / ('ovi_' + scene + '_full400.npz')
    emit('adapting_surface', scene_id=scene, method_name=ovi_name)
    results[ovi_name] = adapt_surface(ovi_path, directory / 'gt.npz', args.config,
        directory / 'fixed_gt_to_ovimap.npz', directory / ovi_name, method_name=ovi_name,
        method_commit='existing_immutable_export_sha256_in_manifest',
        source_provenance=ovi_input_provenance(args, scene, gt, ovi_path))
    emit('surface_evaluated', scene_id=scene, method_name=ovi_name,
        PQ=results[ovi_name]['summary']['CA_PQ']['PQ'], F1=results[ovi_name]['summary']['CA_PRF1_0_5']['F1'])
    with np.load(surfaces[names[0]], allow_pickle=False) as data:
        xyz = data['xyz_m'].copy()
    correspondence = load_fixed_surface_correspondence(cache, xyz, gt.xyz_ref)
    support = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, protocol.geometry_mapping_max_distance_m)
    oracle = geometry_oracle(gt, xyz, correspondence, protocol, support)
    write_json(directory / 'GT_to_TSDF_geometry_oracle.json', oracle)
    # Predeclared offsets audit gate sensitivity; this does not select a threshold.
    perturbations = []
    for millimeters in (() if args.reuse_observed and (directory / 'geometry_perturbation_audit.json').exists() else (1, 3, 5, 10, 20)):
        changed_xyz = xyz + np.array([millimeters / 1000, 0, 0], dtype=xyz.dtype)
        changed_cache = build_fixed_surface_correspondence(changed_xyz, gt.xyz_ref, protocol.geometry_mapping_max_distance_m)
        target = gt.evaluation_region == 1
        perturbations.append({'translation_x_mm': millimeters,
            'target_geometry_coverage': float(np.mean(changed_cache.nearest_native_index[target] >= 0)),
            'native_xyz_sha256': changed_cache.native_xyz_sha256, 'correspondence_sha256': changed_cache.sha256})
    if perturbations:
        write_json(directory / 'geometry_perturbation_audit.json', {'fixed_gate_m': protocol.geometry_mapping_max_distance_m,
            'geometry_xyz_sha256': correspondence.native_xyz_sha256,
            'purpose': 'predeclared geometry sensitivity; thresholds remain development values', 'offsets': perturbations})
    small = {x['raw_gt_id']: x for x in observation['small_target_audit'] if x['evaluable']}
    small_results = []
    for name in results:
        prediction = load_prediction(directory / name / 'canonical_prediction.npz')
        diagnostic = load_prediction(directory / name / 'diagnostic_support_prediction.npz')
        overlap = build_overlap(gt, prediction, protocol)
        diagnostic_overlap = build_overlap(gt, diagnostic, protocol)
        for raw_id, row in small.items():
            col = list(overlap.gt_ids).index(raw_id)
            small_results.append({'method_name': name, 'raw_gt_id': raw_id, 'semantic_class': row['semantic_class'],
                'reference_vertex_count': row['reference_vertex_count'], 'observed_vertex_count': row['observed_vertex_count'],
                'best_iou': float(overlap.iou[:, col].max(initial=0)),
                'significant_predictions_main_10_vertices_5_percent': int(np.count_nonzero(
                    (diagnostic_overlap.intersection[:, col] >= 10) & (diagnostic_overlap.recall[:, col] >= .05))),
                'significant_predictions_secondary_1_vertex_5_percent': int(np.count_nonzero(
                    (diagnostic_overlap.intersection[:, col] >= 1) & (diagnostic_overlap.recall[:, col] >= .05)))})
        secondary = significant_structure_diagnostics(diagnostic_overlap,
            replace(protocol, significant_min_intersection_vertices=1))
        write_json(directory / name / 'structure_relative_only_sensitivity.json',
            {'status': 'SECONDARY_DIAGNOSTIC_NOT_THE_MAIN_PROTOCOL', 'scope': 'all GT for threshold sensitivity', **secondary})
    write_json(directory / 'small_target_method_audit.json', {'per_gt': small_results,
        'main_structure_threshold': {'minimum_vertices': 10, 'minimum_gt_fraction': .05},
        'threshold_is_frozen': False})
    report = {'scene_id': scene, 'status': STATUS, 'observation': observation,
        'methods': {name: {key: value['summary'][key] for key in
            ('CA_PQ', 'CA_PRF1_0_5', 'CA_mCov', 'owner_surface', 'structure', 'background_only_prediction_count',
             'unknown_or_unobserved_prediction_count', 'unverifiable_prediction_count')} for name, value in results.items()},
        'pair_changes': {name: {key: value for key, value in pair.items() if key != 'per_gt'} for name, pair in pairs.items()},
        'shared_native_geometry_sha256': correspondence.native_xyz_sha256,
        'shared_correspondence_sha256': correspondence.sha256,
        'discrete_oracle_passed': True, 'geometry_oracle': {key: oracle[key] for key in ('CA_PQ', 'CA_PRF1_0_5', 'CA_mCov', 'owner_surface')}}
    write_json(directory / 'scene_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path('/data/chenkejun/CVPR')
    parser.add_argument('--reference-root', type=Path, default=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908/reference'))
    parser.add_argument('--annotation-root', type=Path, default=Path('/data/chenkejun/ReplicaSSG/Replica/data'))
    parser.add_argument('--input-config-root', type=Path, default=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs'))
    parser.add_argument('--gt-cache-root', type=Path, default=data / 'datasets/gt_visibility_cache_20261009')
    parser.add_argument('--baseline-root', type=Path, default=data / 'revisable_instance_map/p1a1_raw_replica8_20261002')
    parser.add_argument('--auto-root', type=Path, default=data / 'revisable_instance_map/p1a1_auto_v2_apply_20261004')
    parser.add_argument('--ovimap-root', type=Path, default=Path('/data/chenkejun/beauty/ovimap_aligned_eval_20260908/exports'))
    parser.add_argument('--ovi-input-audit-root', type=Path, default=data / 'evaluation_results/p0_vs_ovimap_v3_20260929')
    parser.add_argument('--output-root', type=Path, default=data / 'results/v3_object_observed_repair_20261009')
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'unified_eval/configs/replica_ca_v3.object_observed_repair.development.json')
    parser.add_argument('--visibility', choices=('mesh', 'cache'), default='mesh')
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=['room0', 'room1'])
    parser.add_argument('--scope-only', action='store_true')
    parser.add_argument('--reuse-observed', action='store_true')
    args = parser.parse_args()
    if not args.output_root.resolve().is_relative_to(Path('/data/chenkejun/CVPR')):
        raise EvaluationError('Experiment outputs must stay under /data/chenkejun/CVPR')
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    Protocol.from_dict(config)
    qualify(args, config)
    if not args.scope_only:
        reports = [audit_scene(args, scene, config) for scene in args.scenes]
        protocol = Protocol.from_dict(config)
        pooled = {}
        for method in reports[0]['methods']:
            directories = [args.output_root / args.visibility / scene for scene in args.scenes]
            summary = evaluate_scenes([(load_gt(directory / 'gt.npz'),
                load_prediction(directory / method / 'canonical_prediction.npz')) for directory in directories], protocol,
                diagnostic_predictions=[load_prediction(directory / method / 'diagnostic_support_prediction.npz') for directory in directories])[0]
            pooled[method] = summary
        write_json(args.output_root / (args.visibility + '_method_comparison.development.json'),
            {'status': STATUS, 'scene_ids': args.scenes, 'methods': pooled})
        with (args.output_root / (args.visibility + '_comparison.development.csv')).open('w', newline='', encoding='utf-8') as handle:
            fields = ['method', 'scene_count', 'gt_count', 'PQ', 'F1@50', 'mCov', 'Correct-owner Coverage',
                'Wrong-owner Coverage', 'Unassigned Coverage', 'No-geometry Coverage', 'Merge Count', 'Split Count',
                'Background-only Predictions', 'Unknown-only Predictions', 'Unverifiable Predictions']
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for method, summary in pooled.items():
                writer.writerow(dict(zip(fields, [method, len(args.scenes), summary['CA_PRF1_0_5']['TP'] + summary['CA_PRF1_0_5']['FN'],
                    summary['CA_PQ']['PQ'], summary['CA_PRF1_0_5']['F1'], summary['CA_mCov'],
                    *[summary['owner_surface'][key] for key in ('Correct_owner_Coverage', 'Wrong_owner_Coverage', 'Unassigned_Coverage', 'No_geometry_Coverage')],
                    summary['structure']['merge_prediction_count'], summary['structure']['split_gt_count'],
                    summary['background_only_prediction_count'], summary['unknown_or_unobserved_prediction_count'], summary['unverifiable_prediction_count']])))
        write_json(args.output_root / (args.visibility + '_acceptance_report.json'), {'status': STATUS, 'scenes': reports,
            'profile_config_sha256': sha256_file(args.config), 'historical_results_modified': False})
        write_json(args.output_root / 'progress.json', {'stage': 'complete', 'scene_ids': args.scenes, 'status': STATUS})
        emit('complete', scene_ids=args.scenes, status=STATUS)


if __name__ == '__main__':
    main()

"""Evaluate fixed OVO exports using the already verified, identical V3 evaluator."""
from pathlib import Path
import os
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

CODE = Path(os.environ.get('CVPR_EVALUATOR_ROOT', '/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot'))
DATA = Path(os.environ.get('CVPR_DATA_ROOT', '/data/chenkejun/CVPR'))
SOURCE = Path(os.environ.get('CVPR_OVO_SOURCE_ROOT', '/data/chenkejun/beauty/ovo_paper_original_20260915'))
OUT = Path(os.environ.get('CVPR_OVO_AUDIT_OUT', str(DATA / 'results/ovo_evaluation_audit_20261007')))
BASE_AUDIT = Path(os.environ.get('CVPR_EVAL_AUDIT_OUT', str(DATA / 'results/evaluation_audit_20261007'))) / 'verification.json'
SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
COMMIT = 'd01d821bec7c25c437803b8ab9c580e73f92e119'
METHOD = 'OVO_first_public_release_reference'
FLAGS = ['--debug-max-distance-m', '0.01', '--debug-diagnostic-max-distance-m', '0.02',
         '--debug-min-valid-instance-vertices', '100', '--debug-significant-min-vertices', '10',
         '--debug-significant-min-gt-fraction', '0.05']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = read(BASE_AUDIT)
    assert base['status'] == 'PASS' and base['total_scene_evaluations'] == 40
    for name, expected in base['source_evaluator_files_sha256'].items():
        assert digest(CODE / name) == expected, f'frozen evaluator changed: {name}'
    sys.path.insert(0, str(CODE))
    from unified_eval.geometry import map_instances_to_reference_v3
    from unified_eval.io import load_gt, load_prediction, save_prediction
    from unified_eval.schema import Protocol
    from unified_eval.cli import effective_protocol_sha256
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_existing_evaluations_20261007 import aggregate, compare

    expected_protocol = base['effective_protocol_config']
    config = CODE / 'unified_eval/configs/replica_ca_v3.pending.json'
    protocol = Protocol.from_dict(expected_protocol)
    assert effective_protocol_sha256(expected_protocol) == base['effective_protocol_sha256']
    source_config = read(config)
    source_config['geometry_mapping']['max_distance_m'] = 0.01
    source_config['diagnostic_mapping']['max_distance_m'] = 0.02
    source_config['instance_filter']['min_valid_instance_vertices'] = 100
    source_config['significant_overlap'].update(min_intersection_vertices=10, min_gt_fraction=0.05)
    assert source_config == expected_protocol
    reference_evidence = {e['scene']: e for e in base['methods'][0]['evidence']}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
              'protocol': 'Replica-CA-v3', 'evaluation_status': 'DEBUG_ONLY / NON_OFFICIAL',
              'effective_protocol_config': expected_protocol,
              'effective_protocol_sha256': base['effective_protocol_sha256'],
              'gt_sha256_by_scene': base['gt_sha256_by_scene'],
              'source_evaluator_snapshot': str(CODE),
              'source_evaluator_files_sha256': base['source_evaluator_files_sha256'],
              'scope': 'Re-evaluate unchanged, previously exported native OVO geometry; no mapping rerun, geometry filling, filtering by predicted semantic class, or parameter search.',
              'comparison_limit': 'OVO uses its native 2000-input-frame sequence, 400 geometry updates and 200 segmentation updates; P1-A1/OVI-MAP each used 400 input/instance frames. Same GT and metric protocol, different input and frontend budgets; OVO is a reference comparison.',
              'method': METHOD, 'source_commit': COMMIT, 'methods': [],
              'historical_v2_summary_sha256': digest(SOURCE / 'evaluation/final_comparison.json')}
    dump(OUT / 'verification.json', report)
    rows, evidence, batch_entries = [], [], []
    for scene in SCENES:
        src = SOURCE / 'results' / scene
        status = read(src / 'run_status.json')
        export = read(src / 'export.json')
        assert status['status'] == 'complete' and status['scene'] == scene
        assert status['source_commit'] == COMMIT and status['empty_map_start'] is True
        assert status['input_frames'] == 2000 and status['geometric_updates_expected'] == 400
        assert status['segmentation_updates_expected'] == status['segmentation_records'] == 200
        records = [json.loads(line) for line in (src / 'progress.jsonl').read_text().splitlines() if line.strip()]
        assert [x['source_frame'] for x in records] == list(range(0, 2000, 10))
        prediction_file = src / 'prediction.npz'
        native_map = src / 'ovo_map.ckpt'
        source_sha = digest(prediction_file)
        assert source_sha == export['export_sha256']
        assert digest(native_map) == export['map_sha256']
        current_config = (src / 'config.yaml').read_text()
        assert '  map_every: 5' in current_config and '  segment_every: 10' in current_config
        assert '  slam_module: vanilla' in current_config and '    use_half: true' in current_config

        with np.load(prediction_file, allow_pickle=False) as data:
            xyz, labels, semantic = data['xyz'].copy(), data['instance'].copy(), data['classes'].copy()
        count = export['objects']
        assert len(xyz) == len(labels) == export['points'] and xyz.shape[1:] == (3,)
        assert np.isfinite(xyz).all() and labels.dtype.kind in 'iu'
        assert np.all((labels >= -1) & (labels < count))
        assert len(export['source_object_ids']) == len(semantic) == count
        assert int((labels < 0).sum()) == export['unknown_geometry_preserved']
        # Export labels index source_object_ids; retain every native object, including empty ones.
        order = np.argsort(labels, kind='stable')
        sorted_labels, sorted_xyz = labels[order], xyz[order]
        clouds = [sorted_xyz[np.searchsorted(sorted_labels, index, side='left'):
                             np.searchsorted(sorted_labels, index, side='right')]
                  for index in range(count)]

        old_manifest = read(Path(reference_evidence[scene]['evaluation_dir']) / 'manifest.json')
        gt_file = Path(old_manifest['gt_file'])
        assert digest(gt_file) == base['gt_sha256_by_scene'][scene]
        gt = load_gt(gt_file)
        adapter = OUT / scene / 'adapter'
        evaluation = OUT / scene / 'evaluation'
        adapter.mkdir(parents=True, exist_ok=True)
        metadata = {'source_export': str(prediction_file), 'source_export_sha256': source_sha,
                    'source_native_map': str(native_map), 'source_native_map_sha256': export['map_sha256'],
                    'source_export_manifest': str(src / 'export.json'),
                    'source_export_manifest_sha256': digest(src / 'export.json'),
                    'source_run_status': str(src / 'run_status.json'),
                    'source_run_status_sha256': digest(src / 'run_status.json'),
                    'source_config_sha256': digest(src / 'config.yaml'),
                    'source_progress_sha256': digest(src / 'progress.jsonl'),
                    'native_object_count': count, 'input_frames': 2000,
                    'frame_count_from_source': 400, 'geometry_updates': 400, 'segmentation_updates': 200,
                    'source_frame_ids_geometry': list(range(0, 2000, 5)),
                    'source_frame_ids_segmentation': list(range(0, 2000, 10)),
                    'empty_map_start': True, 'debug_only': True,
                    'source_label_encoding': 'compact_index_into_source_object_ids',
                    'unassigned_source_points': int((labels < 0).sum()),
                    'comparison_scope': 'reference only; different native input and frontend budget'}
        print(json.dumps({'scene': scene, 'stage': 'projecting_v3', 'objects': count,
                          'native_assigned_points': sum(len(x) for x in clouds)}), flush=True)
        mapped = map_instances_to_reference_v3(clouds, gt.xyz_ref, 0.01, scene_id=scene,
            method_name=METHOD, method_commit=COMMIT, adapter_version='ovo_native_points_shared_v3',
            protocol_version='Replica-CA-v3', diagnostic_distance_m=0.02, metadata=metadata)
        for main_inst, diag_inst, native_id in zip(mapped.prediction.instances,
                                                 mapped.diagnostic_prediction.instances,
                                                 export['source_object_ids']):
            main_inst.instance_uid = diag_inst.instance_uid = f'ovo-id:{native_id}'
        mapped.diagnostic_prediction.metadata['source_export_sha256'] = source_sha
        mapped.prediction.validate()
        mapped.diagnostic_prediction.validate()
        assert mapped.prediction.is_partition and len(mapped.prediction.instances) == count
        pred_file = adapter / 'canonical_prediction.npz'
        diag_file = adapter / 'diagnostic_support_prediction.npz'
        save_prediction(pred_file, mapped.prediction)
        save_prediction(diag_file, mapped.diagnostic_prediction)
        dump(adapter / 'adapter_stats.json', mapped.statistics)
        # Serialization must preserve the exact masks and every native object.
        for original, file in [(mapped.prediction, pred_file), (mapped.diagnostic_prediction, diag_file)]:
            restored = load_prediction(file)
            assert len(restored.instances) == count
            assert all(a.instance_uid == b.instance_uid and np.array_equal(a.vertex_indices, b.vertex_indices)
                       for a, b in zip(original.instances, restored.instances))
        cmd = [sys.executable, '-m', 'unified_eval.cli', 'eval-scene', '--config', str(config), *FLAGS,
               '--gt', str(gt_file), '--pred', str(pred_file), '--diagnostic-pred', str(diag_file),
               '--out', str(evaluation)]
        subprocess.run(cmd, cwd=CODE, check=True)
        manifest = read(evaluation / 'manifest.json')
        assert manifest['effective_protocol_sha256'] == base['effective_protocol_sha256']
        assert manifest['protocol_config'] == expected_protocol
        assert manifest['gt_sha256'] == base['gt_sha256_by_scene'][scene]
        row = read(evaluation / 'metrics.json')
        rows.append(row)
        item = {'scene': scene, 'evaluation_dir': str(evaluation), 'manifest_sha256': digest(evaluation / 'manifest.json'),
                'files_sha256': {name: digest(file) for name, file in [('gt_file', gt_file), ('prediction_file', pred_file), ('diagnostic_prediction_file', diag_file)]},
                'prediction_source': metadata, 'native_export_matches_recorded_sha256': True,
                'source_checkpoint_matches_recorded_sha256': True, 'all_native_objects_retained': True,
                'same_protocol_and_GT_as_P1A1_OVI_MAP': True}
        evidence.append(item)
        batch_entries.append({'gt': str(gt_file), 'prediction': str(pred_file), 'diagnostic_prediction': str(diag_file)})
        report['methods'] = [{'method': METHOD, 'evidence': evidence}]
        dump(OUT / 'verification.json', report)
        print(json.dumps({'scene': scene, 'stage': 'PASS', 'AP50': row['CA_AP50_uniform'],
                          'PQ': row['CA_PQ']['PQ'], 'F1': row['CA_PRF1_0_5']['F1']}), flush=True)
        del xyz, labels, semantic, order, sorted_labels, sorted_xyz, clouds, gt, mapped, original, restored
        gc.collect()

    dump(OUT / 'batch_inputs.json', {'scenes': batch_entries})
    subprocess.run([sys.executable, '-m', 'unified_eval.cli', 'eval-batch', '--config', str(config), *FLAGS,
                    '--scenes', str(OUT / 'batch_inputs.json'), '--out', str(OUT / 'pooled')], cwd=CODE, check=True)
    summary_file = OUT / 'pooled/summary.json'
    summary = read(summary_file)
    assert summary['scene_count'] == 8 and summary['status'] == 'DEBUG_ONLY / NON_OFFICIAL'
    aggregate_check = aggregate(rows, protocol, summary['status'])
    compare(aggregate_check, summary, 'OVO pooled cross-check')
    assert digest(SOURCE / 'evaluation/final_comparison.json') == report['historical_v2_summary_sha256']
    for name, expected in base['source_evaluator_files_sha256'].items():
        assert digest(CODE / name) == expected, f'frozen evaluator changed during scoring: {name}'
    report['methods'] = [{'method': METHOD, 'scene_count': 8, 'evidence': evidence,
                          'recomputed_summary': str(summary_file), 'source_summary': str(summary_file),
                          'recomputed_summary_sha256': digest(summary_file), 'pooled_aggregation_independently_verified': True}]
    report.update(status='PASS', completed_utc=datetime.now(timezone.utc).isoformat(), total_scene_evaluations=8,
                  all_methods_share_identical_protocol_and_GT=True, native_map_rerun=False)
    dump(OUT / 'verification.json', report)
    print(json.dumps({'stage': 'ALL_PASS', 'AP50': summary['CA_AP50_uniform'],
                      'PQ': summary['CA_PQ']['PQ'], 'F1': summary['CA_PRF1_0_5']['F1']}), flush=True)


if __name__ == '__main__':
    main()

"""Separate post-hoc process: unchanged unified v3 and fixed-surface diagnostics."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'strict_local_repair_20261007'))
import evaluate_pilot_v3 as e
import fixed_surface_repair as f


def check_hashes(hashes):
    for path, expected in hashes.items():
        if f.file_sha256(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)


def main(config_path, evaluation_path):
    cfg = json.loads(Path(config_path).read_text())
    evaluation_path = Path(evaluation_path).resolve()
    ev = json.loads(evaluation_path.read_text())
    assert ev['posthoc_evaluation_only'] and ev['scene'] == cfg['scene']
    out = Path(cfg['output_dir'])
    complete = json.loads((out / 'complete.json').read_text())
    assert complete['status'] == 'PASS' and complete['predictions_frozen_before_any_GT_evaluation']
    assert complete['config_sha256'] == f.file_sha256(config_path)
    assert not complete['GT_used_for_repair']
    check_hashes(complete['repair']['output_sha256'])
    source_freeze = json.loads((out / 'source_freeze.json').read_text())
    check_hashes(source_freeze['protected_source_hashes'])
    scene = cfg['scene']
    old_freeze = json.loads((e.ROOT / 'v3/evaluation_freeze.json').read_text())
    assert old_freeze['flags'] == e.FLAGS and old_freeze['code_sha256'] == f.file_sha256(e.__file__)
    baseline_path = e.BASE / ('final/%s/P1-A1/diffusion/holes_geodesic/instance_surface.npz' % scene)
    candidates = {'baseline': baseline_path, **{name: Path(path) for name, path in ev.get('comparison_conditions', {}).items()},
                  'objectwise': out / 'final/instance_surface.npz'}
    freeze = {'status': 'PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION', 'GT_used_for_repair': False,
              'case_config_sha256': f.file_sha256(config_path), 'repair_complete_sha256': f.file_sha256(out / 'complete.json'),
              'evaluation_config_sha256': f.file_sha256(evaluation_path), 'evaluation_config': ev,
              'evaluator_sha256': f.file_sha256(e.__file__), 'wrapper_sha256': f.file_sha256(__file__),
              'protocol_sha256': f.file_sha256(e.PROTOCOL), 'flags': e.FLAGS,
              'effective_protocol': old_freeze['effective_protocol'],
              'predictions': {name: {'path': str(path), 'sha256': f.file_sha256(path)} for name, path in candidates.items()}}
    freeze_path = out / 'evaluation_freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == freeze
    else:
        f.atomic_json(freeze_path, freeze)
    if (out / 'evaluation_summary.json').exists():
        done = json.loads((out / 'evaluation_summary.json').read_text())
        assert done['prediction_freeze_sha256'] == f.file_sha256(freeze_path)
        assert done['mapping_cache_sha256'] == f.file_sha256(done['mapping_cache'])
        return done
    parser = e.ArgumentParser()
    e.add_protocol_args(parser)
    protocol, raw, debug = e.load_protocol(e.PROTOCOL, parser.parse_args(['--config', str(e.PROTOCOL), *e.FLAGS]))
    assert raw == old_freeze['effective_protocol']
    baseline_score = json.loads((e.ROOT / ('v3/baseline/%s/complete.json' % scene)).read_text())
    assert baseline_score['source_map_sha256'] == f.file_sha256(baseline_path)
    score = e.evaluate(scene, candidates['objectwise'], out / 'v3', protocol)
    for key in ('flags', 'protocol_sha256', 'evaluation_code_sha256', 'GT_sha256'):
        assert score[key] == baseline_score[key]
    # First opening of GT: all predictions and source hashes have already been frozen.
    gt = e.load_gt(e.INPUT / scene / 'ground_truth/gt.npz')
    baseline = f.load_arrays(baseline_path)
    mapping_path = Path(ev['fixed_mapping_cache']) if 'fixed_mapping_cache' in ev else out / 'fixed_surface_mapping.npz'
    cache_hash_before = f.file_sha256(mapping_path) if mapping_path.exists() else None
    mapping = f.fixed_mapping(mapping_path, baseline['xyz_m'], gt.xyz_ref, ev['diagnostic_max_distance_m'])
    if cache_hash_before:
        assert cache_hash_before == mapping['cache_sha256']
    identity = {int(pid): int(gid) for pid, gid in ev['posthoc_target_identity'].items()}
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    diagnostics = {}
    for name, path in candidates.items():
        surface = f.load_arrays(path)
        f.require_same_geometry(baseline, surface)
        diagnostics[name] = f.target_diagnostics(mapping, surface['instance_id'], gt.instance_id, valid, identity)
        diagnostics[name]['baseline_to_version_transitions'] = f.target_transitions(mapping, baseline['instance_id'],
            surface['instance_id'], gt.instance_id, valid, identity)
    before = e.per_gt(e.load_overlap(e.ROOT / ('v3/baseline/%s/overlap.npz' % scene)))
    after = e.per_gt(e.load_overlap(out / 'v3/overlap.npz'))
    target = set(identity.values())
    harm = {'previously_correct_unselected_lost': [g for g in before if g not in target and before[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']],
            'unselected_IoU_degraded_over_0_01': [g for g in before if g not in target and after[g]['best_IoU'] < before[g]['best_IoU'] - .01]}
    for row in freeze['predictions'].values():
        assert row['sha256'] == f.file_sha256(row['path'])
    check_hashes(complete['repair']['output_sha256'])
    check_hashes(source_freeze['protected_source_hashes'])
    summary = {'status': 'PASS', 'scene': scene, 'case_uid': cfg['case_uid'], 'seed_mode': cfg['seed_mode'],
               'GT_only_used_after_predictions_frozen': True, 'formal_benchmark_result': False,
               'metric_status': score['metrics']['status'], 'protocol_frozen': bool(raw.get('frozen', False)),
               'prediction_freeze_sha256': f.file_sha256(freeze_path), 'mapping_cache': str(mapping_path),
               'mapping_cache_sha256': mapping['cache_sha256'], 'mapping_metadata': mapping['metadata'],
               'fixed_surface_diagnostics': diagnostics, 'baseline_v3_metrics': baseline_score['metrics'],
               'objectwise_v3_metrics': score['metrics'],
               'target_v3_metrics': [{'before': before[g], 'after': after[g]} for g in sorted(target)],
               'non_target_v3_audit': harm, 'same_geometry_and_correspondence_for_every_version': True,
               'v3_code_flags_protocol_unchanged': True, 'no_parameter_tuning_to_case': True}
    f.atomic_json(out / 'evaluation_summary.json', summary)
    print(json.dumps({'status': 'PASS', 'seed_mode': cfg['seed_mode'], 'v3': score['metrics'],
                      'target_v3_metrics': summary['target_v3_metrics'], 'non_target_v3_audit': harm}, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--evaluation-config', required=True)
    args = parser.parse_args()
    main(args.config, args.evaluation_config)

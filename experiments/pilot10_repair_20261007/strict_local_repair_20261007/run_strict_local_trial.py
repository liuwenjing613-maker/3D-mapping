"""Add bounded commits and a persistent fixed-geometry diagnosis to existing trials."""
from pathlib import Path
import gc
import json
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import evaluate_pilot_v3 as e
import fixed_surface_repair as f

OUT = e.ROOT / 'strict_local_repair_20261007/room2_mask14_mask24'


def progress(stage, **extra):
    value = {'status': 'RUNNING', 'stage': stage, **extra}
    f.atomic_json(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def check_hashes(hashes):
    for path, digest in hashes.items():
        if f.file_sha256(path) != digest:
            raise RuntimeError('Protected input changed: ' + path)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = HERE / 'config.json'
    config = json.loads(config_path.read_text())
    assert config['scene'] == 'room2' and config['case_uid'] == 'room2-14047feb0df3aaa5'
    assert not config['online_causal_claim'] and not config['v3_protocol_changed']
    assert not config['quality_gate_policy_changed'] and not config['vote_confirmation_policy_changed']
    baseline_path = e.BASE / 'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
    native_root = e.BASE / 'native/room2/P1-A1/surface_p0'
    paths = {name: e.ROOT / relative for name, relative in config['conditions'].items()}
    protected = [baseline_path, native_root / 'surface_evidence.npz', native_root / 'surface_instance_frame_votes.npz',
                 native_root / 'materialization_report.json', e.ROOT / 'human_seeds_20ad6139511b/seed_manifest.json',
                 e.ROOT / 'v3/evaluation_freeze.json', e.ROOT / 'v3/baseline/room2/complete.json',
                 Path(e.__file__), e.PROTOCOL, config_path, HERE / 'fixed_surface_repair.py', Path(__file__)]
    for source in paths.values():
        protected.extend([source / 'repair_summary.json', source / 'v3_summary.json', source / 'evaluation_freeze.json',
                          source / 'whole_mask/final/instance_surface.npz', source / config['allowed_set_source'],
                          source / 'whole_mask/native/surface_evidence.npz', source / 'whole_mask/native/surface_instance_frame_votes.npz'])
    protected.extend([paths['gated'] / 'tracking/complete.json',
                      HERE.parent / 'run_joint_mask_trial.py', HERE.parent / 'joint_mask_policy.json',
                      HERE.parent / 'run_joint_mask_no_quality_gate_trial.py', HERE.parent / 'joint_no_quality_gate_policy.json'])
    hashes = {str(path): f.file_sha256(path) for path in protected}
    old_freeze = json.loads((e.ROOT / 'v3/evaluation_freeze.json').read_text())
    assert old_freeze['flags'] == e.FLAGS and old_freeze['code_sha256'] == f.file_sha256(e.__file__)
    journal_path = OUT / 'source_freeze.json'
    journal = {'status': 'FROZEN_BEFORE_STRICT_COMMIT', 'protected_source_hashes': hashes,
               'config': config, 'GT_used_for_commit': False,
               'change_scope': ['bounded final-label commits', 'persistent fixed-geometry supplementary diagnostics'],
               'unchanged': ['TSDF geometry and RGB', 'human seeds', 'SAM2 tracking and weights',
                             'old/new vote ledgers', 'native U/T/CONFIRMED/CONFLICT states',
                             'frame acceptance of either existing condition', 'confirmation thresholds',
                             'unified v3 code, flags and protocol', 'original unrestricted trial outputs'],
               'postprocessing_note': 'Existing globally computed candidates are read-only inputs; only committed final labels are bounded. No new propagation rule or inferred seed is introduced.'}
    if journal_path.exists():
        previous_journal = json.loads(journal_path.read_text())
        if previous_journal != journal:
            # A source schema guard failed before either prediction was written.
            # Preserve that failed attempt, verify every data input, and refreeze only code.
            status = json.loads((OUT / 'status.json').read_text())
            assert status['status'] == 'FAIL'
            assert not any((OUT / (name + '_strict/final/instance_surface.npz')).exists() for name in paths)
            previous_hashes = previous_journal['protected_source_hashes']
            assert set(previous_hashes) == set(hashes)
            changed = {p for p in hashes if previous_hashes[p] != hashes[p]}
            assert changed and changed.issubset({str(HERE / 'fixed_surface_repair.py'), str(Path(__file__))})
            assert previous_journal['config'] == journal['config']
            history = OUT / 'failed_precommit_attempts'
            history.mkdir(exist_ok=True)
            archived = history / ('attempt_%03d.json' % (len(list(history.glob('attempt_*.json'))) + 1))
            assert not archived.exists()
            f.atomic_json(archived, {
                'previous_source_freeze': previous_journal,
                'failure': json.loads((OUT / 'failure.json').read_text()),
                'code_revisions': {p: {'before': previous_hashes[p], 'after': hashes[p]} for p in sorted(changed)},
                'all_data_and_existing_results_unchanged': True,
                'reason': 'Separate known whole-map metadata from per-point evidence; keep exact checks for every actual evidence field and vote.'})
            f.atomic_json(journal_path, journal)
    else:
        f.atomic_json(journal_path, journal)
    if (OUT / 'complete.json').exists():
        done = json.loads((OUT / 'complete.json').read_text())
        check_hashes(hashes)
        for row in done['strict_commits'].values():
            assert f.file_sha256(row['final_map']) == row['final_map_sha256']
        assert f.file_sha256(done['mapping_cache']) == done['mapping_cache_sha256']
        print(json.dumps({'status': 'PASS', 'existing_complete_result_preserved': True}), flush=True)
        return

    baseline = f.load_arrays(baseline_path)
    baseline_evidence = f.load_arrays(native_root / 'surface_evidence.npz')
    baseline_pairs = f.load_arrays(native_root / 'surface_instance_frame_votes.npz')
    commits = {}
    candidate_paths = {'baseline': baseline_path}
    for name, source in paths.items():
        progress('validate_and_commit_bounded_labels', condition=name)
        source_report = json.loads((source / 'repair_summary.json').read_text())
        proof = source_report['arms']['whole_mask']
        assert source_report['status'] == 'PASS' and not source_report['GT_used_for_repair']
        for key in ['complete_400_frame_replay_bit_identical_to_delta_ledger', 'exact_rollback_pass',
                    'unretired_old_observation_supports_preserved_exactly', 'outside_scope_raw_evidence_bit_identical']:
            assert proof[key]
        candidate_path = source / 'whole_mask/final/instance_surface.npz'
        assert f.file_sha256(candidate_path) == proof['final_map_sha256']
        candidate = f.load_arrays(candidate_path)
        indices = np.load(source / config['allowed_set_source'], allow_pickle=False)
        repaired_evidence = f.load_arrays(source / 'whole_mask/native/surface_evidence.npz')
        repaired_pairs = f.load_arrays(source / 'whole_mask/native/surface_instance_frame_votes.npz')
        raw_proof = f.verify_raw_outside(baseline_evidence, repaired_evidence, baseline_pairs, repaired_pairs, indices)
        labels, commit_proof, blocked = f.bounded_final_labels(baseline, candidate, indices)
        assert len(indices) == proof['intervention_surface_points']
        assert len(blocked) == proof['postprocessing_changes_outside_raw_scope']
        destination = OUT / (name + '_strict')
        final_path = destination / 'final/instance_surface.npz'
        if final_path.exists():
            saved = f.load_arrays(final_path)
            f.require_same_geometry(baseline, saved)
            np.testing.assert_array_equal(saved['instance_id'], labels)
        else:
            check_hashes(hashes)
            f.atomic_npz(final_path, xyz_m=baseline['xyz_m'], rgb=baseline['rgb'], instance_id=labels)
        f.atomic_npz(destination / 'allowed_surface_ids.npz', surface_point_index=indices)
        f.atomic_npz(destination / 'blocked_outside_changes.npz', surface_point_index=blocked,
                     unrestricted_label=candidate['instance_id'][blocked], restored_baseline_label=baseline['instance_id'][blocked])
        saved = f.load_arrays(final_path)
        f.require_same_geometry(baseline, saved)
        allowed = f.allowed_mask(indices, len(labels))
        np.testing.assert_array_equal(saved['instance_id'][~allowed], baseline['instance_id'][~allowed])
        np.testing.assert_array_equal(saved['instance_id'][allowed], candidate['instance_id'][allowed])
        commit = {'status': 'PASS', **commit_proof, **raw_proof,
                  'candidate_source': str(candidate_path), 'candidate_source_sha256': f.file_sha256(candidate_path),
                  'allowed_ids_source': str(source / config['allowed_set_source']),
                  'allowed_ids_source_sha256': f.file_sha256(source / config['allowed_set_source']),
                  'allowed_ids_sha256': f.file_sha256(destination / 'allowed_surface_ids.npz'),
                  'blocked_change_log': str(destination / 'blocked_outside_changes.npz'),
                  'raw_evidence_source': str(source / 'whole_mask/native/surface_evidence.npz'),
                  'raw_vote_ledger_source': str(source / 'whole_mask/native/surface_instance_frame_votes.npz'),
                  'raw_states_and_vote_ledgers_reused_without_modification': True,
                  'final_map': str(final_path), 'final_map_sha256': f.file_sha256(final_path),
                  'complete_frame_replay_and_exact_rollback_verified_in_source_trial': True,
                  'geometry_and_RGB_bit_identical_to_baseline': True, 'GT_used_for_commit': False}
        f.atomic_json(destination / 'commit_validation.json', commit)
        commits[name + '_strict'] = commit
        candidate_paths[name + '_unrestricted'] = candidate_path
        candidate_paths[name + '_strict'] = final_path
        del candidate, repaired_evidence, repaired_pairs, labels, saved
        gc.collect()
    del baseline_evidence, baseline_pairs
    gc.collect()
    check_hashes(hashes)
    prediction_freeze = {'status': 'PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION',
                         'GT_used_for_commit': False, 'experiment_mode': config['experiment_mode'],
                         'online_causal_claim': False, 'flags': e.FLAGS,
                         'evaluator_sha256': f.file_sha256(e.__file__), 'protocol_sha256': f.file_sha256(e.PROTOCOL),
                         'effective_protocol': old_freeze['effective_protocol'],
                         'predictions': {name: {'path': str(p), 'sha256': f.file_sha256(p)} for name, p in candidate_paths.items()}}
    f.atomic_json(OUT / 'prediction_freeze.json', prediction_freeze)

    progress('evaluate_strict_versions_with_unchanged_v3')
    parser = e.ArgumentParser()
    e.add_protocol_args(parser)
    protocol, raw, debug = e.load_protocol(e.PROTOCOL, parser.parse_args(['--config', str(e.PROTOCOL), *e.FLAGS]))
    assert raw == old_freeze['effective_protocol']
    baseline_score = json.loads((e.ROOT / 'v3/baseline/room2/complete.json').read_text())
    assert baseline_score['source_map_sha256'] == f.file_sha256(baseline_path)
    scores = {'baseline': baseline_score}
    for name, source in paths.items():
        scores[name + '_unrestricted'] = json.loads((source / 'v3/whole_mask/complete.json').read_text())
        scores[name + '_strict'] = e.evaluate('room2', candidate_paths[name + '_strict'], OUT / 'v3' / (name + '_strict'), protocol)
    for name, score in scores.items():
        assert score['source_map_sha256'] == prediction_freeze['predictions'][name]['sha256']
        for key in ['flags', 'protocol_sha256', 'evaluation_code_sha256', 'GT_sha256']:
            assert score[key] == baseline_score[key]

    progress('build_or_reuse_full_TSDF_reference_mapping')
    gt = e.load_gt(e.INPUT / 'room2/ground_truth/gt.npz')
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    mapping_path = OUT / 'fixed_surface_mapping.npz'
    mapping = f.fixed_mapping(mapping_path, baseline['xyz_m'], gt.xyz_ref, config['diagnostic_max_distance_m'])
    identity = {int(pid): int(gid) for pid, gid in config['posthoc_target_identity'].items()}
    diagnostics = {}
    native_labels = {}
    for name, path in candidate_paths.items():
        progress('fixed_surface_diagnostics', condition=name)
        surface = f.load_arrays(path)
        f.require_same_geometry(baseline, surface)
        native_labels[name] = surface['instance_id']
        diagnostics[name] = f.target_diagnostics(mapping, surface['instance_id'], gt.instance_id, valid, identity)
        diagnostics[name]['baseline_to_version_transitions'] = f.target_transitions(mapping, baseline['instance_id'],
                                                                                 surface['instance_id'], gt.instance_id, valid, identity)
        diagnostics[name]['mapping_cache_sha256'] = mapping['cache_sha256']
    # Cross-check the new reusable full-scene cache against the previous independent target audit.
    prior = json.loads((paths['ungated'] / 'ungated_comparison.json').read_text())
    for row in prior['fixed_geometry_reference_diagnostic']:
        for prior_name, version in [('baseline', 'baseline'), ('gated', 'gated_unrestricted'), ('ungated', 'ungated_unrestricted')]:
            actual = next(t for t in diagnostics[version]['targets'] if t['GT_id'] == row['GT_id'])
            expected = row['states'][prior_name]
            assert actual['reachable_fixed_surface_reference_points'] == row['reachable_fixed_TSDF_reference_points']
            assert actual['reachable_label_counts'].get('52', 0) == expected['ID52']
            assert actual['reachable_label_counts'].get('355', 0) == expected['ID355']
            assert actual['unassigned_reachable_reference_points'] == expected['unassigned']
    changes_due_to_boundary = {}
    for name in paths:
        changes_due_to_boundary[name] = f.target_transitions(mapping, native_labels[name + '_unrestricted'],
                                                            native_labels[name + '_strict'], gt.instance_id, valid, identity)
    f.atomic_json(OUT / 'fixed_surface_diagnostics.json', {
        'status': 'PASS', 'diagnostic_only_not_official_v3': True, 'GT_only_used_after_predictions_frozen': True,
        'mapping_cache': str(mapping_path), 'mapping_cache_sha256': mapping['cache_sha256'],
        'mapping_metadata': mapping['metadata'], 'reference_valid_mask_sha256': f.array_sha256(valid),
        'GT_source_sha256': baseline_score['GT_sha256'], 'conditions': diagnostics,
        'unrestricted_to_strict_target_transitions': changes_due_to_boundary,
        'previous_independent_target_diagnostic_reproduced_exactly': True})

    baseline_overlap = e.per_gt(e.load_overlap(e.ROOT / 'v3/baseline/room2/overlap.npz'))
    harm = {}
    for name in commits:
        after = e.per_gt(e.load_overlap(OUT / 'v3' / name / 'overlap.npz'))
        target_gt = set(identity.values())
        harm[name] = {
            'previously_correct_unselected_lost': [g for g in baseline_overlap if g not in target_gt and baseline_overlap[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']],
            'unselected_IoU_degraded_over_0_01': [g for g in baseline_overlap if g not in target_gt and after[g]['best_IoU'] < baseline_overlap[g]['best_IoU'] - .01],
            'target_v3_metrics': [after[g] for g in sorted(target_gt)]}
    check_hashes(hashes)
    for name, row in prediction_freeze['predictions'].items():
        assert f.file_sha256(row['path']) == row['sha256']
    assert f.file_sha256(mapping_path) == mapping['cache_sha256']
    summary = {'status': 'PASS', 'case_uid': config['case_uid'], 'experiment_mode': config['experiment_mode'],
               'online_causal_claim': False, 'metric_status': baseline_score['metrics']['status'],
               'formal_benchmark_result': False, 'protocol_frozen': bool(raw.get('frozen', False)),
               'source_change_manifest': str(journal_path), 'prediction_freeze': str(OUT / 'prediction_freeze.json'),
               'strict_commits': commits, 'mapping_cache': str(mapping_path), 'mapping_cache_sha256': mapping['cache_sha256'],
               'fixed_surface_diagnostics': str(OUT / 'fixed_surface_diagnostics.json'),
               'metrics': {name: score['metrics'] for name, score in scores.items()},
               'non_target_v3_audit': harm, 'original_inputs_and_results_unchanged': True,
               'same_geometry_and_correspondence_for_every_version': True,
               'no_parameter_tuning_to_case': True, 'seconds': round(time.monotonic() - started, 2)}
    f.atomic_json(OUT / 'complete.json', summary)
    failure = OUT / 'failure.json'
    if failure.exists():
        # The previous precommit error was preserved above before this successful retry.
        assert (OUT / 'failed_precommit_attempts').is_dir()
        failure.unlink()
    f.atomic_json(OUT / 'status.json', {'status': 'PASS', 'stage': 'complete',
                                     'outside_final_label_changes': {name: 0 for name in commits},
                                     'seconds': summary['seconds']})
    print(json.dumps({'status': 'PASS', 'strict_commits': commits,
                      'metrics': summary['metrics'], 'non_target_v3_audit': harm, 'seconds': summary['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        f.atomic_json(OUT / 'failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        f.atomic_json(OUT / 'status.json', {'status': 'FAIL', 'stage': 'failure'})
        raise

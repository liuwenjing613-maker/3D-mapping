"""Post-hoc exact unified v3 and fixed-seed target/damage diagnostics."""
from pathlib import Path
from argparse import ArgumentParser
import hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT = Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0, str(SNAPSHOT))
from unified_eval.geometry import map_instances_to_reference_v3
from unified_eval.io import load_gt, save_prediction, load_prediction
from unified_eval.cli import add_protocol_args, load_protocol
from unified_eval.metrics import build_overlap, _matching
FLAGS = ['--debug-max-distance-m', '0.01', '--debug-diagnostic-max-distance-m', '0.02',
         '--debug-min-valid-instance-vertices', '100', '--debug-significant-min-vertices', '10',
         '--debug-significant-min-gt-fraction', '0.05']
PROTOCOL = SNAPSHOT / 'unified_eval/configs/replica_ca_v3.pending.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def evaluate(scene, surface, out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    checksum = sha(surface)
    complete = out / 'complete.json'
    if complete.exists():
        d = json.loads(complete.read_text())
        assert d['source_map_sha256'] == checksum and d['evaluation_code_sha256'] == sha(__file__)
        return d
    started = time.monotonic()
    with np.load(surface) as z:
        xyz, labels = z['xyz_m'], z['instance_id']
    ids = np.unique(labels[labels > 0])
    order = np.argsort(labels, kind='stable')
    sorted_ids, sorted_xyz = labels[order], xyz[order]
    clouds = [sorted_xyz[np.searchsorted(sorted_ids, i, 'left'):np.searchsorted(sorted_ids, i, 'right')] for i in ids]
    gt_path = INPUT / scene / 'ground_truth/gt.npz'
    gt = load_gt(gt_path)
    adapted = map_instances_to_reference_v3(clouds, gt.xyz_ref, .01, scene_id=scene,
        method_name='P1-A1-human-pilot10', method_commit=sha(__file__),
        adapter_version='shared_tsdf_surface_v3', protocol_version='Replica-CA-v3', diagnostic_distance_m=.02,
        metadata={'source_map_sha256': checksum, 'source_instance_surface': str(surface),
                  'debug_only': True, 'manual_seed_selection': True, 'evaluation_is_posthoc': True})
    for prediction in [adapted.prediction, adapted.diagnostic_prediction]:
        prediction.metadata['source_map_sha256'] = checksum
        for uid, instance in zip(ids, prediction.instances):
            instance.instance_uid = 'p1a1-id:' + str(int(uid))
    save_prediction(out / 'canonical_prediction.npz', adapted.prediction)
    save_prediction(out / 'diagnostic_support_prediction.npz', adapted.diagnostic_prediction)
    command = [sys.executable, '-m', 'unified_eval.cli', 'eval-scene', '--config', str(PROTOCOL), *FLAGS,
               '--gt', str(gt_path), '--pred', str(out / 'canonical_prediction.npz'),
               '--diagnostic-pred', str(out / 'diagnostic_support_prediction.npz'), '--out', str(out / 'evaluation')]
    with (out / 'evaluation.log').open('w') as log:
        subprocess.run(command, cwd=SNAPSHOT, env={**os.environ, 'PYTHONPATH': str(SNAPSHOT)},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    main = build_overlap(gt, adapted.prediction, protocol)
    diagnostic = build_overlap(gt, adapted.diagnostic_prediction, protocol)
    np.savez_compressed(out / 'overlap.npz', gt_ids=main.gt_ids,
        pred_uids=np.asarray(main.pred_uids), gt_size=main.gt_size,
        iou=main.iou, recall=main.recall, precision=main.precision,
        diagnostic_iou=diagnostic.iou, diagnostic_recall=diagnostic.recall,
        diagnostic_intersection=diagnostic.intersection, diagnostic_pred_size=diagnostic.pred_size,
        diagnostic_pred_void_fraction=diagnostic.pred_void_fraction)
    assert sha(surface) == checksum
    metrics = json.loads((out / 'evaluation/metrics.json').read_text())
    report = {'status': 'PASS', 'scene': scene, 'source_map_sha256': checksum,
              'protocol_sha256': sha(PROTOCOL), 'GT_sha256': sha(gt_path), 'flags': FLAGS,
              'evaluation_code_sha256': sha(__file__), 'adapter': adapted.statistics,
              'metrics': {k: metrics[k] for k in ['CA_AP_uniform', 'CA_AP50_uniform', 'CA_AP25_uniform',
                                               'CA_PQ', 'CA_PRF1_0_5', 'structure', 'status']},
              'seconds': round(time.monotonic() - started, 2)}
    dump(complete, report)
    print(json.dumps({'evaluated': str(surface), 'AP50': report['metrics']['CA_AP50_uniform'], 'seconds': report['seconds']}), flush=True)
    return report


def load_overlap(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def per_gt(overlap):
    ids = overlap['gt_ids']
    matched_gt = {g for _, g in _matching(overlap['iou'], .5, strict=True)}
    diag_matches = _matching(overlap['diagnostic_iou'], .5, strict=True)
    matched_pred = {p for p, _ in diag_matches}
    matched_diag_gt = {g for _, g in diag_matches}
    eligible = (overlap['diagnostic_pred_size'] > 0) & (overlap['diagnostic_pred_void_fraction'] <= .5)
    significant = (overlap['diagnostic_intersection'] >= 10) & (overlap['diagnostic_recall'] >= .05) & eligible[:, None]
    merge_pred = np.sum(significant, axis=1) >= 2
    duplicates = [p for p in range(len(eligible)) if eligible[p] and p not in matched_pred
                  and any(overlap['diagnostic_iou'][p, g] > .5 for g in matched_diag_gt)]
    result = {}
    for col, gt_id in enumerate(ids):
        best = int(np.argmax(overlap['iou'][:, col])) if len(overlap['iou']) else None
        dupe_count = sum(overlap['diagnostic_iou'][p, col] > .5 for p in duplicates if col in matched_diag_gt)
        count = int(np.sum(significant[:, col]))
        result[int(gt_id)] = {'GT_id': int(gt_id), 'GT_vertices': int(overlap['gt_size'][col]),
            'best_IoU': float(overlap['iou'][best, col]) if best is not None else 0.,
            'best_pred_uid': str(overlap['pred_uids'][best]) if best is not None else None,
            'completeness_best_single_instance': float(overlap['recall'][:, col].max(initial=0)),
            'purity_of_best_IoU_instance': float(overlap['precision'][best, col]) if best is not None else 0.,
            'matched_IoU_gt_0_5': col in matched_gt,
            'significant_predictions': count, 'split': count >= 2 and dupe_count == 0,
            'merged_predictions_touching_GT': int(np.sum(merge_pred & significant[:, col])),
            'duplicate_predictions': int(dupe_count)}
    return result


def fixed_targets(seed, baseline_overlap):
    scene, uid = seed['scene'], seed['case_uid']
    gt = load_gt(INPUT / scene / 'ground_truth/gt.npz')
    with np.load(BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as z:
        xyz = z['xyz_m']
    with np.load(ROOT / 'repair' / uid / 'seed_projected_support.npz') as z:
        points, tracks = z['surface_point_index'], z['track_id']
    distance, nearest = cKDTree(gt.xyz_ref).query(xyz[points], workers=8)
    valid = (distance < .01) & gt.valid_vertex_mask[nearest] & ~gt.ignore_vertex_mask[nearest]
    allowed = baseline_overlap['gt_ids']
    targets, rows = set(), []
    for obj in seed['objects']:
        oid = obj['track_id']
        keep = valid & (tracks == oid) & np.isin(gt.instance_id[nearest], allowed)
        gids, counts = np.unique(gt.instance_id[nearest[keep]], return_counts=True)
        total = max(1, int(np.sum(keep)))
        significant = (counts >= 10) & (counts / total >= .05)
        targets.update(int(i) for i in gids[significant])
        rows.append({'track_id': oid, 'native_mask_id': obj['native_mask_id'],
            'nearest_valid_native_points': int(np.sum(keep)),
            'GT_overlap': [{'GT_id': int(g), 'native_points': int(c), 'fraction': float(c / total),
                            'included': bool(s)} for g, c, s in zip(gids, counts, significant)]})
    return sorted(targets), rows


def compare_case(seed, results, protocol):
    scene, uid = seed['scene'], seed['case_uid']
    base_out = ROOT / 'v3/baseline' / scene
    base = results['baseline:' + scene]
    before_overlap = load_overlap(base_out / 'overlap.npz')
    before = per_gt(before_overlap)
    if seed['status'] != 'READY':
        return {'case_uid': uid, 'scene': scene, 'ROI': seed['ROI'], 'status': 'SKIP_UNRELIABLE',
                'included_in_denominator': True, 'outcome': 'unchanged', 'targets': [],
                'conditions': {c: {'scene_metrics': base['metrics'], 'outcome': 'unchanged',
                                   'AP50_delta': 0., 'F1_delta': 0., 'PQ_delta': 0.,
                                   'target_mean_IoU_delta': 0., 'previously_correct_unselected_lost': 0,
                                   'strict_success': False} for c in ['seed_only', 'seed_plus_short_track']}}
    targets, target_source = fixed_targets(seed, before_overlap)
    report = {'case_uid': uid, 'scene': scene, 'ROI': seed['ROI'], 'status': 'PASS',
              'choice': seed['source_choice']['choice'], 'objects': len(seed['objects']),
              'targets': targets, 'target_source': target_source,
              'target_definition': 'fixed seed native support; nearest valid GT within 1cm; >=10 hits and >=5% per seed',
              'conditions': {}}
    for condition in ['seed_only', 'seed_plus_short_track']:
        out = ROOT / 'v3' / uid / condition
        current = results[uid + ':' + condition]
        after = per_gt(load_overlap(out / 'overlap.npz'))
        assert before.keys() == after.keys()
        rows = [{'before': before[g], 'after': after[g]} for g in targets]
        iou_delta = float(np.mean([after[g]['best_IoU'] - before[g]['best_IoU'] for g in targets])) if targets else None
        lost = [g for g in before if g not in targets and before[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']]
        degraded = [g for g in before if g not in targets and after[g]['best_IoU'] < before[g]['best_IoU'] - .01]
        structural_delta = sum(int(after[g]['split']) + after[g]['merged_predictions_touching_GT'] + after[g]['duplicate_predictions']
                               - int(before[g]['split']) - before[g]['merged_predictions_touching_GT'] - before[g]['duplicate_predictions'] for g in targets)
        m, b = current['metrics'], base['metrics']
        f1_delta = m['CA_PRF1_0_5']['F1'] - b['CA_PRF1_0_5']['F1']
        pq_delta = m['CA_PQ']['PQ'] - b['CA_PQ']['PQ']
        improvement = iou_delta is not None and (iou_delta >= .01 or structural_delta < 0) and iou_delta >= -1e-8
        regression = bool(lost) or (iou_delta is not None and iou_delta < -.01) or structural_delta > 0
        success = improvement and not regression and not degraded and f1_delta >= -1e-8 and pq_delta >= -1e-8
        outcome = 'improved' if success else 'mixed' if improvement and regression else 'regressed' if regression else 'small_or_unchanged'
        geometry = json.loads((ROOT / 'repair' / uid / condition / 'complete.json').read_text())
        report['conditions'][condition] = {'scene_metrics': m, 'AP50_delta': m['CA_AP50_uniform'] - b['CA_AP50_uniform'],
            'F1_delta': f1_delta, 'PQ_delta': pq_delta, 'target_mean_IoU_delta': iou_delta,
            'target_structure_count_delta': structural_delta,
            'previously_correct_unselected_lost': len(lost), 'lost_unselected_GT_ids': lost,
            'unselected_IoU_degraded_over_0_01': degraded, 'target_objects': rows,
            'strict_success': success, 'outcome': outcome,
            'native_label_changes': geometry['native_label_changes'], 'final_label_changes': geometry['final_label_changes'],
            'final_changes_outside_intervention': geometry['final_changes_outside_intervention']}
    dump(ROOT / 'v3' / uid / 'case_comparison.json', report)
    return report


def main():
    out = ROOT / 'v3'
    out.mkdir(exist_ok=True)
    while not (ROOT / 'repair/status.json').exists() or json.loads((ROOT / 'repair/status.json').read_text())['status'] != 'PASS':
        if (ROOT / 'repair/failure.json').exists():
            raise RuntimeError('Repair failed; predictions not all frozen')
        time.sleep(10)
    manifest_path = ROOT / 'human_seeds_20ad6139511b/seed_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads((BASE / 'run_manifest.json').read_text())
    assert FLAGS == baseline['v3_flags']
    assert sha(PROTOCOL) == '0863fc50a69b8773988bc588a4b71253fdd6caf09e65ab2217195894fcf79ede'
    parser = ArgumentParser()
    add_protocol_args(parser)
    protocol, raw, debug = load_protocol(PROTOCOL, parser.parse_args(['--config', str(PROTOCOL), *FLAGS]))
    jobs = {}
    for seed in manifest['seeds']:
        scene, uid = seed['scene'], seed['case_uid']
        jobs['baseline:' + scene] = (scene, BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz', out / 'baseline' / scene)
        if seed['status'] == 'READY':
            for condition in ['seed_only', 'seed_plus_short_track']:
                jobs[uid + ':' + condition] = (scene, ROOT / 'repair' / uid / condition / 'final/instance_surface.npz', out / uid / condition)
    freeze = {'status': 'FROZEN_BEFORE_GT', 'predictions': {key: {'path': str(job[1]), 'sha256': sha(job[1])} for key, job in jobs.items()},
              'manifest_sha256': sha(manifest_path), 'code_sha256': sha(__file__), 'flags': FLAGS,
              'effective_protocol': raw, 'GT_used_for_repair': False}
    dump(out / 'evaluation_freeze.json', freeze)
    status = {'status': 'RUNNING', 'total_maps': len(jobs), 'completed_maps': [],
              'flags': FLAGS, 'predictions_frozen_before_GT': True}
    dump(out / 'status.json', status)
    results = {}
    for key, (scene, surface, destination) in jobs.items():
        report = evaluate(scene, surface, destination, protocol)
        results[key] = report
        status['completed_maps'].append(key)
        dump(out / 'status.json', status)
    cases = [compare_case(seed, results, protocol) for seed in manifest['seeds']]
    aggregate = {}
    for condition in ['seed_only', 'seed_plus_short_track']:
        values = [c['conditions'][condition] for c in cases]
        aggregate[condition] = {'strict_success_cases': sum(r['strict_success'] for r in values), 'denominator': 10,
            'outcomes': {key: sum(r['outcome'] == key for r in values) for key in sorted(set(r['outcome'] for r in values))},
            'case_mean_scene_AP50_delta': float(np.mean([r['AP50_delta'] for r in values])),
            'case_mean_scene_F1_delta': float(np.mean([r['F1_delta'] for r in values])),
            'case_mean_scene_PQ_delta': float(np.mean([r['PQ_delta'] for r in values])),
            'lost_previously_correct_unselected_count_across_independent_cases': sum(r['previously_correct_unselected_lost'] for r in values)}
    for key, record in freeze['predictions'].items():
        assert sha(record['path']) == record['sha256']
    original_seed = next(s for s in manifest['seeds'] if s.get('source_choice', {}).get('choice') == 'original')
    original_key = original_seed['case_uid'] + ':seed_only'
    assert results[original_key]['metrics'] == results['baseline:' + original_seed['scene']]['metrics']
    summary = {'status': 'PASS', 'evaluation': 'exact frozen unified v3 debug preset; 1cm main, 2cm structure',
        'metric_status': results[original_key]['metrics']['status'], 'flags': FLAGS,
        'all_predictions_unchanged_after_evaluation': True, 'original_mask_seed_control_v3_identical': True,
        'manual_selected_pilot': True, 'automatic_selection_performance_claimed': False,
        'primary_comparison': '10 independent cases each starting from baseline; scene deltas are case averages, not a pooled combined-scene score',
        'strict_success_rule': 'target mean IoU +0.01 or fewer structural errors, no target regression, no unselected IoU loss >0.01, no F1/PQ regression',
        'aggregate': aggregate, 'cases': cases}
    dump(out / 'summary.json', summary)
    status['status'] = 'PASS'
    dump(out / 'status.json', status)
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(ROOT / 'v3/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise

"""Re-score fixed canonical maps without changing methods, maps, or V3 settings."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import os
import sys

SNAPSHOT = Path(os.environ.get('CVPR_EVALUATOR_ROOT', '/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot'))
ROOT = Path(os.environ.get('CVPR_DATA_ROOT', '/data/chenkejun/CVPR'))
OUT = Path(os.environ.get('CVPR_EVAL_AUDIT_OUT', str(ROOT / 'results/evaluation_audit_20261007')))
P1 = ROOT / 'revisable_instance_map/p1a1_raw_replica8_20261002'
OVI = ROOT / 'evaluation_results/p0_vs_ovimap_v3_20260929'
OPENVOX = ROOT / 'openvox_reproduction_20261002/evaluation_v3_tsdf_surface2000'
AUTO = ROOT / 'revisable_instance_map/p1a1_auto_v2_apply_20261004'
SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(actual, expected, location):
    if isinstance(expected, dict):
        for k, value in expected.items():
            assert k in actual, f'{location}/{k}: missing'
            compare(actual[k], value, f'{location}/{k}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f'{location}: list size'
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f'{location}/{i}')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), f'{location}: {actual} != {expected}'
    else:
        assert actual == expected, f'{location}: {actual!r} != {expected!r}'


def aggregate(rows, protocol, original_status):
    import numpy as np
    ap = {}
    for key in rows[0]['AP_by_threshold']:
        counts = {name: sum(x['AP_by_threshold'][key][name] for x in rows) for name in ['TP', 'FP', 'FN']}
        tp, fp, fn = (counts[x] for x in ['TP', 'FP', 'FN'])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        counts['ap'] = float(np.mean([precision if level <= recall else 0.0 for level in np.linspace(0, 1, 101)]))
        counts['status'] = 'ok'
        ap[key] = counts
    counts = {name: sum(x['CA_PRF1_0_5'][name] for x in rows) for name in ['TP', 'FP', 'FN', 'ignored_prediction_count']}
    tp, fp, fn = (counts[x] for x in ['TP', 'FP', 'FN'])
    denom = tp + 0.5 * fp + 0.5 * fn
    prf = {**counts, 'P': tp / (tp + fp) if tp + fp else 0.0, 'R': tp / (tp + fn), 'F1': tp / denom, 'status': 'ok'}
    sum_iou = sum(x['CA_PQ']['SQ'] * x['CA_PQ']['TP'] for x in rows)
    pq = {**counts, 'PQ': sum_iou / denom, 'SQ': sum_iou / tp if tp else 0.0, 'RQ': tp / denom, 'status': 'ok'}
    structure = {name: sum(x['structure'][name] for x in rows) for name in ['split_gt_count', 'merge_prediction_count', 'duplicate_prediction_count', 'diagnostic_gt_count', 'diagnostic_prediction_count']}
    structure.update(split_gt_rate=structure['split_gt_count'] / structure['diagnostic_gt_count'],
                     merge_prediction_rate=structure['merge_prediction_count'] / structure['diagnostic_prediction_count'],
                     duplicate_prediction_rate=structure['duplicate_prediction_count'] / structure['diagnostic_prediction_count'])
    mean = lambda values: sum(values) / len(values)
    macro = {name: mean([x[name] for x in rows]) for name in ['CA_AP_uniform', 'CA_AP50_uniform', 'CA_AP25_uniform']}
    macro.update(CA_F1_0_5=mean([x['CA_PRF1_0_5']['F1'] for x in rows]), CA_PQ=mean([x['CA_PQ']['PQ'] for x in rows]))
    macro.update({name: mean([x['structure'][name] for x in rows]) for name in ['split_gt_rate', 'merge_prediction_rate', 'duplicate_prediction_rate']})
    return {'protocol': protocol.name, 'dataset': protocol.dataset, 'confidence_mode': 'uniform',
            'scene_count': len(rows), 'status': original_status,
            'CA_AP_uniform': mean([v['ap'] for k, v in ap.items() if k != '0.25']),
            'CA_AP50_uniform': ap['0.50']['ap'], 'CA_AP25_uniform': ap['0.25']['ap'],
            'AP_by_threshold': ap, 'CA_PQ': pq, 'CA_PRF1_0_5': prf,
            'structure': structure, 'structure_status': 'ok', 'macro_per_scene': macro,
            'aggregation': 'top-level pooled predictions/GT; macro_per_scene is unweighted scene mean', 'per_scene': rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run = read(P1 / 'run_manifest.json')
    code_hashes = {name: value for name, value in run['source_sha256'].items() if name.startswith('unified_eval/') and '/tests/' not in name}
    for name, expected in code_hashes.items():
        assert digest(SNAPSHOT / name) == expected, f'frozen evaluator changed: {name}'
    sys.path.insert(0, str(SNAPSHOT))
    from unified_eval.io import load_gt, load_prediction
    from unified_eval.schema import Protocol
    from unified_eval.evaluate import evaluate_scenes
    import numpy as np

    definitions = [
        ('OVI-MAP_full400', OVI / 'OVI-MAP/batch/summary.json', lambda scene: OVI / scene / 'OVI-MAP/evaluation'),
        ('P1-A1_native', P1 / 'pooled/native/summary.json', lambda scene: P1 / 'native' / scene / 'P1-A1/v3/evaluation'),
        ('P1-A1_final', P1 / 'pooled/final/summary.json', lambda scene: P1 / 'final' / scene / 'P1-A1/v3/evaluation'),
        ('P1-A1_auto_revision_20261004', AUTO / 'pooled/summary.json', lambda scene: AUTO / scene / 'v3/evaluation'),
        ('OpenVox_TSDF_surface_reference', OPENVOX / 'pooled/summary.json', lambda scene: OPENVOX / scene / 'evaluation'),
    ]
    gt_hashes = {}
    hash_cache = {}
    previous_alignment_audit = {x['scene']: x['aligned_frames_audit'] for x in read(OVI / 'comparison.json')['scenes']}
    protocol_hash = None
    protocol_config = None
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
              'scope': 'Re-score previously fixed canonical predictions and diagnostics; original native-to-reference geometry adaptation is hash checked but not rerun.',
              'source_evaluator_snapshot': str(SNAPSHOT), 'source_evaluator_files_sha256': code_hashes,
              'methods': []}
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for method, summary_path, eval_dir in definitions:
        assert summary_path.is_file(), str(summary_path)
        original = read(summary_path)
        assert original['scene_count'] == 8
        original_scene_rows = {x['scene_id']: x for x in original['per_scene']}
        assert set(original_scene_rows) == set(SCENES)
        rows, evidence = [], []
        for scene in SCENES:
            path = eval_dir(scene)
            # Revision variants may retain the shared surface adapter subdirectory.
            if not (path / 'manifest.json').is_file():
                candidates = list((AUTO / scene).rglob('manifest.json')) if method.startswith('P1-A1_auto') else []
                candidates = [p for p in candidates if 'prediction_file' in read(p) and 'protocol_config' in read(p)]
                assert len(candidates) == 1, f'{method}/{scene}: cannot locate single evaluation'
                path = candidates[0].parent
            manifest = read(path / 'manifest.json')
            source = manifest['prediction_source']
            for file_key, hash_key in [('source_instance_surface', 'source_map_sha256'), ('source_export', 'source_export_sha256'), ('source_export_manifest', 'source_export_manifest_sha256')]:
                if source.get(file_key) and source.get(hash_key):
                    p = Path(source[file_key])
                    if str(p) not in hash_cache:
                        hash_cache[str(p)] = digest(p)
                    assert hash_cache[str(p)] == source[hash_key], f'{method}/{scene}/{file_key}: native source changed'
            current_hash = manifest['effective_protocol_sha256']
            if protocol_hash is None:
                protocol_hash = current_hash
                protocol_config = manifest['protocol_config']
            assert current_hash == protocol_hash
            assert manifest['protocol_config'] == protocol_config
            assert manifest['scene_id'] == scene
            protocol = Protocol.from_dict(manifest['protocol_config'])
            file_hashes = {}
            for file_key, hash_key in [('gt_file', 'gt_sha256'), ('prediction_file', 'prediction_sha256'), ('diagnostic_prediction_file', 'diagnostic_prediction_sha256')]:
                p = Path(manifest[file_key])
                actual = hash_cache.setdefault(str(p), digest(p)) if str(p) not in hash_cache else hash_cache[str(p)]
                assert actual == manifest[hash_key], f'{method}/{scene}/{file_key}: hash changed'
                file_hashes[file_key] = actual
            if scene in gt_hashes:
                assert manifest['gt_sha256'] == gt_hashes[scene], f'{method}/{scene}: different GT'
            gt_hashes[scene] = manifest['gt_sha256']
            gt = load_gt(manifest['gt_file'])
            pred = load_prediction(manifest['prediction_file'])
            diag = load_prediction(manifest['diagnostic_prediction_file'])
            assert pred.is_partition
            single, scene_rows, overlap = evaluate_scenes([(gt, pred)], protocol, diagnostic_predictions=[diag])
            row = json.loads(json.dumps(scene_rows[0], default=lambda value: value.item()))
            compare(row, original_scene_rows[scene], f'{method}/{scene}')
            with np.load(path / 'overlap_matrix.npz', allow_pickle=False) as stored:
                for name in ['gt_ids', 'intersection', 'iou', 'precision', 'recall', 'pred_size', 'gt_size', 'pred_void_fraction']:
                    a, b = np.asarray(getattr(overlap[0], name)), stored[name]
                    assert np.array_equal(a, b), f'{method}/{scene}/{name}: stored overlap differs'
                assert overlap[0].pred_uids == stored['pred_uids'].tolist()
            item = {'scene': scene, 'evaluation_dir': str(path), 'manifest_sha256': digest(path / 'manifest.json'),
                    'files_sha256': file_hashes, 'frame_count': manifest['prediction_source'].get('frame_count_from_source'),
                    'prediction_source': manifest['prediction_source'], 'gt_instances': row['CA_PRF1_0_5']['TP'] + row['CA_PRF1_0_5']['FN'],
                    'reference_vertices': gt.vertex_count, 'original_metrics_match': True, 'rebuilt_overlap_matches': True}
            if method == 'OVI-MAP_full400':
                frames = sorted(int(p.name.split('_', 1)[0]) for p in (Path(os.environ.get('CVPR_OVI_NATIVE_RESULTS', '/data/chenkejun/ovimap_runtime_20260908/results')) / (scene + '_full400') / 'geometrics').glob('*_mask.png'))
                assert frames == run['frame_ids'] == list(range(0, 2000, 5))
                frame_sha = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
                assert source['frame_list_sha256'] == frame_sha
                historic = previous_alignment_audit[scene]
                assert historic['all_verified'] and historic['frames_match'] == historic['poses_match_atol_1e_6'] == historic['intrinsics_match'] == historic['decoded_rgb_hashes_match'] == 400
                item['input_alignment'] = {'frame_list_sha256': frame_sha, 'frame_ids_match_P1A1': True, 'historic_alignment_audit': historic,
                                           'alignment_scope': 'Current native mask filenames verify exact frames; prior per-frame RGB/pose/intrinsics audit is reused after checking native source-map hashes.'}
                align_path = source.get('source_input_alignment')
                if align_path:
                    align = read(align_path)
                    assert align['source_frames'] == frames
                    assert align['rgb_decoded_equal'] and align['poses_equal_atol_1e_6'] and align['intrinsics_equal']
                    item['input_alignment']['input_alignment_sha256'] = digest(align_path)
            rows.append(row)
            evidence.append(item)
            print(json.dumps({'method': method, 'scene': scene, 'status': 'PASS', 'AP50': row['CA_AP50_uniform'], 'F1': row['CA_PRF1_0_5']['F1']}), flush=True)
            del gt, pred, diag, overlap, single
            gc.collect()
        rebuilt = aggregate(rows, protocol, original['status'])
        compare(rebuilt, original, method + '/pooled')
        target = OUT / (method + '_summary.json')
        target.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2))
        audit = {'method': method, 'source_summary': str(summary_path), 'source_summary_sha256': digest(summary_path),
                 'recomputed_summary': str(target), 'original_pooled_metrics_match': True, 'scene_count': 8, 'evidence': evidence}
        report['methods'].append(audit)
        report.update(effective_protocol_sha256=protocol_hash, effective_protocol_config=protocol_config, gt_sha256_by_scene=gt_hashes)
        (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({'method': method, 'status': 'POOLED_PASS', 'AP50': rebuilt['CA_AP50_uniform']}), flush=True)
    report['status'] = 'PASS'
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['total_scene_evaluations'] = len(definitions) * len(SCENES)
    report['all_methods_share_identical_protocol_and_GT'] = True
    report['OVI_MAP_and_P1A1_have_identical_400_frame_ids'] = True
    report['P1A1_method_definition'] = run['method_definition']
    report['P1A1_allow_identity_repair'] = run['allow_identity_repair']
    report['OpenVox_comparison_limit'] = 'Different input budget: 2000 geometry frames and 200 instance opportunities per scene; reference only.'
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('ALL_PASS', OUT, flush=True)


if __name__ == '__main__':
    main()

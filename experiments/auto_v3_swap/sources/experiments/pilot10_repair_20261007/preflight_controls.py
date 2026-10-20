"""Real frozen P1-A1 votes: exact republication, no-op replacement and undo."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, traceback
import numpy as np
from evidence_replacement import EvidenceReplacement

BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0, str(SNAPSHOT / 'revisable_instance_map/src'))
from revisable_instance_map.surface_evidence import reduce_surface_votes, frame_instance_keys, CONFIRMED


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def verify_scene(scene, frames, baseline_manifest):
    started = time.monotonic()
    native = BASE / f'native/{scene}/P1-A1/surface_p0'
    final = BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
    report_path = native / 'materialization_report.json'
    report = json.loads(report_path.read_text())
    associations = BASE / f'native/{scene}/P1-A1/association/associations.jsonl'
    observations = BASE / f'raw_inputs/{scene}/observations/observations.jsonl'
    required = [native / 'surface_evidence.npz', native / 'surface_instance_frame_votes.npz',
                native / 'instance_surface.npz', final, report_path, associations, observations]
    before = {str(p): sha(p) for p in required}
    assert before[str(associations)] == report['association_decisions_sha256']
    assert before[str(observations)] == report['observation_catalog_sha256']
    assert report['frame_count'] == 400 and not report['ground_truth_used']
    records = {int(r['frame_id']): r for r in report['frame_records']}
    assert sorted(records) == baseline_manifest['frame_ids']
    lookup = {}
    max_instance = 0
    for line in associations.read_text().splitlines():
        row = json.loads(line)
        max_instance = max(max_instance, int(row['instance_id']))
        if row['frame_id'] in frames:
            key = (row['frame_id'], row['mask_local_id'])
            assert key not in lookup
            lookup[key] = int(row['instance_id'])
    # Reserve a single extra ID for a synthetic undo test, never for a repair.
    instance_base = max_instance + 2
    with np.load(native / 'surface_instance_frame_votes.npz', allow_pickle=False) as z:
        keys = z['surface_point_index'].astype(np.int64) * instance_base + z['instance_id']
        votes = z['frame_votes'].astype(np.int64)
    order = np.argsort(keys, kind='stable')
    keys, votes = keys[order], votes[order]
    with np.load(native / 'surface_evidence.npz', allow_pickle=False) as z:
        n = len(z['state'])
        expected = {k: z[k] for k in ['top1_instance_id', 'top1_votes', 'top2_instance_id', 'top2_votes',
                                    'total_frame_votes', 'confidence', 'margin', 'state']}
    actual = reduce_surface_votes(keys, votes, n, instance_base,
                                  report['min_confirmed_votes'], report['min_confirmed_ratio'])
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=scene + ':' + name)
    with np.load(native / 'instance_surface.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(np.where(actual['state'] == CONFIRMED, actual['top1_instance_id'], -1), z['instance_id'])
    ledger = EvidenceReplacement(keys, votes)
    baseline_digest = ledger.digest()
    controls = []
    for fid in sorted(frames):
        r = records[fid]
        path = Path(r['support_file'])
        assert sha(path) == r['support_sha256']
        with np.load(path, allow_pickle=False) as z:
            points, local = z['surface_point_index'], z['mask_local_id']
            assert int(z['frame_id'][0]) == fid
            table = np.full(int(local.max()) + 1, -1, np.int32)
            for mid in np.unique(local):
                table[mid] = lookup[(fid, int(mid))]
            old = frame_instance_keys(points, local, table, instance_base)
        assert len(old) == r['frame_instance_votes']
        position = np.searchsorted(keys, old)
        assert np.all(position < len(keys)) and np.array_equal(keys[position], old)
        assert np.all(votes[position] >= 1)
        transaction = scene + ':old-evidence:' + str(fid)
        assert ledger.replace(transaction, old, old)
        assert ledger.digest() == baseline_digest
        assert not ledger.replace(transaction, old, old)
        ledger.undo(transaction)
        assert ledger.digest() == baseline_digest
        controls.append({'frame': fid, 'original_frame_votes': len(old), 'replacement': 'PASS', 'repeat': 'PASS', 'undo': 'PASS'})
    # Exercise nontrivial retraction and insertion on actual frame support.
    take = old[:min(64, len(old))]
    changed = np.unique((take // instance_base) * instance_base + instance_base - 1)
    assert ledger.replace(scene + ':synthetic-undo-test', take, changed)
    altered_digest = ledger.digest()
    assert altered_digest != baseline_digest
    assert not ledger.replace(scene + ':synthetic-undo-test', take, changed)
    assert ledger.digest() == altered_digest
    ledger.undo(scene + ':synthetic-undo-test')
    assert ledger.digest() == baseline_digest
    missing = np.asarray([n * instance_base + 1], np.int64)
    try:
        ledger.replace(scene + ':invalid-test', missing, missing)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing old evidence was accepted')
    assert ledger.digest() == baseline_digest
    for path, digest in before.items():
        assert sha(path) == digest
    return {'scene': scene, 'status': 'PASS', 'surface_points': n, 'sparse_vote_pairs': len(keys),
            'native_evidence_arrays_reproduced': list(expected), 'native_publication_reproduced': True,
            'original_frame_controls': controls, 'nontrivial_undo_and_repeat': 'PASS',
            'invalid_retraction_rejected': True, 'source_hashes': before,
            'original_files_unchanged': True, 'ground_truth_used': False,
            'final_diffusion_recomputed': False, 'human_repair_run': False,
            'seconds': round(time.monotonic() - started, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    baseline = json.loads((BASE / 'run_manifest.json').read_text())
    assert baseline['status'] == 'PASS'
    module = SNAPSHOT / 'revisable_instance_map/src/revisable_instance_map/surface_evidence.py'
    assert sha(module) == baseline['source_sha256']['revisable_instance_map/src/revisable_instance_map/surface_evidence.py']
    scenes = {}
    for case in plan['pilot_cases']:
        scenes.setdefault(case['scene'], set()).update(v['frame'] for v in case['fixed_views'])
    status = {'status': 'RUNNING', 'human_selection_status': 'WAITING_FOR_EXPORT',
              'completed_scenes': [], 'total_scenes': len(scenes), 'GT_used': False,
              'source_plan_sha256': sha(args.plan), 'module_sha256': sha(module),
              'experiment_phase': 'native_evidence_and_transaction_preflight'}
    dump(args.out / 'status.json', status)
    try:
        for scene, frames in scenes.items():
            print(json.dumps({'scene': scene, 'status': 'STARTING', 'frames': sorted(frames)}), flush=True)
            report = verify_scene(scene, frames, baseline)
            dump(args.out / (scene + '.json'), report)
            status['completed_scenes'].append({'scene': scene, 'status': 'PASS', 'frames': len(frames), 'seconds': report['seconds']})
            dump(args.out / 'status.json', status)
            print(json.dumps(status['completed_scenes'][-1]), flush=True)
        status['status'] = 'PASS'
        status['human_repair_run'] = False
        status['full_diffusion_and_v3_controls_pending'] = True
        dump(args.out / 'status.json', status)
    except Exception:
        status['status'] = 'FAIL'
        status['error'] = traceback.format_exc()
        dump(args.out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

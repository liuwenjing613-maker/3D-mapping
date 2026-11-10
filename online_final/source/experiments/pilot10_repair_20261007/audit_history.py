"""Prepare scoped historical reassociation transactions; do not apply them."""
from pathlib import Path
from dataclasses import replace
import json, time, traceback
import numpy as np
import run_repairs as repair


def prepare_case(data, seed):
    r = repair
    uid = seed['case_uid']
    association = json.loads((r.ROOT / 'repair' / uid / 'frozen_association.json').read_text())
    base = association['instance_base']
    table = np.full(len(seed['objects']) + 1, -1, np.int32)
    for obj in association['objects']:
        table[obj['track_id']] = obj['persistent_id']
    records = json.loads((r.ROOT / 'tracking' / uid / 'complete.json').read_text())['frames']
    records = {row['frame']: row for row in records}
    arrays = []
    for frame_id in seed['mapping_vote_frames']:
        row = records[frame_id]
        assert r.sha(row['label_file']) == row['label_sha256']
        mask = np.array(r.Image.open(row['label_file']), np.uint16)
        frame = data.source.load_frame(frame_id)
        points, local, _ = r.project_frame_regions(replace(frame, mask_local=mask), data.tree,
                                                  pixel_stride=2, max_distance_m=.015, workers=8)
        arrays.append(r.frame_instance_keys(points, local, table, base))
    keys, counts = np.unique(np.concatenate(arrays), return_counts=True)
    evidence = r.reduce_surface_votes(keys, counts, len(data.xyz), base, min_confirmed_votes=2, min_confirmed_ratio=.8)
    support = r.load_npz(r.ROOT / 'repair' / uid / 'seed_projected_support.npz')
    seed_keys = np.unique(support['surface_point_index'].astype(np.int64) * base + table[support['track_id']])
    points, starts, num = np.unique(seed_keys // base, return_index=True, return_counts=True)
    unambiguous = num == 1
    anchored = np.full(len(data.xyz), -1, np.int32)
    anchored[points[unambiguous]] = (seed_keys[starts[unambiguous]] % base).astype(np.int32)
    trusted = (anchored > 0) & (evidence['state'] == r.CONFIRMED) & (evidence['top1_instance_id'] == anchored)
    owner = np.where(trusted, anchored, -1).astype(np.int32)
    out = r.ROOT / 'history_audit' / uid
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'trusted_seed_surface_owner.npy', owner)
    objects = [{'track_id': o['track_id'], 'native_mask_id': o['native_mask_id'], 'persistent_id': o['persistent_id'],
                'trusted_seed_surface_points': int(np.sum(owner == o['persistent_id']))} for o in association['objects']]
    return {'case_uid': uid, 'scene': seed['scene'], 'ROI': seed['ROI'], 'base': base, 'owner': owner,
            'out': out, 'window': set(seed['mapping_vote_frames']), 'objects': objects, 'frames': [],
            'seed_points': len(points), 'trusted_points': int(np.sum(trusted)),
            'conflicting_vote_mass': 0, 'conflicting_vote_mass_outside_short_window': 0}


def main():
    r = repair
    root = r.ROOT / 'history_audit'
    root.mkdir(exist_ok=True)
    manifest = json.loads((r.SEEDS / 'seed_manifest.json').read_text())
    protocol = {'status': 'FROZEN_BEFORE_HISTORY_AUDIT', 'purpose': 'reviewable historical association proposals',
        'scope': 'human seed surfaces only; unique seed owner; >=2 mapping frames support same owner with >=0.8 agreement',
        'minimum_supporting_frames': 2, 'minimum_track_consensus_ratio': .8,
        'frame_list': 'unchanged range(0,2000,5)', 'vote_weight': 1,
        'target_3D_association': 'reuse exactly the frozen mapping of the first experiment',
        'GT_used': False, 'predictions_or_original_votes_modified': False,
        'original_CF_case': 'keep as no-op control',
        'new_experiment_required_before_claiming_improvement': True,
        'code_sha256': r.sha(__file__), 'source_seed_manifest_sha256': r.sha(r.SEEDS / 'seed_manifest.json')}
    r.dump(root / 'protocol.json', protocol)
    ready = [s for s in manifest['seeds'] if s['status'] == 'READY' and s['source_choice']['choice'] != 'original']
    scenes = {}
    for seed in ready:
        scenes.setdefault(seed['scene'], []).append(seed)
    completed = []
    r.dump(root / 'status.json', {'status': 'RUNNING', 'completed_cases': 0, 'total_cases': len(ready), 'GT_used': False})
    for scene, seeds in scenes.items():
        data = r.Scene(scene)
        cases = [prepare_case(data, s) for s in seeds]
        for fid, record in sorted(data.records.items()):
            support_path = Path(record['support_file'])
            assert r.sha(support_path) == record['support_sha256']
            support = r.load_npz(support_path)
            points, local = support['surface_point_index'], support['mask_local_id']
            lookup = np.full(int(local.max()) + 1, -1, np.int32)
            for mid in np.unique(local):
                lookup[mid] = data.lookup[(fid, int(mid))]
            for case in cases:
                owner, base = case['owner'], case['base']
                old = r.frame_instance_keys(points, local, lookup, base)
                in_scope = owner[old // base] > 0
                old_scope = old[in_scope]
                p = old_scope // base
                conflicting = (old_scope % base) != owner[p]
                mass = int(np.sum(conflicting))
                if mass == 0:
                    continue
                new_scope = np.unique(p * base + owner[p])
                removed = np.setdiff1d(old_scope, new_scope)
                added = np.setdiff1d(new_scope, old_scope)
                assert np.all(owner[removed // base] > 0) and np.all(owner[added // base] > 0)
                assert len(new_scope) <= len(old_scope)
                filename = case['out'] / f'f{fid:06d}_proposed_transaction.npz'
                np.savez_compressed(filename, old_keys=old_scope, new_keys=new_scope, instance_base=base)
                case['conflicting_vote_mass'] += mass
                if fid not in case['window']:
                    case['conflicting_vote_mass_outside_short_window'] += mass
                case['frames'].append({'frame': fid, 'inside_short_window': fid in case['window'],
                    'old_in_scope_votes': len(old_scope), 'new_in_scope_votes': len(new_scope),
                    'removed_identity_pairs': len(removed), 'inserted_identity_pairs': len(added),
                    'conflicting_with_frozen_seed_identity_votes': mass,
                    'source_support_sha256': record['support_sha256'], 'proposed_transaction_sha256': r.sha(filename)})
        data.verify_unchanged()
        for case in cases:
            report = {k: v for k, v in case.items() if k not in ['owner', 'out', 'window']}
            mass = case['conflicting_vote_mass']
            report['outside_short_window_fraction'] = case['conflicting_vote_mass_outside_short_window'] / mass if mass else 0
            report['status'] = 'PROPOSAL_ONLY'
            report['GT_used'] = False
            report['predictions_or_original_votes_modified'] = False
            r.dump(case['out'] / 'historical_plan.json', report)
            completed.append({k: report[k] for k in ['case_uid', 'scene', 'ROI', 'trusted_points', 'seed_points',
                'conflicting_vote_mass', 'conflicting_vote_mass_outside_short_window', 'outside_short_window_fraction']})
            r.dump(root / 'status.json', {'status': 'RUNNING', 'completed_cases': len(completed), 'total_cases': len(ready), 'GT_used': False})
            print(json.dumps(completed[-1]), flush=True)
        del data
    r.dump(root / 'summary.json', {'status': 'PASS', 'cases': completed, 'scope_protocol': protocol,
        'new_predictions_generated': False, 'all_original_inputs_unchanged': True})
    r.dump(root / 'status.json', {'status': 'PASS', 'completed_cases': len(completed), 'total_cases': len(ready), 'GT_used': False})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        repair.dump(repair.ROOT / 'history_audit/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise

"""Read-only post-hoc explanation of unresolved votes and removed-support states."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'strict_local_repair_20261007'))
import run_repairs as r
import evaluate_pilot_v3 as e
import fixed_surface_repair as f


def main():
    root = r.ROOT / 'objectwise_repair_20261007'
    for folder in ('room2_sam_seed', 'room2_original_seed'):
        assert json.loads((root / folder / 'complete.json').read_text())['predictions_frozen_before_any_GT_evaluation']
        assert json.loads((root / folder / 'evaluation_summary.json').read_text())['status'] == 'PASS'
    gt = e.load_gt(r.INPUT / 'room2/ground_truth/gt.npz')
    baseline = f.load_arrays(r.BASE / 'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz')
    mapping = f.fixed_mapping(r.ROOT / 'strict_local_repair_20261007/room2_mask14_mask24/fixed_surface_mapping.npz',
                              baseline['xyz_m'], gt.xyz_ref, .01)
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    records = json.loads((r.BASE / 'native/room2/P1-A1/surface_p0/materialization_report.json').read_text())['frame_records']
    lookup = {}
    for line in (r.BASE / 'native/room2/P1-A1/association/associations.jsonl').read_text().splitlines():
        row = json.loads(line); lookup[(row['frame_id'], row['mask_local_id'])] = row['instance_id']
    cases = {}
    for folder in ('room2_sam_seed', 'room2_original_seed'):
        work = root / folder
        done = json.loads((work / 'complete.json').read_text())
        ev = json.loads((work / 'evaluation_summary.json').read_text())
        assert done['status'] == ev['status'] == 'PASS'
        for p, expected in done['repair']['output_sha256'].items():
            assert r.sha(p) == expected
        assert r.sha(ev['mapping_cache']) == ev['mapping_cache_sha256']
        final = f.load_arrays(work / 'final/instance_surface.npz')['instance_id']
        evidence = f.load_arrays(work / 'native/surface_evidence.npz')
        cohort = f.load_arrays(work / 'retired_surface_ids.npz')['surface_point_index']
        decisions = {row['frame']: row for row in json.loads((work / 'frame_decisions.json').read_text())['frames']}
        base = done['association']['instance_base']
        b_refs = valid & (gt.instance_id == 4002) & mapping['reachable']
        b_points = np.unique(mapping['nearest_surface_index'][b_refs])
        residual_points = b_points[final[b_points] == 52]
        per_frame, reasons = [], Counter()
        for row in records:
            fid = row['frame_id']; d = decisions[fid]
            tx = work / 'transactions' / ('f%06d.npz' % fid)
            if tx.exists():
                transaction = f.load_arrays(tx)
                before, after = transaction['old_frame_keys'], transaction['new_frame_keys']
            else:
                old = f.load_arrays(row['support_file'])
                points, local = old['surface_point_index'], old['mask_local_id']
                ids = np.array([lookup[(fid, int(mid))] for mid in local])
                before = np.unique(points.astype(np.int64) * base + ids)
                after = before
            b_old = before[np.isin(before // base, b_points) & (before % base == 52)]
            b_residual = after[np.isin(after // base, b_points) & (after % base == 52)]
            b_new = after[np.isin(after // base, b_points) & (after % base == 355)]
            residual_old = after[np.isin(after // base, residual_points) & (after % base == 52)]
            if len(b_old) or len(b_new) or len(b_residual):
                target = d.get('objects', {}).get('7', {})
                if 7 not in d.get('accepted_track_ids', []):
                    reasons.update(target.get('reasons', []))
                per_frame.append({'frame': fid, 'old_ID52_votes_before': len(b_old), 'ID52_votes_after': len(b_residual),
                                  'ID355_votes_after': len(b_new), 'ID52_votes_on_final_wrong_ID52_points': len(residual_old),
                                  'track7_accepted': 7 in d.get('accepted_track_ids', []), 'track7_reasons': target.get('reasons', [])})
        cases[folder] = {'retired_support_unique_native_points': len(cohort),
            'retired_support_raw_states_and_final_labels': [{'state': s, 'surface_points': int(np.sum(evidence['state'][cohort] == s)),
                'final_assigned': int(np.sum((evidence['state'][cohort] == s) & (final[cohort] > 0))),
                'final_unassigned': int(np.sum((evidence['state'][cohort] == s) & (final[cohort] <= 0)))} for s in range(4)],
            'target_B_unique_reachable_native_points': len(b_points), 'target_B_final_wrong_ID52_native_points': len(residual_points),
            'target_B_ID52_votes_before': sum(row['old_ID52_votes_before'] for row in per_frame),
            'target_B_ID52_votes_after': sum(row['ID52_votes_after'] for row in per_frame),
            'target_B_ID355_votes_after': sum(row['ID355_votes_after'] for row in per_frame),
            'rejected_target_B_reason_frame_counts_nonexclusive': dict(reasons), 'per_frame_remaining_votes': per_frame}
    report = {'status': 'PASS', 'posthoc_readonly_audit': True, 'GT_used_only_after_prediction_freeze': True,
              'no_new_repair_parameters_or_decisions': True, 'cases': cases}
    f.atomic_json(root / 'remaining_vote_audit.json', report)
    print(json.dumps({'status': 'PASS', 'cases': {name: {k: v for k, v in row.items() if k != 'per_frame_remaining_votes'} for name, row in cases.items()}}), flush=True)


if __name__ == '__main__':
    main()

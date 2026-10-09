"""Read-only diagnosis: reproduce frozen postprocessing, explain each newly gray point.

Writes diagnostic reports only. Does not modify masks, votes, maps or evaluator.
"""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
LEGACY = Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0, str(LEGACY))
import run_repairs as r
from revisable_instance_map.offline_surface_assignment import REJECTION_NAMES


def read(path):
    return json.loads(path.read_text())


def histogram(array):
    values, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def distribution(array):
    if not len(array):
        return None
    return {'min': float(np.min(array)), 'median': float(np.median(array)),
            'mean': float(np.mean(array)), 'p90': float(np.quantile(array, .9)), 'max': float(np.max(array))}


def main():
    freeze = read(ROOT / 'evaluation_freeze.json')
    complete = read(ROOT / 'validated/full_track/complete.json')
    protected = {row['path']: row['sha256'] for row in freeze['predictions'].values()}
    protected.update(complete['output_sha256'])
    for path, digest in protected.items():
        assert r.sha(path) == digest
    scene = r.Scene('room2')
    out = ROOT / 'root_cause_20261008'
    out.mkdir(exist_ok=True)
    folder = ROOT / 'validated/full_track'
    evidence = r.load_npz(folder / 'native/surface_evidence.npz')
    pair = r.load_npz(folder / 'native/surface_instance_frame_votes.npz')
    candidate = r.load_npz(folder / 'candidate/instance_surface.npz')['instance_id']
    final = r.load_npz(folder / 'final/instance_surface.npz')['instance_id']
    native = np.where(evidence['state'] == r.CONFIRMED, evidence['top1_instance_id'], -1).astype(np.int32)
    variants, detail, stats = r.assign_surface(scene.xyz, scene.normals, scene.rgb, evidence, pair, native, scene.settings)
    np.testing.assert_array_equal(variants['holes_geodesic'], candidate)
    assert complete['postprocessing_settings_sha256'] == scene.settings_sha
    reason = np.full(len(final), -1, np.int16)
    reason[detail['surface_point_index']] = detail['rejection_reason']
    lost = (scene.final > 0) & (final <= 0)
    gained = (scene.final <= 0) & (final > 0)
    changed = scene.final != final
    oldgray = scene.final <= 0
    finalgray = final <= 0
    scope = r.load_npz(folder / 'allowed_surface_ids.npz')['surface_point_index']
    retired = r.load_npz(folder / 'retired_surface_ids.npz')['surface_point_index']
    association = read(ROOT / 'global_association.json')
    objects = association['objects']
    repair_pids = np.array([row['persistent_id'] for row in objects])
    new_pids = repair_pids[repair_pids > scene.maximum_id]
    repaired_existing_pids = repair_pids[repair_pids <= scene.maximum_id]
    loss_reasons = {REJECTION_NAMES[int(key)]: count for key, count in histogram(reason[lost]).items()}
    groups = np.stack([scene.evidence['state'][lost], evidence['state'][lost]], axis=1)
    values, counts = np.unique(groups, axis=0, return_counts=True)
    transitions = [{'baseline_native_state': int(value[0]), 'repaired_native_state': int(value[1]), 'points': int(count)}
                   for value, count in zip(values, counts)]

    # Complete candidate ledger, including candidates beyond top two.
    p = pair['surface_point_index']
    pid = pair['instance_id']
    votes = pair['frame_votes']
    support_new = np.zeros(len(final), np.int32)
    support_other = np.zeros(len(final), np.int32)
    support_reused = np.zeros(len(final), np.int32)
    isnew = np.isin(pid, new_pids)
    isreused = np.isin(pid, repaired_existing_pids)
    np.add.at(support_new, p[isnew], votes[isnew])
    np.add.at(support_reused, p[isreused], votes[isreused])
    np.add.at(support_other, p[~(isnew | isreused)], votes[~(isnew | isreused)])
    support_groups = {}
    for a in [False, True]:
        for b in [False, True]:
            for c in [False, True]:
                take = lost & ((support_new > 0) == a) & ((support_reused > 0) == b) & ((support_other > 0) == c)
                if take.any():
                    support_groups[f'new_identity={a},reused_repair_identity={b},other_old_identity={c}'] = int(take.sum())
    pairs = np.stack([evidence['top1_instance_id'][lost], evidence['top2_instance_id'][lost]], axis=1)
    ids, counts = np.unique(pairs, axis=0, return_counts=True)
    top_conflicts = sorted([{'top1': int(value[0]), 'top2': int(value[1]), 'newly_gray_points': int(count)}
                            for value, count in zip(ids, counts)], key=lambda row: -row['newly_gray_points'])[:25]

    decisions = read(folder / 'frame_decisions.json')
    whole = np.zeros(len(final), bool)
    partial = np.zeros(len(final), bool)
    removal_events = np.zeros(len(final), np.int32)
    addition_events = np.zeros(len(final), np.int32)
    frame_by_id = {row['frame']: row for row in decisions['frames']}
    for transaction in sorted((folder / 'transactions').glob('*.npz')):
        z = r.load_npz(transaction)
        fid = int(transaction.stem[1:])
        plans = frame_by_id[fid]['old_observation_plans']
        region = z['retired_surface_local_pairs']
        for action, flags in [('whole', whole), ('partial', partial)]:
            mids = [int(mid) for mid, value in plans.items() if value == action]
            flags[region[np.isin(region[:, 1], mids), 0]] = True
        base = int(z['instance_base'][0])
        removed = np.setdiff1d(z['old_frame_keys'], z['new_frame_keys'])
        added = np.setdiff1d(z['new_frame_keys'], z['old_frame_keys'])
        np.add.at(removal_events, removed // base, 1)
        np.add.at(addition_events, added // base, 1)
    assert int(removal_events.sum()) == complete['removed_frame_surface_identity_votes']
    assert int(addition_events.sum()) == complete['added_frame_surface_identity_votes']
    before_total = scene.evidence['total_frame_votes']
    after_total = evidence['total_frame_votes']
    assert np.array_equal(before_total - removal_events + addition_events, after_total)
    top_loss_ids = sorted(histogram(scene.final[lost]).items(), key=lambda item: -item[1])
    loss_by_old_id = []
    for identity, count in top_loss_ids:
        take = lost & (scene.final == int(identity))
        loss_by_old_id.append({'baseline_id': int(identity), 'newly_gray_points': count,
                              'states_after': histogram(evidence['state'][take]),
                              'post_rejections': {REJECTION_NAMES[int(k)]: v for k, v in histogram(reason[take]).items()}})

    diagnostic = read(ROOT / 'seed_diagnostics.json')
    track_lookup = {obj['track_id']: {**obj, 'ROI': case['source_choice']['ROI'], 'seed_frame': case['source_choice']['frame'],
                                    'case_uid': case['case_uid']} for case in diagnostic['cases'] for obj in case['objects']}
    track_checks = defaultdict(lambda: {'accepted': 0, 'rejected': 0, 'reason_counts': Counter()})
    for frame in decisions['frames']:
        for track, check in frame['objects'].items():
            row = track_checks[int(track)]
            row['accepted' if check['accepted'] else 'rejected'] += 1
            row['reason_counts'].update(check['reasons'])
    track_reports = []
    for obj in objects:
        members = obj['member_track_ids']
        pid = obj['persistent_id']
        track_reports.append({**obj, 'final_points': int((final == pid).sum()),
                              'native_confirmed_points': int((native == pid).sum()),
                              'has_reliable_postprocessing_boundary_seed': pid in stats['original_seed_instance_ids'],
                              'loss_points_with_direct_votes_for_identity': int((lost & np.isin(np.arange(len(final)), p[pid == pair['instance_id']])).sum()),
                              'members': [{'track_id': oid, 'ROI': track_lookup[oid]['ROI'], 'mask_id': track_lookup[oid]['native_mask_id'],
                                           'seed_frame': track_lookup[oid]['seed_frame'], 'old_family_id': track_lookup[oid]['old_family_id'],
                                           'projected_seed_points': track_lookup[oid]['projected_seed_points'],
                                           'accepted_mapping_frames': track_checks[oid]['accepted'],
                                           'rejected_mapping_frames': track_checks[oid]['rejected'],
                                           'rejection_counts': dict(track_checks[oid]['reason_counts'])} for oid in members]})

    evaluation = read(ROOT / 'evaluation_summary.json')
    comparison = evaluation['comparisons']['full_track']
    per_gt = []
    for row in comparison['targets']:
        b, a = row['before'], row['after']
        per_gt.append({'GT_id': b['GT_id'], 'IoU_before': b['best_IoU'], 'IoU_after': a['best_IoU'],
                       'IoU_delta_pp': (a['best_IoU'] - b['best_IoU']) * 100,
                       'completeness_before': b['completeness_best_single_instance'], 'completeness_after': a['completeness_best_single_instance'],
                       'completeness_delta_pp': (a['completeness_best_single_instance'] - b['completeness_best_single_instance']) * 100,
                       'purity_before': b['purity_of_best_IoU_instance'], 'purity_after': a['purity_of_best_IoU_instance'],
                       'pred_before': b['best_pred_uid'], 'pred_after': a['best_pred_uid'],
                       'fragments_before': b['significant_predictions'], 'fragments_after': a['significant_predictions']})
    result = {
        'status': 'PASS', 'purpose': 'read-only root-cause diagnosis; GT only used for posthoc per-object evaluation',
        'source_hashes': protected, 'source_code_sha256': r.sha(__file__), 'prediction_vote_masks_unchanged': True,
        'postprocessing_rerun_equals_frozen_candidate_exactly': True,
        'postprocessing_recomputed_statistics': stats,
        'points': len(final), 'baseline_instances': int(len(np.unique(scene.final[scene.final > 0]))),
        'final_instances': int(len(np.unique(final[final > 0]))), 'baseline_unknown': int(oldgray.sum()),
        'final_unknown': int(finalgray.sum()), 'assigned_to_unknown': int(lost.sum()), 'unknown_to_assigned': int(gained.sum()),
        'newly_gray_native_states': histogram(evidence['state'][lost]), 'newly_gray_native_state_transitions': transitions,
        'newly_gray_postprocessing_rejections': loss_reasons,
        'newly_gray_support_identity_categories': support_groups, 'newly_gray_top1_top2_pairs': top_conflicts,
        'newly_gray_vote_distribution_before': distribution(before_total[lost]),
        'newly_gray_vote_distribution_after': distribution(after_total[lost]),
        'newly_gray_total_votes_not_decreased': int((lost & (after_total >= before_total)).sum()),
        'newly_gray_with_removed_votes': int((lost & (removal_events > 0)).sum()),
        'newly_gray_with_added_votes': int((lost & (addition_events > 0)).sum()),
        'newly_gray_no_added_votes': int((lost & (addition_events == 0)).sum()),
        'newly_gray_without_direct_vote_changes': int((lost & (removal_events == 0) & (addition_events == 0)).sum()),
        'newly_gray_whole_retirement_only': int((lost & whole & ~partial).sum()),
        'newly_gray_partial_retirement_only': int((lost & ~whole & partial).sum()),
        'newly_gray_whole_and_partial_retirement': int((lost & whole & partial).sum()),
        'newly_gray_no_retired_region': int((lost & ~whole & ~partial).sum()),
        'loss_by_baseline_id': loss_by_old_id, 'tracks': track_reports,
        'old_identity_max': scene.maximum_id, 'new_repair_ids': new_pids.tolist(), 'reused_repair_ids': repaired_existing_pids.tolist(),
        'target_GT_count': len(per_gt), 'per_GT_diagnostics': per_gt,
        'target_best_IoU_decreased_over_1pp': [row for row in per_gt if row['IoU_delta_pp'] < -1],
        'target_completeness_decreased_over_1pp': [row for row in per_gt if row['completeness_delta_pp'] < -1],
        'target_fragment_count_increased': [row for row in per_gt if row['fragments_after'] > row['fragments_before']],
        'multiple_selected_parts_same_GT': evaluation['multiple_selected_parts_of_same_GT_instance'],
        'boundary_constraint': {'outside_changes_blocked': complete['strict_commit']['candidate_outside_changes_blocked'],
                                'final_outside_changes': complete['strict_commit']['final_outside_changes']},
        'vote_event_sums_verified': True,
    }
    scene.verify_unchanged()
    for path, digest in protected.items():
        assert r.sha(path) == digest
    np.savez_compressed(out / 'newly_gray_diagnostics.npz', surface_point_index=np.flatnonzero(lost).astype(np.int32),
                        baseline_instance_id=scene.final[lost], baseline_state=scene.evidence['state'][lost],
                        repaired_state=evidence['state'][lost], top1_id=evidence['top1_instance_id'][lost],
                        top2_id=evidence['top2_instance_id'][lost], top1_votes=evidence['top1_votes'][lost],
                        top2_votes=evidence['top2_votes'][lost], total_votes=evidence['total_frame_votes'][lost],
                        rejection_reason=reason[lost], removed_votes=removal_events[lost], added_votes=addition_events[lost])
    (out / 'root_cause_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = {key: value for key, value in result.items() if key not in {'source_hashes', 'tracks', 'per_GT_diagnostics', 'postprocessing_recomputed_statistics'}}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Run one configurable offline case, then optionally evaluate its frozen result."""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'strict_local_repair_20261007'))
import run_repairs as r
import fixed_surface_repair as f
from repair_association import associate_seed
from joint_vote_ops import keys_from_pairs, batch_delta
import objectwise_ops as ops


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    allowed = {'schema_version', 'scene', 'case_uid', 'experiment_mode', 'seed_mode', 'seed_frame',
               'seed_manifest', 'objects', 'old_family_ids', 'policy_file', 'output_dir', 'tracking'}
    if set(cfg) != allowed or cfg['schema_version'] != 1:
        raise ValueError('Case config has missing/unknown fields; GT belongs in a separate evaluation config')
    if cfg['experiment_mode'] != 'offline_retrospective' or cfg['tracking']['directions'] != ['forward', 'reverse']:
        raise ValueError('This experiment requires independent offline forward AND reverse tracking')
    if cfg['seed_mode'] not in ops.SEED_MODES:
        raise ValueError('Use an explicit seed mode')
    for key in ('output_dir', 'seed_manifest'):
        if not Path(cfg[key]).is_absolute() or '/CVPR/' not in cfg[key]:
            raise ValueError(key + ' must be an absolute CVPR path')
    if not cfg['old_family_ids'] or any(int(pid) <= 0 for pid in cfg['old_family_ids']):
        raise ValueError('Specify the manually identified original family IDs')
    policy_path = (path.parent / cfg['policy_file']).resolve()
    policy = json.loads(policy_path.read_text())
    if policy['GT_used_for_repair']:
        raise ValueError('GT cannot participate in repair')
    return path, cfg, policy_path, policy


def check_hashes(hashes):
    for path, expected in hashes.items():
        if f.file_sha256(path) != expected:
            raise RuntimeError('Protected input changed: ' + path)


def progress(out, stage, **extra):
    row = {'status': 'RUNNING', 'stage': stage, **extra}
    f.atomic_json(out / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def prepare(scene, cfg, out, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    seed = next(row for row in manifest['seeds'] if row['case_uid'] == cfg['case_uid'])
    if seed['status'] != 'READY' or seed['scene'] != cfg['scene'] or seed['source_choice']['frame'] != cfg['seed_frame']:
        raise ValueError('Human seed/case/frame mismatch')
    source = manifest_path.parent / seed['seed_file']
    assert r.sha(source) == seed['seed_sha256']
    candidate = np.array(Image.open(source), np.uint16)
    fid = cfg['seed_frame']
    frame = scene.source.load_frame(fid)
    selected = ops.seed_labels(frame.mask_local, candidate, cfg['objects'], cfg['seed_mode'])
    seed_path = out / 'prepared_seed.png'
    if seed_path.exists():
        np.testing.assert_array_equal(np.array(Image.open(seed_path)), selected)
    else:
        Image.fromarray(selected).save(seed_path)
    ids = [int(obj['track_id']) for obj in cfg['objects']]
    # Protect human-selected neighbouring seed objects, but do not repair them.
    context = np.where(~np.isin(candidate, ids), candidate, 0).astype(np.uint16)
    pts, local, stats = r.project_frame_regions(replace(frame, mask_local=selected), scene.tree,
                                               pixel_stride=2, max_distance_m=.015, workers=8)
    context_pts, context_local, _ = r.project_frame_regions(replace(frame, mask_local=context), scene.tree,
                                                           pixel_stride=2, max_distance_m=.015, workers=8)
    provisional_base = scene.maximum_id + max(ids) + 1
    _, table, _, visible = scene.old_frame(fid, provisional_base)
    rows = associate_seed(pts, local, visible, scene.final, ids, scene.maximum_id)
    objects = ops.freeze_association(rows, cfg['objects'], cfg['seed_mode'],
                                    {mid: int(table[mid]) for mid in np.unique(frame.mask_local) if mid})
    base = max([scene.maximum_id, *[row['persistent_id'] for row in objects]]) + 1
    report = {'status': 'PASS', 'scene': cfg['scene'], 'case_uid': cfg['case_uid'], 'seed_frame': fid,
              'seed_mode': cfg['seed_mode'], 'objects': objects, 'instance_base': base,
              'seed_file': str(seed_path), 'seed_sha256': r.sha(seed_path),
              'human_candidate_path': str(source), 'human_candidate_sha256': r.sha(source),
              'human_manifest_sha256': r.sha(manifest_path), 'projection': stats,
              'selected_original_masks_pixel_identical': cfg['seed_mode'].startswith('original_'),
              'GT_used': False, 'association_threshold_dice': .5}
    f.atomic_json(out / 'frozen_association.json', report)
    f.atomic_npz(out / 'seed_projected_support.npz', surface_point_index=pts, track_id=local,
                 protected_surface_point_index=context_pts, protected_track_id=context_local)
    return report


def ensure_tracking(config_path, cfg, association, out):
    if cfg['seed_mode'] == 'original_control':
        return None
    tracking_cfg = cfg['tracking']
    reuse = 'reuse_dir' in tracking_cfg
    folder = Path(tracking_cfg['reuse_dir']) if reuse else out / 'tracking'
    if not (folder / 'complete.json').exists():
        if reuse:
            raise RuntimeError('Configured reusable tracking does not exist')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = tracking_cfg['cuda_visible_devices']
        subprocess.run([tracking_cfg['python'], str(HERE / 'track_case.py'), '--config', str(config_path)],
                       env=env, check=True)
    report = json.loads((folder / 'complete.json').read_text())
    ids = [row['track_id'] for row in association['objects']]
    assert report['status'] == 'PASS' and report['case_uid'] == cfg['case_uid']
    assert report['track_ids'] == ids and report['seed_frame'] == cfg['seed_frame']
    assert report['directions_use_independent_seed_only_states'] and not report['GT_used']
    assert report['checkpoint_sha256'] == tracking_cfg['checkpoint_sha256'] == r.sha(tracking_cfg['checkpoint'])
    if reuse:
        assert report['code_sha256'] == r.sha(tracking_cfg['source_code'])
    else:
        assert report['code_sha256'] == r.sha(HERE / 'track_case.py')
    raw_count = len(scene_frame_ids(cfg['scene']))
    frames = {row['frame']: row for row in report['frames']}
    assert sorted(frames) == list(range(raw_count))
    assert report['total_frames'] == report['completed_frames'] == raw_count
    selected = np.array(Image.open(association['seed_file']))
    np.testing.assert_array_equal(np.array(Image.open(frames[cfg['seed_frame']]['label_file'])), selected)
    for fid, row in frames.items():
        assert r.sha(row['label_file']) == row['label_sha256']
        if fid != cfg['seed_frame']:
            assert row['direction'] == ('reverse' if fid < cfg['seed_frame'] else 'forward')
    proof = {'status': 'PASS', 'reuse_existing_tracking': reuse, 'tracking_report': str(folder / 'complete.json'),
             'tracking_sha256': r.sha(folder / 'complete.json'), 'all_raw_frame_checksums_verified': raw_count,
             'seed_pixels_identical': True, 'offline_forward_and_reverse': True,
             'independent_seed_only_states': True, 'original_inference_seconds': report['elapsed_seconds'],
             'inference_seconds_this_run': 0 if reuse else report['elapsed_seconds']}
    f.atomic_json(out / 'tracking_validation.json', proof)
    return report


def scene_frame_ids(scene):
    raw = json.loads((r.INPUT / scene / 'configs/raw.json').read_text())
    source = raw['source']
    poses = np.loadtxt(Path(source['scene_root']) / source['trajectory']).reshape(-1, 4, 4)
    return range(len(poses))


def visible_seed(scene, frame, points, policy):
    pc = (scene.xyz[points] - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
    pc = pc[(pc[:, 2] > .05) & (pc[:, 2] < 10)]
    if not len(pc):
        return np.empty(0, int), np.empty(0, int)
    u = np.rint(frame.camera.fx * pc[:, 0] / pc[:, 2] + frame.camera.cx).astype(int)
    v = np.rint(frame.camera.fy * pc[:, 1] / pc[:, 2] + frame.camera.cy).astype(int)
    h, w = frame.mask_local.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, pc = u[inside], v[inside], pc[inside]
    d = frame.depth_m[v, u]
    seen = (d > 0) & np.isfinite(d) & (np.abs(d - pc[:, 2]) <= policy['seed_surface_visibility_depth_tolerance_m'])
    return u[seen], v[seen]


def bbox_inside(xyz, bounds):
    return np.any(np.stack([np.all((xyz >= lo) & (xyz <= hi), axis=1) for lo, hi in bounds]), axis=0) if len(xyz) else np.ones(0, bool)


def aggregate(arrays):
    keys, votes = np.unique(np.concatenate(arrays), return_counts=True)
    return keys, votes.astype(np.int64)


def repair(scene, cfg, policy, association, tracking, out):
    started = time.monotonic()
    fixed = {row['track_id']: row['persistent_id'] for row in association['objects']}
    base = association['instance_base']
    support = f.load_arrays(out / 'seed_projected_support.npz')
    samples, bounds = {}, {}
    for oid in fixed:
        points = np.unique(support['surface_point_index'][support['track_id'] == oid])
        if not len(points):
            raise ValueError('Seed has no visible projected surface')
        samples[oid] = points[np.linspace(0, len(points) - 1, min(len(points), 2048), dtype=int)]
        xyz = scene.xyz[points]
        bounds[oid] = (xyz.min(0) - policy['seed_surface_bbox_padding_m'], xyz.max(0) + policy['seed_surface_bbox_padding_m'])
    for oid in np.unique(support['protected_track_id']):
        points = np.unique(support['protected_surface_point_index'][support['protected_track_id'] == oid])
        samples[int(oid)] = points[np.linspace(0, len(points) - 1, min(len(points), 2048), dtype=int)]
    keys = scene.pair['surface_point_index'].astype(np.int64) * base + scene.pair['instance_id']
    order = np.argsort(keys, kind='stable')
    original_keys, original_votes = keys[order], scene.pair['frame_votes'][order].astype(np.int64)
    old_frames, new_frames, removed, added, scopes, retired_points = [], [], [], [], [], []
    decisions, observations = [], []
    tracked = {row['frame']: row for row in tracking['frames']} if tracking else {}
    counts = {'related_mapping_frames': 0, 'updated_mapping_frames': 0, 'single_object_frames': 0,
              'all_objects_frames': 0, 'object_accepted_frames': {str(oid): 0 for oid in fixed},
              'whole_observations': 0, 'partial_observations': 0}
    for index, fid in enumerate(sorted(scene.records)):
        frame, table, old_keys, _ = scene.old_frame(fid, base)
        cached = r.load_npz(Path(scene.records[fid]['support_file']))
        op, ol = cached['surface_point_index'], cached['mask_local_id']
        old_frames.append(old_keys)
        family_local = [int(mid) for mid in np.unique(ol) if int(table[mid]) in cfg['old_family_ids']]
        counts['related_mapping_frames'] += bool(family_local)
        if cfg['seed_mode'] == 'original_control':
            new_frames.append(old_keys)
            decisions.append({'frame': fid, 'accepted_track_ids': [], 'action': 'original_control_noop'})
            continue
        row = tracked[fid]
        assert r.sha(row['label_file']) == row['label_sha256']
        labels = np.array(Image.open(row['label_file']), np.uint16)
        assert labels.shape == frame.mask_local.shape and set(np.unique(labels)).issubset({0, *fixed})
        np_, nt, _ = r.project_frame_regions(replace(frame, mask_local=labels), scene.tree,
                                              pixel_stride=2, max_distance_m=.015, workers=8)
        visible = {oid: visible_seed(scene, frame, pts, policy) for oid, pts in samples.items()}
        checks, accepted = {}, []
        for oid in fixed:
            mask = labels == oid
            u, v = visible[oid]
            points = np_[nt == oid]
            foreign = mask & (frame.mask_local > 0) & ~np.isin(frame.mask_local, family_local)
            metrics = {'visible_seed_samples': len(u), 'tracked_coverage': float(np.mean(labels[v, u] == oid)) if len(u) else 0.,
                       'new_surface_bbox_fraction': float(np.mean(bbox_inside(scene.xyz[points], [bounds[oid]]))) if len(points) else 0.,
                       'foreign_old_mask_pixel_fraction': float(np.sum(foreign) / max(1, np.sum(mask))),
                       'other_seed_overlaps': [{'track_id': other, 'visible_seed_samples': len(uv[0]),
                           'coverage': float(np.mean(mask[uv[1], uv[0]])) if len(uv[0]) else 0.}
                           for other, uv in visible.items() if other != oid]}
            accept, reasons = ops.object_reliability(metrics, policy)
            if not family_local:
                accept = False
                reasons.append('no_identified_old_family_observation')
            checks[str(oid)] = {**metrics, 'accepted': accept, 'reasons': reasons}
            if accept:
                accepted.append(oid)
                counts['object_accepted_frames'][str(oid)] += 1
        if not accepted:
            new_frames.append(old_keys)
            decisions.append({'frame': fid, 'accepted_track_ids': [], 'objects': checks, 'old_local_masks': family_local})
            continue
        counts['updated_mapping_frames'] += 1
        counts['single_object_frames'] += len(accepted) == 1
        counts['all_objects_frames'] += len(accepted) == len(fixed)
        union = np.isin(labels, accepted)
        protected_ok = all(len(uv[0]) < policy['minimum_visible_protected_seed_samples'] or
                           float(np.mean(union[uv[1], uv[0]])) <= policy['maximum_new_union_coverage_of_other_visible_seed_samples']
                           for oid, uv in visible.items() if oid not in fixed)
        plans, old_checks = {}, {}
        for mid in family_local:
            spatial = float(np.mean(bbox_inside(scene.xyz[op[ol == mid]], list(bounds.values()))))
            plans[mid] = ops.observation_plan(accepted, fixed, spatial, protected_ok, policy)
            old_checks[str(mid)] = {'old_surface_bbox_fraction': spatial, 'action': plans[mid]}
        partial_ids = [mid for mid, action in plans.items() if action == 'partial']
        residual_mask = np.where(np.isin(frame.mask_local, partial_ids), frame.mask_local, 0).astype(frame.mask_local.dtype)
        residual_mask[union] = 0
        pp, pl, _ = r.project_frame_regions(replace(frame, mask_local=residual_mask), scene.tree,
                                           pixel_stride=2, max_distance_m=.015, workers=8)
        accepted_new = np.isin(nt, accepted)
        np_, nt = np_[accepted_new], nt[accepted_new]
        new_ids = np.array([fixed[int(oid)] for oid in nt], np.int32)
        revised, retained, retired, scope = ops.replace_observations(op, ol, table, plans, pp, pl, np_, new_ids, base)
        # Every wholly unselected old observation remains at every original support.
        unselected = ~np.isin(ol, [mid for mid, action in plans.items() if action != 'retain'])
        assert np.all(np.isin(keys_from_pairs(op[unselected], table[ol[unselected]], base), retained))
        rem, add = np.setdiff1d(old_keys, revised), np.setdiff1d(revised, old_keys)
        assert np.all(np.isin(np.setxor1d(old_keys, revised) // base, scope))
        new_frames.append(revised); removed.append(rem); added.append(add); scopes.append(scope)
        retired_points.append(retired[:, 0].astype(np.int32))
        transaction = out / 'transactions' / ('f%06d.npz' % fid)
        f.atomic_npz(transaction, old_frame_keys=old_keys, new_frame_keys=revised, retained_old_frame_keys=retained,
                     retired_surface_local_pairs=retired, accepted_track_ids=np.array(accepted, np.int32),
                     allowed_surface_point_index=scope, instance_base=np.array([base], np.int64))
        for mid, action in plans.items():
            removed_pairs = retired[retired[:, 1] == mid, 0]
            observations.append({'frame': fid, 'old_local_mask': mid, 'old_persistent_id': int(table[mid]),
                                 'action': action, 'old_surface_region_pairs': int(np.sum(ol == mid)),
                                 'retired_surface_region_pairs': len(removed_pairs),
                                 'unknown_residual_region_pairs': int(np.sum(pl == mid))})
            counts[action + '_observations'] += 1
        decisions.append({'frame': fid, 'accepted_track_ids': accepted, 'objects': checks,
                          'old_local_masks': family_local, 'old_observations': old_checks,
                          'removed_frame_votes': len(rem), 'added_frame_votes': len(add),
                          'transaction_sha256': r.sha(transaction), 'allowed_points': len(scope)})
        if (index + 1) % 40 == 0:
            progress(out, 'per_object_observation_replacement', completed_mapping_frames=index + 1, **counts)
    bk, bv = aggregate(old_frames)
    np.testing.assert_array_equal(bk, original_keys); np.testing.assert_array_equal(bv, original_votes)
    del old_frames, bk, bv
    rem = np.concatenate(removed) if removed else np.empty(0, np.int64)
    add = np.concatenate(added) if added else np.empty(0, np.int64)
    k, v = batch_delta(original_keys, original_votes, rem, add)
    nk, nv = aggregate(new_frames)
    np.testing.assert_array_equal(k, nk); np.testing.assert_array_equal(v, nv)
    back_k, back_v = batch_delta(k, v, add, rem)
    np.testing.assert_array_equal(back_k, original_keys); np.testing.assert_array_equal(back_v, original_votes)
    del new_frames, nk, nv, back_k, back_v
    scope = np.unique(np.concatenate(scopes)).astype(np.int32) if scopes else np.empty(0, np.int32)
    evidence = r.reduce_surface_votes(k, v, len(scene.xyz), base,
                                      scene.report['min_confirmed_votes'], scene.report['min_confirmed_ratio'])
    native = np.where(evidence['state'] == r.CONFIRMED, evidence['top1_instance_id'], -1).astype(np.int32)
    pair = {'surface_point_index': (k // base).astype(np.int32), 'instance_id': (k % base).astype(np.int32),
            'frame_votes': v.astype(np.int32)}
    progress(out, 'postprocess_and_strict_commit', **counts)
    if np.array_equal(k, original_keys) and np.array_equal(v, original_votes):
        candidate = scene.final.copy()
        stats = {'identical_baseline': True}
    else:
        variants, _, stats = r.assign_surface(scene.xyz, scene.normals, scene.rgb, evidence, pair, native, scene.settings)
        candidate = variants['holes_geodesic']
    baseline = {'xyz_m': scene.xyz, 'rgb': scene.rgb, 'instance_id': scene.final}
    proposed = {**baseline, 'instance_id': candidate}
    labels, commit, blocked = f.bounded_final_labels(baseline, proposed, scope)
    raw_evidence = {'xyz_m': scene.xyz, 'rgb': scene.rgb, **evidence}
    proof = f.verify_raw_outside(scene.evidence, raw_evidence, scene.pair, pair, scope)
    f.atomic_npz(out / 'native/surface_evidence.npz', **raw_evidence)
    f.atomic_npz(out / 'native/surface_instance_frame_votes.npz', **pair)
    f.atomic_npz(out / 'native/instance_surface.npz', xyz_m=scene.xyz, rgb=scene.rgb, instance_id=native)
    f.atomic_npz(out / 'candidate/instance_surface.npz', **proposed)
    f.atomic_npz(out / 'final/instance_surface.npz', xyz_m=scene.xyz, rgb=scene.rgb, instance_id=labels)
    f.atomic_npz(out / 'allowed_surface_ids.npz', surface_point_index=scope)
    f.atomic_npz(out / 'blocked_outside_changes.npz', surface_point_index=blocked,
                 unrestricted_label=candidate[blocked], restored_baseline_label=scene.final[blocked])
    f.atomic_npz(out / 'retired_surface_ids.npz', surface_point_index=np.unique(np.concatenate(retired_points)).astype(np.int32) if retired_points else np.empty(0, np.int32))
    f.atomic_json(out / 'frame_decisions.json', {'status': 'PASS', 'counts': counts, 'frames': decisions, 'observations': observations})
    old_final_unknown = int(np.sum(scene.final[scope] <= 0))
    final_unknown = int(np.sum(labels[scope] <= 0))
    if cfg['seed_mode'] == 'original_control':
        assert not len(scope) and np.array_equal(k, original_keys) and np.array_equal(v, original_votes)
        np.testing.assert_array_equal(labels, scene.final)
    scene.verify_unchanged()
    outputs = {str(path): r.sha(path) for path in [out / 'final/instance_surface.npz', out / 'candidate/instance_surface.npz',
        out / 'native/surface_evidence.npz', out / 'native/surface_instance_frame_votes.npz', out / 'allowed_surface_ids.npz',
        out / 'blocked_outside_changes.npz', out / 'frame_decisions.json', out / 'retired_surface_ids.npz']}
    report = {'status': 'PASS', 'scene': cfg['scene'], 'case_uid': cfg['case_uid'], 'seed_mode': cfg['seed_mode'],
              'experiment_mode': cfg['experiment_mode'], 'GT_used_for_repair': False,
              'complete_frame_replay_bit_identical_to_delta_ledger': True, 'exact_rollback_pass': True,
              'unretired_old_observation_supports_preserved_exactly': True,
              'scope_definition': 'retired old observation-region support union accepted new support; no geometric expansion',
              'strict_commit': {**commit, **proof}, 'counts': counts,
              'removed_frame_surface_identity_votes': len(rem), 'added_frame_surface_identity_votes': len(add),
              'scope_unknown_before': old_final_unknown, 'scope_unknown_after': final_unknown,
              'assigned_to_unassigned_inside': int(np.sum((scene.final[scope] > 0) & (labels[scope] <= 0))),
              'unassigned_to_assigned_inside': int(np.sum((scene.final[scope] <= 0) & (labels[scope] > 0))),
              'postprocessing_settings_sha256': scene.settings_sha,
              'confirmation_thresholds': {'minimum_votes': scene.report['min_confirmed_votes'],
                                          'minimum_ratio': scene.report['min_confirmed_ratio']},
              'postprocessing_statistics': stats, 'output_sha256': outputs,
              'original_inputs_unchanged': True, 'seconds': round(time.monotonic() - started, 2)}
    f.atomic_json(out / 'repair_summary.json', report)
    return report


def run(config_path):
    config_path, cfg, policy_path, policy = load_config(config_path)
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    progress(out, 'source_preflight', seed_mode=cfg['seed_mode'])
    scene = r.Scene(cfg['scene'])
    scene.baseline_control()
    manifest_path = Path(cfg['seed_manifest'])
    source_seed = next(row for row in json.loads(manifest_path.read_text())['seeds'] if row['case_uid'] == cfg['case_uid'])
    protected = [config_path, policy_path, manifest_path, manifest_path.parent / source_seed['seed_file'],
                 HERE / 'run_case.py', HERE / 'objectwise_ops.py', HERE / 'track_case.py',
                 HERE.parent / 'run_repairs.py', HERE.parent / 'repair_association.py',
                 HERE.parent / 'joint_vote_ops.py', HERE.parent / 'evidence_replacement.py',
                 HERE.parent / 'strict_local_repair_20261007/fixed_surface_repair.py']
    if 'reuse_dir' in cfg['tracking']:
        protected.extend([Path(cfg['tracking']['reuse_dir']) / 'complete.json', Path(cfg['tracking']['source_code'])])
    hashes = {**scene.source_hashes, **{str(path): r.sha(path) for path in protected}}
    freeze = {'status': 'FROZEN_BEFORE_REPAIR', 'case_config': cfg, 'protected_source_hashes': hashes,
              'GT_used_for_repair': False, 'explicit_changes': ['seed association mode', 'per-object acceptance',
              'whole or reliable-foreground partial observation replacement', 'configurable case entry point'],
              'unchanged': ['independent forward/reverse SAM2 tracking', 'numeric quality thresholds',
                            'TSDF geometry/RGB', 'vote deduplication and weights', 'confirmation rules',
                            'original assignment settings', 'strict commit helper', 'fixed diagnostic helper', 'unified v3']}
    if (out / 'source_freeze.json').exists():
        assert json.loads((out / 'source_freeze.json').read_text()) == freeze, 'Frozen source changed; use a new output directory'
    else:
        f.atomic_json(out / 'source_freeze.json', freeze)
    if (out / 'complete.json').exists():
        done = json.loads((out / 'complete.json').read_text())
        check_hashes(hashes); check_hashes(done['repair']['output_sha256'])
        f.atomic_json(out / 'status.json', {'status': 'PASS', 'stage': 'existing_complete_preserved'})
        return done
    association = prepare(scene, cfg, out, manifest_path)
    tracking = ensure_tracking(config_path, cfg, association, out)
    summary = repair(scene, cfg, policy, association, tracking, out)
    check_hashes(hashes); check_hashes(summary['output_sha256'])
    complete = {'status': 'PASS', 'config_sha256': r.sha(config_path), 'repair': summary,
                'association': association, 'tracking_validation': str(out / 'tracking_validation.json') if tracking else None,
                'predictions_frozen_before_any_GT_evaluation': True, 'GT_used_for_repair': False}
    f.atomic_json(out / 'complete.json', complete)
    f.atomic_json(out / 'status.json', {'status': 'PASS', 'stage': 'complete', 'counts': summary['counts'],
                                     'outside_final_label_changes': summary['strict_commit']['final_outside_changes']})
    print(json.dumps({'status': 'PASS', 'seed_mode': cfg['seed_mode'], 'counts': summary['counts'],
                      'strict_commit': summary['strict_commit'], 'seconds': summary['seconds']}, ensure_ascii=False), flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--evaluation-config')
    args = parser.parse_args()
    try:
        run(args.config)
        if args.evaluation_config:
            # Separate process: prediction/source freeze is complete before GT is opened.
            subprocess.run([sys.executable, str(HERE / 'evaluate_case.py'), '--config', args.config,
                            '--evaluation-config', args.evaluation_config], check=True)
    except Exception:
        cfg = json.loads(Path(args.config).read_text())
        out = Path(cfg['output_dir'])
        f.atomic_json(out / 'failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        f.atomic_json(out / 'status.json', {'status': 'FAIL', 'stage': 'failure'})
        raise

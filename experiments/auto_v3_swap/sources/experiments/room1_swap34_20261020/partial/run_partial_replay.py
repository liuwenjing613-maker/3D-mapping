"""Replay auto-v2 changing only frozen whole plans to existing partial plans.

All tracking, final gate decisions, identities, trusted regions, geometry and
postprocessing are reused from the scene-specific frozen auto-v2 input. No GT is loaded.
"""
from bootstrap import C, R, SOURCE, SOURCE_CODE, SCENE, COUNT, verify_sources
from pathlib import Path
import gc
import hashlib
import json
import time
import traceback
import numpy as np
from PIL import Image
import run_history_core as h
from partial_policy import partial_decision
from instance_repair.cli import load_script

obs = load_script('_local_frozen_observation_ops', SOURCE_CODE / 'observation_ops.py')
STAGES = ('direct', 'core_all', 'whole_to_partial')
PARENT = dict(zip(STAGES, ('baseline', 'direct', 'core_all')))
FINAL = ('whole_to_partial',)
strict_frame = obs.strict_frame
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    h.f.atomic_json(p, value)


def emit(stage, **fields):
    row = {'stage': stage, 'unix': time.time(), **fields}
    dump(R / 'progress.json', row)
    print(json.dumps(row), flush=True)


def cat(chunks):
    return np.concatenate(chunks) if chunks else np.empty(0, np.int64)


def aggregate(chunks):
    return np.unique(cat(chunks), return_counts=True)


def adopted_mask(frame, decision, obj, info, canonical):
    accepted = list(map(int, decision['accepted_track_ids']))
    if not accepted:
        merged = np.zeros(frame.mask_local.shape, np.uint16)
    else:
        initial = set(accepted) | {int(oid) for oid, item in decision['objects'].items()
                                  if item.get('rechecked_after_overlap_abstention')}
        labels = {}
        for uid in sorted({obj[oid] for oid in initial}):
            row = info[uid]['frames'][frame.frame_id]
            h.protect(row['label_file'], row['label_sha256'])
            labels[uid] = np.asarray(Image.open(row['label_file'])).copy()
        _, conflict = h.merge_foregrounds(
            [(oid, labels[obj[oid]] == oid) for oid in sorted(initial)], canonical)
        assert int(conflict.sum()) == decision['cross_case_conflict_pixels']
        for label in labels.values():
            label[conflict] = 0
        merged, remaining = h.merge_foregrounds(
            [(oid, labels[obj[oid]] == oid) for oid in accepted], canonical)
        assert not remaining.any()
    path = SOURCE / 'validated/full_track/accepted_masks' / f'f{frame.frame_id:06d}.png'
    h.protect(path)
    np.testing.assert_array_equal(merged, np.asarray(Image.open(path)))
    return merged


def main():
    started = time.time()
    assert not (R / 'predictions_freeze.json').exists()
    tests = json.loads((R / 'tests_complete.json').read_text())
    assert tests['status'] == 'PASS'
    codehash = {str(p): sha(p) for p in C.glob('*.py')}
    policy = {
        'status': 'FROZEN_BEFORE_MAPPING_AND_NEW_SCORING', 'source': str(SOURCE),
        'scene': SCENE, 'selected_masks': COUNT, 'GT_read_by_mapper': False,
        'only_algorithm_change': "old_observation_plans values whole -> partial",
        'whole_retirement': False,
        'direct': 'Original partial retirement: retain old_family_ids eligibility; remove only eligible adopted-mask overlap',
        'history': 'Unchanged E1 core and expanded-region revision of retained original observations',
        'accepted_masks': 'Unchanged; old observations outside old_family_ids remain separate candidates and may abstain',
        'no_observation': 'No synthetic historical fallback vote',
        'geometry': 'Original source projection, prior-core rule and expanded-region 2cm visibility',
        'fixed': ['seeds', 'SAM2 tracking', 'gate', 'adopted masks', 'old_family_ids', 'identity association',
                  'core and expanded regions', 'TSDF', 'confirmation thresholds', 'postprocessing', 'stage commit rules'],
        'control': 'Original E1 source candidates, frame votes, aggregate votes, evidence and stage scopes reproduced',
    }
    dump(R / 'policy.json', policy)
    h.protect(C / 'adapter_provenance.json')
    selection = h.read(SOURCE / 'confirmed_selection.json')
    selection_path = (Path('/data/chenkejun/CVPR/results/auto_v2_replay_20261019_r2/seed_manifest.json')
                      if SCENE == 'room2' else SOURCE.parent / 'v2/seed_manifest.json')
    selected = h.read(selection_path)
    groups = {g['id']: g for g in selected['groups']}
    assert len(groups) == len(selection['included']) == COUNT
    assert set(groups) == {x['group_id'] for x in selection['included']}
    for item in selection['included']:
        group = groups[item['group_id']]
        primary = group['primary']
        assert item['mask_id'] == primary['id']
        assert item['frame'] == primary['frame']
        assert item['mask_sha256'] == primary['mask_sha256']
        assert item['original_track_id'] == group.get('original_track_id', int(group['id'][1:]))
    dump(R / 'input_selection_audit.json', {'status': 'PASS', 'scene': SCENE,
         'count': COUNT, 'auto_v2_manifest': str(selection_path),
         'auto_v2_manifest_sha256': sha(selection_path),
         'original_E1_selection_sha256': sha(SOURCE / 'confirmed_selection.json'),
         'all_groups_frames_masks_and_original_track_ids_match': True,
         'user_retained_exceptions': selection.get('user_retained_exceptions', [])})
    for name in ('input_freeze.json', 'pipeline_code_freeze.json', 'e1/region_freeze.json'):
        h.protect(SOURCE / name)
    for p, digest in h.read(SOURCE / 'pipeline_code_freeze.json')['files'].items():
        h.protect(p, digest)
    sourcefreeze = h.read(SOURCE / 'e1/region_freeze.json')
    for p, digest in sourcefreeze['code_sha256'].items():
        h.protect(p, digest)
    for p, digest in sourcefreeze['source_sha256'].items():
        h.protect(p, digest)
    h.protect(SOURCE / 'e1/region.npz', sourcefreeze['region_sha256'])
    h.protect(SOURCE / 'e1/policy.json', sourcefreeze['policy_sha256'])
    h.protect(SOURCE_CODE / 'run_history_core.py')
    h.protect(SOURCE_CODE / 'observation_ops.py')
    cases, assoc, obj, info, canonical, fixed = h.catalog(SOURCE)
    assert len(obj) == COUNT
    base = assoc['instance_base']
    scene = h.r.Scene(SCENE)
    assert sorted(scene.records) == list(range(0, 2000, 5))
    decisions = {d['frame']: d for d in h.read(SOURCE / 'validated/full_track/frame_decisions.json')['frames']}
    region = h.load(SOURCE / 'e1/region.npz')
    prior, owner = region['prior_owner'], region['owner']
    claims = h.load(SOURCE / 'modified_repair' / SCENE / 'core_claims.npz')
    prior_expected = np.full(len(scene.xyz), -1, np.int32)
    prior_expected[claims['surface_point_index']] = claims['claimed_persistent_id']
    np.testing.assert_array_equal(prior, prior_expected)
    apoints = np.flatnonzero((prior != -1) | (owner != -1))
    baseline_records = h.read(SOURCE / 'original_repair/predictions_freeze.json')['scenes'][SCENE]['predictions']
    baseline_dir = Path(baseline_records['strict_baseline_final']['path']).parent.parent
    baseline_pair = h.load(baseline_dir / 'surface_instance_frame_votes.npz')
    baseline_evidence = h.load(baseline_dir / 'surface_evidence.npz')
    baseline = h.load(baseline_records['strict_baseline_final']['path'], baseline_records['strict_baseline_final']['sha256'])
    np.testing.assert_array_equal(scene.xyz, baseline['xyz_m'])
    np.testing.assert_array_equal(scene.rgb, baseline['rgb'])
    bk, bv = h.encode(baseline_pair, base)
    frames = {name: [] for name in ('baseline', *STAGES)}
    removed = {name: [] for name in STAGES}
    added = {name: [] for name in STAGES}
    scopes = {name: [] for name in STAGES}
    receipts = []
    can_to_pid = np.zeros(max(fixed) + 1, np.int64)
    for cid, pid in fixed.items():
        can_to_pid[cid] = pid
    old_frames = {name: [] for name in STAGES}
    old_scopes = {name: [] for name in STAGES}
    old_paths = {'direct': SOURCE / 'original_repair' / SCENE / 'full_track',
                 'core_all': SOURCE / 'modified_repair' / SCENE,
                 'whole_to_partial': SOURCE / 'e1/E1_expanded_history' / SCENE}
    for index, fid in enumerate(sorted(scene.records)):
        frame = scene.source.load_frame(fid)
        rec = scene.records[fid]
        support = h.load(rec['support_file'], rec['support_sha256'])
        maskpath = scene.source.mask_root / scene.source.config['source']['mask_pattern'].format(frame=fid)
        h.protect(maskpath, str(support['source_mask_sha256'][0]))
        table = np.zeros(int(frame.mask_local.max()) + 1, np.int64)
        for mid in np.unique(frame.mask_local):
            if mid:
                table[mid] = scene.lookup[(fid, int(mid))]
        p, mid, u, v = h.sourcepixels(frame, scene)
        original_local = h.keys(p, mid, len(table))
        np.testing.assert_array_equal(original_local,
            h.keys(support['surface_point_index'], support['mask_local_id'], len(table)))
        old = table[mid]
        merged = adopted_mask(frame, decisions[fid], obj, info, canonical)
        maskp, maskcid = h.project(frame, merged, scene)
        expected_new = h.keys(maskp, can_to_pid[maskcid], base)
        vis = np.zeros(len(scene.xyz), bool)
        vis[apoints], _, _ = obs.visible_samples(scene.xyz[apoints], frame.camera_to_world,
                                                frame.camera, frame.depth_m, .02)
        baseline_raw = h.keys(p, old, base)
        baseline_eff = strict_frame(baseline_raw, base)
        frames['baseline'].append(baseline_eff)
        revised = partial_decision(decisions[fid])
        assert revised.get('old_observation_plans', {}) == {
            k: 'partial' for k in decisions[fid].get('old_observation_plans', {})}
        tx = {'baseline_candidate_keys': baseline_raw, 'baseline_frame_keys': baseline_eff}
        frame_stats = {}
        for variant, decision in [('old', decisions[fid]), ('partial', revised)]:
            # Execute the frozen original reconstruction with only the decision value changed.
            residual, rp, rl, np_, nc = h.reconstruct_current(
                frame, table, scene, decision, obj, info, canonical, fixed)
            newkeys = h.keys(np_, can_to_pid[nc], base)
            np.testing.assert_array_equal(newkeys, expected_new)
            available = residual[v, u] == mid
            retained = h.keys(p[available], old[available], base)
            np.testing.assert_array_equal(retained, h.keys(rp, table[rl], base))
            if variant == 'old':
                old_available = available.copy()
            else:
                assert np.all(available[old_available])
                # Whenever there were no whole plans, all source observations must be identical.
                if decision.get('old_observation_plans', {}) == decisions[fid].get('old_observation_plans', {}):
                    np.testing.assert_array_equal(available, old_available)
            ids_core = old.copy()
            choose = available & (prior[p] > 0) & (old != prior[p])
            ids_core[choose] = prior[p[choose]]
            ids_expanded, changed = obs.revise_all_history(p, old, available, prior, owner, vis)
            stage_ids = dict(zip(STAGES, (old, ids_core, ids_expanded)))
            raw, eff = {'baseline': baseline_raw}, {'baseline': baseline_eff}
            saved_raw, saved_eff = baseline_raw, baseline_eff
            for name in STAGES:
                ids = stage_ids[name]
                histkeys = h.keys(p[available], ids[available], base)
                raw[name] = np.union1d(histkeys, newkeys)
                # Separate history and adopted-mask projections preserve conflicting identities.
                image = table[residual].astype(np.uint16)
                image[v[available], u[available]] = ids[available]
                hp, hi = h.project(frame, image, scene)
                np.testing.assert_array_equal(h.keys(hp, hi, base), histkeys)
                eff[name] = strict_frame(raw[name], base)
                assert len(eff[name]) == len(np.unique(eff[name] // base))
                parent = PARENT[name]
                rem = np.setdiff1d(eff[parent], eff[name])
                add = np.setdiff1d(eff[name], eff[parent])
                scope = np.unique(np.r_[rem // base, add // base]).astype(np.int32)
                if name == 'direct':
                    retired_local = np.setdiff1d(original_local, h.keys(rp, rl, len(table)))
                    direct_scope = np.unique(np.r_[retired_local // len(table), np_]).astype(np.int32)
                    assert np.all(np.isin(scope, direct_scope))
                    scope = direct_scope
                    if variant == 'partial':
                        assert np.all(np.isin(scope, maskp))
                elif name == 'core_all':
                    assert np.all(prior[scope] > 0)
                else:
                    assert np.all((owner[scope] != -1) | (prior[scope] != -1))
                tx[variant + '_' + name + '_candidate_keys'] = raw[name]
                tx[variant + '_' + name + '_frame_keys'] = eff[name]
                if variant == 'old':
                    path = old_paths[name] / 'transactions' / f'f{fid:06d}.npz'
                    if path.exists():
                        saved = h.load(path)
                        saved_raw, saved_eff = saved['new_candidate_keys'], saved['new_frame_keys']
                    np.testing.assert_array_equal(raw[name], saved_raw, err_msg=name + ' frozen source candidates')
                    np.testing.assert_array_equal(eff[name], saved_eff, err_msg=name + ' frozen frame votes')
                    old_frames[name].append(eff[name])
                    old_scopes[name].append(scope)
                else:
                    removed[name].append(rem)
                    added[name].append(add)
                    scopes[name].append(scope)
                    frames[name].append(eff[name])
            frame_stats[variant] = {'retained_source_pixels': int(available.sum()),
                'prior_core_relabelled_source_pixels': int(choose.sum()),
                'expanded_region_relabelled_source_pixels': int(changed.sum())}
        h.f.atomic_npz(R / 'transactions' / f'f{fid:06d}.npz', **tx)
        receipts.append({'frame': fid, **frame_stats,
            'whole_plans_converted': sum(x == 'whole' for x in decisions[fid].get('old_observation_plans', {}).values())})
        if (index + 1) % 20 == 0:
            emit('replay', frames=index + 1, total_frames=400)
    # Verify the original complete pipeline before accepting the new predictions.
    previous_pair = (bk, bv)
    for name in STAGES:
        ok, ov = aggregate(old_frames.pop(name))
        stored_pair = h.load(old_paths[name] / 'native/surface_instance_frame_votes.npz')
        ek, ev = h.encode(stored_pair, base)
        np.testing.assert_array_equal(ok, ek)
        np.testing.assert_array_equal(ov, ev)
        oe = h.r.reduce_surface_votes(ok, ov, len(scene.xyz), base,
                                      scene.report['min_confirmed_votes'], scene.report['min_confirmed_ratio'])
        stored_evidence = h.load(old_paths[name] / 'native/surface_evidence.npz')
        for field in h.r.FIELDS:
            np.testing.assert_array_equal(oe[field], stored_evidence[field])
        oscope = np.unique(cat(old_scopes.pop(name))).astype(np.int32)
        if name in FINAL:
            before_k, before_v = previous_pair
            common, oi, bi = np.intersect1d(ok, before_k, return_indices=True)
            oscope = np.unique(np.r_[np.setxor1d(ok, before_k) // base,
                                    common[ov[oi] != before_v[bi]] // base]).astype(np.int32)
        frozen_scope = h.load(old_paths[name] / 'allowed_surface_ids.npz')['surface_point_index']
        np.testing.assert_array_equal(oscope, frozen_scope)
        previous_pair = (ok, ov)
        emit('original_stage_reproduced', variant=name)
    pk, pv = aggregate(frames.pop('baseline'))
    np.testing.assert_array_equal(pk, bk)
    np.testing.assert_array_equal(pv, bv)
    reduced = h.r.reduce_surface_votes(bk, bv, len(scene.xyz), base,
                                       scene.report['min_confirmed_votes'], scene.report['min_confirmed_ratio'])
    for field in h.r.FIELDS:
        np.testing.assert_array_equal(reduced[field], baseline_evidence[field])
    results = {'baseline': (bk, bv, baseline_evidence, baseline['instance_id'])}
    inventory = baseline.get('native_instance_ids', np.unique(baseline['instance_id'][baseline['instance_id'] > 0]))
    reports = {}
    for name in STAGES:
        emit('reduce', variant=name)
        pk, pv, pe, pf = results[PARENT[name]]
        rem, add = cat(removed.pop(name)), cat(added.pop(name))
        nk, nv = h.batch_delta(pk, pv, rem, add)
        backk, backv = h.batch_delta(nk, nv, add, rem)
        np.testing.assert_array_equal(backk, pk)
        np.testing.assert_array_equal(backv, pv)
        fullk, fullv = aggregate(frames.pop(name))
        np.testing.assert_array_equal(nk, fullk)
        np.testing.assert_array_equal(nv, fullv)
        scope = np.unique(cat(scopes.pop(name))).astype(np.int32)
        if name in FINAL:
            common, ni, pi = np.intersect1d(nk, pk, return_indices=True)
            scope = np.unique(np.r_[np.setxor1d(nk, pk) // base,
                                    common[nv[ni] != pv[pi]] // base]).astype(np.int32)
        monitor = np.zeros(len(scene.xyz), bool)
        monitor[scope] = True
        np.testing.assert_array_equal(nk[~monitor[nk // base]], pk[~monitor[pk // base]])
        np.testing.assert_array_equal(nv[~monitor[nk // base]], pv[~monitor[pk // base]])
        e = h.r.reduce_surface_votes(nk, nv, len(scene.xyz), base,
                                     scene.report['min_confirmed_votes'], scene.report['min_confirmed_ratio'])
        for field in h.r.FIELDS:
            np.testing.assert_array_equal(e[field][~monitor], pe[field][~monitor])
        native = np.where(e['state'] == h.r.CONFIRMED, e['top1_instance_id'], -1).astype(np.int32)
        pair = h.pairtable(nk, nv, base)
        variants, _, post = h.r.assign_surface(scene.xyz, scene.normals, scene.rgb, e, pair,
            native, scene.settings, progress=lambda s, v: emit('postprocess', variant=name, substage=s, values=v))
        final = pf.copy()
        final[scope] = variants['holes_geodesic'][scope]
        np.testing.assert_array_equal(final[~monitor], pf[~monitor])
        folder = R / name / SCENE
        for stage, labels in (('native', native), ('final', final)):
            # Retain the source E1 inventory, including IDs which disappear.
            reference_stage = (f'original_repair/{SCENE}/full_track' if name == 'direct'
                               else f'modified_repair/{SCENE}' if name.startswith('core_')
                               else f'e1/E1_expanded_history/{SCENE}')
            oldmap = h.load(SOURCE / reference_stage / stage / 'instance_surface.npz')
            inv = np.union1d(oldmap.get('native_instance_ids', inventory), np.unique(labels[labels > 0])).astype(np.int32)
            h.f.atomic_npz(folder / stage / 'instance_surface.npz', xyz_m=scene.xyz,
                           rgb=scene.rgb, instance_id=labels, native_instance_ids=inv)
        h.f.atomic_npz(folder / 'native/surface_evidence.npz', xyz_m=scene.xyz, rgb=scene.rgb, **e)
        h.f.atomic_npz(folder / 'native/surface_instance_frame_votes.npz', **pair)
        h.f.atomic_npz(folder / 'allowed_surface_ids.npz', surface_point_index=scope)
        reports[name] = {'status': 'PASS', 'parent': PARENT[name], 'scope_points': len(scope),
                         'removed_votes': len(rem), 'added_votes': len(add),
                         'final_changed_vs_parent': int((final != pf).sum()),
                         'summary': h.summary(e, final), 'postprocessing': post}
        dump(folder / 'complete.json', reports[name])
        results[name] = (nk, nv, e, final)
        del variants, fullk, fullv, backk, backv, rem, add
        gc.collect()
    dump(R / 'frame_receipts.json', receipts)
    scene.verify_unchanged()
    for p, digest in h.protected.items():
        assert sha(p) == digest, p
    for p, digest in codehash.items():
        assert sha(p) == digest, p
    verify_sources()
    outputs = {str(p): sha(p) for name in FINAL for p in (R / name).rglob('*.npz')}
    dump(R / 'predictions_freeze.json', {
        'status': 'PREDICTIONS_FROZEN_BEFORE_GT', 'GT_read_by_mapper': False,
        'source_sha256': h.protected, 'code_sha256': codehash, 'output_sha256': outputs,
        'policy_sha256': sha(R / 'policy.json'), 'baseline_modified': False,
        'checks': {'original_projection_matches_frozen_support_all_400_frames': True,
                   'adopted_masks_match_frozen_gate_results': True,
                   'original_E1_all_stages_candidates_votes_evidence_and_scopes_exact': True,
                   'only_whole_plan_values_changed': True,
                   'no_E1_baseline_full_replay_exact': True,
                   'all_three_stages_fresh_projection_matches_source_oracle': True,
                   'all_three_stages_full_replay_equals_delta': True,
                   'all_three_stages_exact_rollback': True,
                   'strict_one_vote_per_frame_point': True,
                   'outside_stage_scope_unchanged': True},
        'variants': FINAL, 'seconds': time.time() - started})
    dump(R / 'mapping_complete.json', {'status': 'PASS', 'seconds': time.time() - started,
                                      'variants': FINAL, 'stages': reports})
    emit('mapping_complete', seconds=time.time() - started)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(R / 'mapping_failure.json', {'traceback': traceback.format_exc()})
        raise

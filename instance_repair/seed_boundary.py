"""Whole-mask boundary exclusion, before noise filtering and seed grouping.

No geometry, labels, mask pixels, semantic routing or numerical repair rules
are changed here. Coordinates use half-open source boxes in the full image.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image

POLICY = 'full-frame-and-crop-edge-20261101'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_evidence(mask, aliases):
    a = np.asarray(mask)
    require(a.ndim == 2 and a.shape == (680, 1200), 'Expected native 1200x680 binary mask')
    require(np.isin(a, [0, 1, 255]).all(), 'Expected binary mask, not multi-instance labels')
    a = a != 0
    require(a.any(), 'Empty mask is not an eligible seed')
    require(bool(aliases), 'Source aliases are required; do not bypass crop checks')
    h, w = a.shape
    edges = dict(left=int(a[:, 0].sum()), top=int(a[0].sum()),
                 right=int(a[:, -1].sum()), bottom=int(a[-1].sum()))
    contacts = []
    for alias in aliases:
        require('model' in alias and 'source_box' in alias, 'Incomplete source provenance')
        model = alias['model']
        require(type(model) is int and model in (0, 1, 2), 'Unknown source model')
        box = alias['source_box']
        require(len(box) == 4 and all(type(x) is int for x in box), 'Invalid source box')
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, 'Source box outside image')
        require(model != 0 or box == [0, 0, w, h], 'Whole-frame source has a crop box')
        local = {}
        if model != 0:
            if x0 > 0: local['left'] = int(a[y0:y1, x0].sum())
            if y0 > 0: local['top'] = int(a[y0, x0:x1].sum())
            if x1 < w: local['right'] = int(a[y0:y1, x1 - 1].sum())
            if y1 < h: local['bottom'] = int(a[y1 - 1, x0:x1].sum())
        local = {k: v for k, v in local.items() if v}
        if local:
            contacts.append(dict(model=model, source_box=box, edge_pixels=local,
                                 crop_key=alias.get('crop_key')))
    full = any(edges.values())
    return dict(policy=POLICY, frame_edge_pixels=edges, touches_full_frame=full,
                crop_contacts=contacts,
                artificial_crop_edges=sorted({e for c in contacts for e in c['edge_pixels']}),
                touches_any_edge=full or bool(contacts))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def verify_mask(row, root):
    path = Path(root) / row['mask']
    require(sha(path) == row['mask_sha256'], 'Mask source hash changed: ' + str(path))
    a = np.asarray(Image.open(path))
    require(a.ndim == 2 and np.isin(a, [0, 1, 255]).all(), 'Mask must be binary')
    a = a != 0
    require(int(a.sum()) == row['area'], 'Mask pixel count changed')
    digest = hashlib.sha256(int(row['frame']).to_bytes(4, 'little') + np.packbits(a.ravel()).tobytes()).hexdigest()
    require(digest == row['digest'], 'Mask identity changed')
    return a


def screen(grouping, supports, pairs, rules_dir, out):
    """Rebuild v1 from source-eligible grouping_r2 candidates, never final seeds.

    Pass the branch's original v1 rules directory. The source routing has
    already happened upstream. The new v1 must go through semantic/v2
    measurement again before tracking; old group-ID selections cannot be reused.
    """
    grouping, supports, pairs, rules_dir, out = map(Path, (grouping, supports, pairs, rules_dir, out))
    require(not out.exists(), 'Use a new output directory; never overwrite a run')
    source = read(grouping)
    require(source.get('seeds') is not None, 'Expected pre-boundary candidate grouping')
    require(not any(s.get('touches_any_edge') is not None for s in source['seeds']),
            'Expected original candidates, not already screened final seeds')
    noise = load_module('_boundary_noise', rules_dir / 'noise_filter.py')
    group = load_module('_boundary_groups', rules_dir / 'grouping_rules.py')
    source_policy = load_module('_boundary_source', rules_dir / 'source_policy.py')
    inputs = [grouping, supports, pairs] + [rules_dir / n for n in ('noise_filter.py', 'grouping_rules.py', 'source_policy.py')]
    locked = {str(p.resolve()): sha(p) for p in inputs}
    with np.load(supports, allow_pickle=False) as z:
        indices, offsets = z['indices'], z['offsets']
    with np.load(pairs, allow_pickle=False) as z:
        intersections = z['intersection']
    require(intersections.shape == (len(source['seeds']),) * 2, 'Pair matrix shape mismatch')
    active, selected_supports, old_indices, excluded, noise_excluded, audit = [], [], [], [], [], []
    for expected_index, original in enumerate(source['seeds']):
        require(original['index'] == expected_index, 'Candidate/support index mismatch')
        s = copy.deepcopy(original)
        a = verify_mask(s, grouping.parent)
        path = (grouping.parent / s['mask']).resolve()
        locked[str(path)] = s['mask_sha256']
        evidence = boundary_evidence(a, s['aliases'])
        audit.append(dict(id=s['id'], frame=s['frame'], digest=s['digest'], mask_sha256=s['mask_sha256'], **evidence))
        s.update(evidence)
        s['original_index'] = original['index']
        s['mask'] = str(path)
        if evidence['touches_any_edge']:
            excluded.append(s)
            continue
        features = noise.shape_features(a)
        decision = noise.classify(features)
        s.update(noise_features=features, noise_decision=decision)
        if decision['exclude']:
            noise_excluded.append(s)
            continue
        ix = indices[offsets[expected_index]:offsets[expected_index + 1]]
        require(len(ix) == s['support_count'] and hashlib.sha256(ix.tobytes()).hexdigest() == s['support_sha256'], 'Support identity changed')
        s['index'] = len(active)
        active.append(s)
        selected_supports.append(ix)
        old_indices.append(expected_index)
    counts = np.array([len(x) for x in selected_supports], dtype=np.int64)
    require(np.all(counts[:-1] >= counts[1:]), 'Preserve original max-support ranking')
    inter = intersections[np.ix_(old_indices, old_indices)].copy()
    union = counts[:, None] + counts[None, :] - inter
    require(np.array_equal(np.diag(inter), counts), 'Pair diagonal does not match support counts')
    passes = (union > 0) & (4 * inter > union)
    grouped = group.group_seeds(range(len(active)), lambda i, j: bool(passes[i, j]))
    source_policy.assert_sam_only_groups(grouped, active)
    groups, manifests = [], []
    for gi, members in enumerate(grouped):
        primary = members[0]
        gid = f'A{gi + 1:03d}'
        views = {}
        for j in members:
            active[j]['group'] = gi
            if j != primary:
                require(len(np.intersect1d(selected_supports[primary], selected_supports[j], assume_unique=True)) == inter[primary, j], 'Primary/member support mismatch')
                if active[j]['frame'] != active[primary]['frame']:
                    views.setdefault(active[j]['frame'], []).append(j)
        validations = [dict(frame=f, members=js, representative=js[0]) for f, js in sorted(views.items())]
        groups.append(dict(id=gid, index=gi, primary=primary, members=members,
                           frames=sorted({active[j]['frame'] for j in members}), validation_frames=validations,
                           semantic_window_present=any(a.get('semantic_hit') for j in members for a in active[j]['aliases'])))
        manifests.append(dict(id=gid, primary=active[primary], auxiliary_ids=[active[j]['id'] for j in members[1:]], validation_frames=validations))
    for path, expected in locked.items():
        require(sha(path) == expected, 'Input changed during boundary screening')
    stats = dict(input_masks=len(source['seeds']), total_edge_removed=len(excluded),
                 full_frame_removed=sum(s['touches_full_frame'] for s in excluded),
                 crop_removed=sum(bool(s['artificial_crop_edges']) for s in excluded),
                 noise_removed=len(noise_excluded), after_noise=len(active), groups=len(groups))
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    rule = dict(boundary_policy=POLICY, boundary_stage='before_noise_and_grouping',
                exclude_entire_mask=True, check_all_aliases=True, mask_pixels_unchanged=True,
                downstream_selection_required=True, GT_used=False)
    save('grouping.json', dict(rule=rule, stats=stats, seeds=active, groups=groups,
                             source_sha256=locked, visibility_rule=source.get('visibility_rule')))
    save('seed_manifest.json', dict(status='boundary_screened_needs_semantic_and_v2_selection',
                                   rule=rule, groups=manifests, tracking_requested=False, map_write=False))
    np.savez_compressed(out / 'surface_supports.npz', indices=np.concatenate(selected_supports) if selected_supports else indices[:0], offsets=np.r_[0, np.cumsum(counts)])
    np.savez_compressed(out / 'pair_metrics.npz', intersection=inter, union=union, passes=passes,
                        iou=np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0))
    save('boundary_audit.json', dict(rule=rule, stats=stats, masks=audit, edge_excluded=excluded, noise_excluded=noise_excluded))
    save('audit.json', dict(status='PASS', policy=POLICY, stats=stats, source_sha256=locked,
                           output_sha256={n: sha(out/n) for n in ('grouping.json','seed_manifest.json','surface_supports.npz','pair_metrics.npz')},
                           all_active_and_auxiliary_masks_border_free=True, mask_pixels_unchanged=True))
    return stats


def validate_selected(manifest, grouping, mask_root):
    """Fail closed at the tracking handoff, checking primary and auxiliary masks."""
    selected, candidates = read(manifest), read(grouping)
    require(selected.get('status') in ('complete', 'automatic_v2_complete'),
            'Semantic and v2 selection must finish before tracking')
    require(candidates.get('rule', {}).get('boundary_policy') == POLICY, 'Missing current boundary stage')
    by_id = {s['id']: s for s in candidates['seeds']}
    checked = set()
    for g in selected['groups']:
        primary = g['primary']
        require(primary['id'] in by_id and primary['digest'] == by_id[primary['id']]['digest'], 'Selected primary is not a screened candidate')
        for key in ('frame', 'aliases', 'mask_sha256', 'area', 'support_sha256'):
            require(primary[key] == by_id[primary['id']][key], 'Selected primary provenance changed: ' + key)
        for ident in [primary['id'], *g['auxiliary_ids']]:
            require(ident in by_id, 'Selected auxiliary is not a screened candidate')
            s = by_id[ident]
            require(not boundary_evidence(verify_mask(s, mask_root), s['aliases'])['touches_any_edge'], 'Selected mask touches boundary: ' + ident)
            checked.add(ident)
        require(primary['mask_sha256'] == by_id[primary['id']]['mask_sha256'], 'Selected primary hash changed')
        require(sha(Path(mask_root)/primary['mask']) == primary['mask_sha256'], 'Selected primary file differs')
    return dict(status='PASS', policy=POLICY, selected_groups=len(selected['groups']), checked_masks=len(checked))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('screen')
    for name in ('grouping','supports','pairs','rules-dir','out'):
        q.add_argument('--'+name, required=True, type=Path)
    q = sub.add_parser('validate-selected')
    for name in ('manifest','grouping','mask-root'):
        q.add_argument('--'+name, required=True, type=Path)
    sub.add_parser('current-policy')
    a = p.parse_args(argv)
    if a.command == 'screen':
        result = screen(a.grouping,a.supports,a.pairs,a.rules_dir,a.out)
    elif a.command == 'validate-selected':
        result = validate_selected(a.manifest,a.grouping,a.mask_root)
    else:
        registry = Path(__file__).resolve().parent/'configs/current_auto_pipeline.json'
        result = read(registry)
        require(result['seed_boundary_policy'] == POLICY, 'Default registry does not require boundary gate')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

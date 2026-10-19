"""Recompute all 990 full-support pairs on the server and freeze decisions."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import sys
import time
import numpy as np
from PIL import Image
from rules import compare, decide


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    started = time.perf_counter()
    code = Path(__file__).resolve().parent
    cfg = json.loads((code / 'config.json').read_text())
    inp, out, original = map(Path, (cfg['input_directory'], cfg['output_directory'], cfg['original_directory']))
    assert cfg['threshold'] == .7 and not cfg['common_visibility_restriction']
    assert not (out / 'classification.json').exists(), 'Refuse to overwrite frozen decisions.'
    locked = {}

    def lock(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, ('Input changed', str(path))
        locked[str(path)] = digest

    lock(inp / 'selection_binary.json', cfg['input_selection_sha256'])
    lock(inp / 'seed_manifest.json', cfg['input_manifest_sha256'])
    lock(inp / 'measurement_verification.json', cfg['input_measurement_verification_sha256'])
    lock(cfg['pair_reference'], cfg['pair_reference_sha256'])
    source_lock = json.loads((inp / 'measurement_verification.json').read_text())['source_sha256']
    support_path = original / 'surface_supports.npz'
    lock(support_path, source_lock[str(support_path)])
    selection = json.loads((inp / 'selection_binary.json').read_text())
    manifest = json.loads((inp / 'seed_manifest.json').read_text())
    groups = {g['id']: g for g in manifest['groups']}
    selected = {r['group_id']: r for r in selection['retained']}
    assert set(groups) == set(selected) and len(groups) == 45
    assert not (set(groups) & {r['group_id'] for r in selection['skipped']})
    with np.load(support_path) as z:
        indices, offsets = z['indices'], z['offsets']
    supports, visibility = {}, {}
    for gid, group in groups.items():
        seed = group['primary']
        assert group['original_track_id'] == int(gid[1:])
        for key, value in [('mask_id', seed['id']), ('frame', seed['frame']), ('mask_sha256', seed['mask_sha256'])]:
            assert selected[gid][key] == value
        mask_path = original / seed['mask']
        lock(mask_path, seed['mask_sha256'])
        assert (inp / seed['mask']).resolve() == mask_path.resolve()
        support = indices[offsets[seed['index']]:offsets[seed['index'] + 1]]
        assert len(support) == seed['support_count'] == selected[gid]['support_count']
        assert hashlib.sha256(support.tobytes()).hexdigest() == seed['support_sha256']
        frame = seed['frame']
        if frame not in visibility:
            path = Path(cfg['visibility_directory']) / f'frame{frame:06d}_visible.npz'
            lock(path, source_lock[str(path)])
            with np.load(path) as z:
                visibility[frame] = z['vertex_index'], z['uv']
        visible, uv = visibility[frame]
        mask = np.asarray(Image.open(mask_path)) > 0
        assert np.array_equal(visible[mask[uv[:, 1], uv[:, 0]]], support)
        supports[gid] = support
    reference = json.loads(Path(cfg['pair_reference']).read_text())
    old_pairs = {tuple(sorted((p['a'], p['b']))): p for p in reference['pairs']}
    pairs = []
    for a, b in itertools.combinations(sorted(groups), 2):
        p = compare(a, supports[a], b, supports[b])
        old = old_pairs[a, b]
        assert p['intersection'] == old['original_intersection']
        assert p['original_iou'] == old['original_iou']
        assert {old['a']: old['count_a'], old['b']: old['count_b']} == {a: p['count_a'], b: p['count_b']}
        p.update(frame_a=groups[a]['primary']['frame'], frame_b=groups[b]['primary']['frame'])
        pairs.append(p)
    excluded, retained = decide(groups, pairs)
    # Actual data have neither ties nor chained exclusions; do not invent a retained ancestor.
    chains = [p['pair_id'] for p in pairs if p['passes'] and p['large'] in excluded]
    assert len(pairs) == 990
    assert not chains, 'Chained exclusions need explicit review before publishing this trial.'
    records = []
    for gid in sorted(groups):
        seed = groups[gid]['primary']
        related = [p for p in pairs if gid in (p['a'], p['b']) and p['intersection'] > 0]
        related.sort(key=lambda p: (-int(p['passes']), -p['containment'], p['pair_id']))
        causes = [p for p in related if p['passes'] and p['small'] == gid]
        larger = [p for p in related if p['small'] == gid and not p['equal_size']]
        records.append(dict(group_id=gid, original_track_id=groups[gid]['original_track_id'],
            mask_id=seed['id'], frame=seed['frame'], area=seed['area'], support_count=seed['support_count'],
            mask_sha256=seed['mask_sha256'], support_sha256=seed['support_sha256'],
            decision='exclude' if gid in excluded else 'retain',
            containing_group_ids=[p['large'] for p in causes],
            reason_pair_ids=[p['pair_id'] for p in causes], related_pair_ids=[p['pair_id'] for p in related],
            max_containment_by_larger=max((p['containment'] for p in larger), default=0)))
    stats = dict(input=len(groups), excluded=len(excluded), retained=len(retained), pairs=len(pairs),
                 overlapping_pairs=sum(p['intersection'] > 0 for p in pairs),
                 qualifying_pairs=sum(p['passes'] for p in pairs), source_frames=len(visibility))
    for path, digest in locked.items():
        assert sha(path) == digest
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'pair_results.json', dict(config=cfg, stats=stats, pairs=pairs))
    save(out / 'classification.json', dict(schema='full-support-containment-trial-v1', config=cfg, stats=stats,
        records=records, pair_results_sha256=sha(out / 'pair_results.json')))
    save(out / 'selection_binary.json', dict(schema='full-support-containment-selection-v1',
        counts=dict(excluded=len(excluded), retained=len(retained)),
        input_selection_sha256=cfg['input_selection_sha256'], classification_sha256=sha(out / 'classification.json'),
        excluded=[r for r in records if r['decision'] == 'exclude'],
        retained=[r for r in records if r['decision'] == 'retain'], tracking_requested=False, map_write=False))
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(groups=[groups[g] for g in retained], containment_config=cfg,
        containment_excluded_group_ids=excluded, containment_source_manifest_sha256=cfg['input_manifest_sha256'],
        status='full_support_containment_trial_only', tracking_requested=False, map_write=False)
    save(out / 'seed_manifest.json', new_manifest)
    (out / 'masks').symlink_to((inp / 'masks').resolve(), target_is_directory=True)
    for gid in retained:
        assert sha(out / groups[gid]['primary']['mask']) == groups[gid]['primary']['mask_sha256']
    save(out / 'verification.json', dict(status='PASS', source_sha256=locked,
        code_sha256={p.name: sha(p) for p in code.iterdir() if p.suffix in ('.py', '.json')},
        checks=['Exact 45 retained B inputs, original IDs and mask hashes preserved',
                'All supports reconstructed from unchanged masks and locked source-visible fixed TSDF vertices',
                'All 990 full-set intersections independently recomputed and match archived original intersections',
                'Strict >70% integer boundary; no common-visible denominator or added point gate',
                'Every exclusion has a direct >70% containment witness with more points',
                'No equal-size passing pairs, chained exclusions, mask changes, E1 run or GT use'],
        elapsed_seconds=time.perf_counter() - started,
        output_sha256={name: sha(out / name) for name in
                      ['pair_results.json', 'classification.json', 'selection_binary.json', 'seed_manifest.json']}))
    print(json.dumps(stats))
    print('Excluded:', ','.join(excluded))
    if '--measure-only' not in sys.argv:
        from build_page import build
        build()


if __name__ == '__main__':
    main()

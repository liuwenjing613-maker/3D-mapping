"""Replay all accepted filters and export unchanged original seed groups."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from pipeline import identity, require, run_filters

CODE = Path(__file__).resolve().parent
ACCEPTED_RULES = {
    'semantic': dict(excluded_classes=['wall', 'floor', 'ceiling', 'rug'], pixel_confidence_gte=.7, mask_fraction_gte=.9),
    'covisible_dedup': dict(iou_gte=.8, minimum_individual_visibility=None, minimum_visible_points=None, grouping='fixed_primary_direct_edges'),
    'skip_repair_B': dict(p_gte=.95, r_gte=.95, source_frame_only=True, conflict_gate=False, minimum_support=None),
    'containment': dict(full_support_fraction_gt=.7, exclude='smaller_TSDF_support', equal_size_or_chain='require_user_review'),
    'pr_filter': dict(p_gt=.1, r_lt=.2, operator='AND'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def validate_reference(result, reference, automatic_only=False):
    for stage in result['stages']:
        expected = reference['stages'][stage['stage']]
        for key in ('removed_group_ids', 'retained_group_ids'):
            require(stage[key] == expected[key], 'Historical stage reproduction failed: ' + stage['stage'])
    expected_final = (reference['stages']['pr_filter']['retained_group_ids'] if automatic_only
                      else reference['final_retained_group_ids'])
    require(result['retained_group_ids'] == expected_final, 'Final selection reproduction failed')


def replay(out, original_manifest=None, mask_root=None, verify_server_sources=False, automatic_only=False):
    config = load(CODE / 'config.json')
    require(config['rules'] == ACCEPTED_RULES, 'Unsupported changed rules; do not reuse stale measurements')
    inputs = CODE / config['inputs']
    source_lock = load(inputs / 'source_lock.json')
    for name, digest in source_lock['local_files_sha256'].items():
        require(sha(inputs / name) == digest, 'Prepared input changed: ' + name)
    measured = load(inputs / 'measurements.json')
    require(measured['case_id'] == source_lock['case_id'] == config['case_id'], 'Case changed')
    require(measured['baseline_sha256'] == source_lock['baseline']['sha256'], 'Baseline changed')
    require(measured['semantic_pixel_confidence'] == .7 and
            set(measured['semantic_exclude_classes']) == {'wall', 'floor', 'ceiling', 'rug'},
            'Unsupported semantic pixel measurements')
    if verify_server_sources:
        for name, item in source_lock['sources'].items():
            require(sha(item['server_path']) == item['sha256'], 'Server source changed: ' + name)
        for key in ('baseline', 'evidence'):
            item = source_lock[key]
            require(sha(item['path']) == item['sha256'], 'Server fixed surface changed: ' + key)
    original_manifest = Path(original_manifest) if original_manifest else CODE / config['original_manifest']
    require(sha(original_manifest) == source_lock['original_manifest']['sha256'], 'Original manifest changed')
    original = load(original_manifest)
    groups = {group['id']: group for group in original['groups']}
    require(len(groups) == len(original['groups']), 'Duplicate original seed group')
    require(set(groups) == {r['group_id'] for r in measured['seeds']}, 'Original group scope changed')
    mask_hashes = {}
    for row in measured['seeds']:
        group = groups[row['group_id']]
        primary = group['primary']
        actual = dict(group_id=group['id'], original_track_id=int(group['id'][1:]),
                      mask_id=primary['id'], frame=primary['frame'], mask_sha256=primary['mask_sha256'])
        require(identity(row) == actual, 'Original identity changed: ' + group['id'])
        require(row['support_count'] == primary['support_count'] and
                row['support_sha256'] == primary['support_sha256'], 'Original support changed')
        if mask_root:
            path = Path(mask_root) / primary['mask']
            digest = sha(path)
            require(digest == primary['mask_sha256'], 'Original mask changed: ' + group['id'])
            mask_hashes[group['id']] = digest
    overrides = None if automatic_only else load(CODE / config['manual_override'])
    result = run_filters(measured, overrides)
    validate_reference(result, load(inputs / 'reference_decisions.json'), automatic_only)
    out = Path(out)
    require(not out.exists(), 'Refuse to overwrite an existing output directory')
    out.mkdir(parents=True)
    save(out / 'selection.json', result)

    def manifest_for(ids, kind):
        manifest = copy.deepcopy(original)
        manifest['groups'] = []
        for gid in ids:
            group = copy.deepcopy(groups[gid])
            group['original_track_id'] = int(gid[1:])
            manifest['groups'].append(group)
        manifest.update(status=kind, original_track_ids_preserved=True,
            source_manifest_sha256=source_lock['original_manifest']['sha256'],
            selection_sha256=sha(out / 'selection.json'), auto_v2_config=copy.deepcopy(config),
            mask_source_directory=str(mask_root) if mask_root else None,
            tracking_requested=False, map_write=False, E1_run=False)
        return manifest

    save(out / 'automatic_seed_manifest.json', manifest_for(result['automatic_retained_group_ids'], 'auto_v2_automatic_33'))
    final_manifest = manifest_for(result['retained_group_ids'],
                                 'auto_v2_automatic_33' if automatic_only else 'auto_v2_room2_confirmed_35')
    final_manifest['manual_overrides'] = overrides
    save(out / 'seed_manifest.json', final_manifest)
    save(out / 'verification.json', dict(status='PASS', case_id=config['case_id'],
        counts=[dict(stage=s['stage'], input=s['input'], removed=s['removed'], retained=s['retained']) for s in result['stages']],
        automatic_retained=len(result['automatic_retained_group_ids']), final_retained=len(result['retained_group_ids']),
        manual_restored=result['manual_restored_group_ids'], all_historical_decisions_reproduced=True,
        input_sha256={name: sha(inputs / name) for name in ['source_lock.json', 'measurements.json', 'manual_overrides.json', 'reference_decisions.json']},
        original_manifest_sha256=sha(original_manifest),
        source_binary_masks_verified=bool(mask_root), verified_mask_sha256=mask_hashes,
        external_server_sources_verified=verify_server_sources,
        code_sha256={p.name: sha(p) for p in CODE.iterdir() if p.suffix in {'.py', '.json'}},
        output_sha256={name: sha(out / name) for name in ['selection.json', 'automatic_seed_manifest.json', 'seed_manifest.json']},
        limits=['No new segmentation, tracking or E1 execution', 'Manual restores apply only to this room2 input',
                'Original metric producers remain archived in experiments/auto_v1; this entry replays measured counts']))
    print(json.dumps(dict(automatic_retained=len(result['automatic_retained_group_ids']),
                         final_retained=len(result['retained_group_ids']),
                         manual_restored=result['manual_restored_group_ids'], output=str(out))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New output directory; existing directories are never overwritten')
    parser.add_argument('--original-manifest', help='Override original manifest location; its SHA-256 remains fixed')
    parser.add_argument('--mask-root', help='Original r4 directory containing masks/. Verify all 90 file hashes')
    parser.add_argument('--verify-server-sources', action='store_true', help='Also verify original server artifact and map hashes')
    parser.add_argument('--automatic-only', action='store_true', help='Export 33 seeds before the two explicit manual restores')
    args = parser.parse_args()
    replay(args.out, args.original_manifest, args.mask_root, args.verify_server_sources, args.automatic_only)

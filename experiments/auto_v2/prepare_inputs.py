"""Extract small, hash-linked measurements from the frozen room2 artifacts.

No inference, projection, image rendering, E1 repair or GT loading is performed.
This is an adapter for the accepted 2026-10-19 room2 experiment, not a new scene
measurement implementation. Legacy measurement producers are kept in auto_v1.
"""
import argparse
import hashlib
import json
from pathlib import Path

CASE_ID = 'room2_auto_v2_20261019'
BASELINE_SHA = '3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def extract(artifact_root, original_manifest, output, local_gallery_names=False):
    artifact_root, original_manifest, output = map(Path, (artifact_root, original_manifest, output))
    if output.exists():
        raise FileExistsError('Refuse to overwrite existing prepared inputs: ' + str(output))
    locks = {}

    def load_stage(directory, filename):
        path = artifact_root / (('room2_' if local_gallery_names else '') + directory) / filename
        locks[directory + '/' + filename] = dict(sha256=sha(path), bytes=path.stat().st_size,
            server_path='/data/chenkejun/CVPR/results/' + directory + '/' + filename)
        return json.loads(path.read_text(encoding='utf-8'))

    original = json.loads(original_manifest.read_text(encoding='utf-8'))
    original_hash = sha(original_manifest)
    sem = load_stage('seed_semantic_filter_20261018', 'classification.json')
    pair_data = load_stage('seed_covisible_dedup_iou_only_20261018', 'pair_results.json')
    measured = load_stage('seed_skip_repair_B_20261019', 'measurements.json')
    measurement_lock = load_stage('seed_skip_repair_B_20261019', 'measurement_verification.json')
    contains = load_stage('seed_containment_filter_20261019', 'pair_results.json')
    source42 = load_stage('seed_containment_filter_20261019', 'seed_manifest.json')
    if original_hash != sem['source_manifest_sha256']:
        raise ValueError('Original auto-v1 manifest changed')
    if sha_value := measurement_lock['source_sha256'].get(measured['source_map']):
        if sha_value != BASELINE_SHA:
            raise ValueError('Coverage uses a different baseline map')
    else:
        raise ValueError('Missing frozen baseline source lock')
    if measurement_lock['measurements_sha256'] != locks['seed_skip_repair_B_20261019/measurements.json']['sha256']:
        raise ValueError('Coverage measurements no longer match source verification')
    if (sem['config']['pixel_confidence'], sem['config']['mask_fraction']) != (.7, .9):
        raise ValueError('Semantic thresholds changed')
    if set(sem['config']['exclude_classes']) != {'wall', 'floor', 'ceiling', 'rug'}:
        raise ValueError('Semantic exclusion classes changed')
    if pair_data['config']['mode'] != 'iou_only':
        raise ValueError('Wrong pair reference version')
    bygroup = {group['id']: group for group in original['groups']}
    seeds = []
    for record in sem['records']:
        group = bygroup[record['group_id']]
        primary = group['primary']
        for key, other in [('mask_id', 'id'), ('frame', 'frame'), ('mask_sha256', 'mask_sha256'), ('area', 'area')]:
            if record[key] != primary[other]:
                raise ValueError('Semantic seed identity changed')
        item = {key: record[key] for key in ['group_id', 'original_track_id', 'mask_id', 'frame',
                    'mask_sha256', 'area', 'high_excluded_pixels']}
        item.update(support_count=primary['support_count'], support_sha256=primary['support_sha256'],
                    source_model=primary['representative']['model'], mask_digest=primary['digest'])
        if record['high_excluded_pixels'] != sum(category['high_confidence_pixels']
                for category in record['categories'] if category['name'] in sem['config']['exclude_classes']):
            raise ValueError('Semantic category totals differ')
        seeds.append(item)
    if len(seeds) != len(bygroup) or {r['group_id'] for r in seeds} != set(bygroup):
        raise ValueError('Incomplete original seed scope')
    pairs = [{key: p[key] for key in ['a', 'b', 'count_a', 'count_b', 'visible_a', 'visible_b',
              'intersection', 'union', 'original_intersection']} for p in pair_data['pairs']]
    pair_index = {tuple(sorted((p['a'], p['b']))): p for p in pairs}
    # Independently recomputed containment evidence must agree with original full supports.
    for pair in contains['pairs']:
        reference = pair_index[tuple(sorted((pair['a'], pair['b'])))]
        if reference['original_intersection'] != pair['intersection']:
            raise ValueError('Containment intersection differs from frozen full-support evidence')
    coverage_keys = ['group_id', 'original_track_id', 'mask_id', 'frame', 'mask_sha256',
                     'support_count', 'instance_id', 'intersection', 'instance_visible_count']
    coverage = [{key: row[key] for key in coverage_keys} for row in measured['primaries']]
    data = dict(schema='auto-v2-count-measurements-v1', case_id=CASE_ID, seeds=seeds, pairs=pairs,
                coverage=coverage, baseline_sha256=BASELINE_SHA,
                semantic_model=dict(model_id=sem['model']['model_id'], revision=sem['model']['revision'],
                                    weights_sha256=sem['model']['weights_sha256']),
                semantic_pixel_confidence=.7, semantic_exclude_classes=sem['config']['exclude_classes'])

    references = {}
    for name, directory, remove_key in [
        ('semantic', 'seed_semantic_filter_20261018', 'excluded'),
        ('covisible_iou_only', 'seed_covisible_dedup_iou_only_20261018', 'excluded'),
        ('skip_repair_B', 'seed_skip_repair_B_20261019', 'skipped'),
        ('full_support_containment', 'seed_containment_filter_20261019', 'excluded'),
        ('pr_filter', 'seed_pr_filter_20261019', 'excluded')]:
        value = load_stage(directory, 'selection_binary.json')
        references[name] = dict(removed_group_ids=sorted(row['group_id'] for row in value[remove_key]),
                               retained_group_ids=sorted(row['group_id'] for row in value['retained']))
    ids42 = sorted(group['id'] for group in source42['groups'])
    if ids42 != references['full_support_containment']['retained_group_ids']:
        raise ValueError('42-seed reference manifest disagrees')
    identities = {row['group_id']: {key: row[key] for key in coverage_keys[:5]} for row in coverage}
    override = dict(schema='auto-v2-case-manual-override-v1', case_id=CASE_ID, date='2026-10-19',
        stage='pr_filter', scope='Only this room2 42-seed input, not a general automatic rule',
        user_quote='等一下，刚才的筛选保留017和025其他的都排除之后用保留的做E1之后的修复',
        input_manifest_sha256=locks['seed_containment_filter_20261019/seed_manifest.json']['sha256'],
        input_group_ids=ids42, automatic_excluded_group_ids=references['pr_filter']['removed_group_ids'],
        retain=[identities[gid] for gid in ['G017', 'G025']],
        final_excluded_group_ids=['G069', 'G071', 'G074', 'G075', 'G077', 'G082', 'G086'])
    output.mkdir(parents=True)
    save(output / 'measurements.json', data)
    save(output / 'manual_overrides.json', override)
    save(output / 'reference_decisions.json', dict(schema='auto-v2-historical-validation-only-v1',
        not_used_by_algorithm=True, stages=references,
        final_retained_group_ids=sorted(set(references['pr_filter']['retained_group_ids']) | {'G017', 'G025'})))
    save(output / 'source_lock.json', dict(schema='auto-v2-room2-sources-v1', case_id=CASE_ID,
        original_manifest=dict(server_path='/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4/seed_manifest.json',
                               sha256=original_hash), sources=locks,
        baseline=dict(path=measured['source_map'], sha256=BASELINE_SHA),
        evidence=dict(path='/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/surface_evidence.npz',
                      sha256='b106b99741a4cb115d9ceed0ea3604ed46e0cc9a48d1feb865c2a0a60661c992'),
        local_files_sha256={name: sha(output / name) for name in
                           ['measurements.json', 'manual_overrides.json', 'reference_decisions.json']}))
    print(json.dumps(dict(seeds=len(seeds), pairs=len(pairs), coverage=len(coverage), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--local-gallery-names', action='store_true')
    arguments = parser.parse_args()
    extract(arguments.artifact_root, arguments.original_manifest, arguments.out, arguments.local_gallery_names)

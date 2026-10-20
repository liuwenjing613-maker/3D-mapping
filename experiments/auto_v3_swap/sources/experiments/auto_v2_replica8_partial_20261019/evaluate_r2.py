"""Score baseline, original auto-v2 E1, and whole-to-partial using main R2."""
from bootstrap import C, R, SOURCE, MAIN, SCENE, COUNT, verify_sources
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import numpy as np
from unified_eval.current_protocol import CURRENT_PROTOCOL, CURRENT_CONFIG, REGISTRY_PATH, require_current_protocol, require_current_gt
from unified_eval.io import load_gt, sha256_file
from unified_eval.geometry import load_fixed_surface_correspondence

read = lambda p: json.loads(Path(p).read_text())
OUT = R / 'evaluation_r2'


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def cli(args, log):
    with log.open('w') as stream:
        run = subprocess.run([sys.executable, '-m', 'unified_eval.cli', *map(str, args)],
                             cwd=MAIN, stdout=stream, stderr=subprocess.STDOUT)
    assert run.returncode == 0, str(log)


def main():
    verify_sources()
    assert require_current_protocol(CURRENT_CONFIG)
    freeze_path = R / 'predictions_freeze.json'
    freeze = read(freeze_path)
    assert freeze['status'] == 'PREDICTIONS_FROZEN_BEFORE_GT' and not freeze['GT_read_by_mapper']
    for kind in ('code_sha256', 'output_sha256', 'source_sha256'):
        for p, digest in freeze[kind].items():
            assert sha256_file(p) == digest, p
    old_freeze_path = SOURCE / 'e1/E1_expanded_history/predictions_freeze.json'
    old_freeze = read(old_freeze_path)
    for kind in ('code_sha256', 'output_sha256'):
        for p, digest in old_freeze[kind].items():
            assert sha256_file(p) == digest, p
    OUT.mkdir(exist_ok=True)
    gtpath = Path(CURRENT_PROTOCOL['canonical_GT'][SCENE]['file'])
    assert require_current_gt(gtpath) == SCENE
    gt = load_gt(gtpath)
    materialpath = Path(f'/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native/{SCENE}/P1-A1/surface_p0/materialization_report.json')
    associationpath = materialpath.parent.parent / 'association/association_report.json'
    material, association = read(materialpath), read(associationpath)
    frames = [x['frame_id'] for x in material['frame_records']]
    assert frames == [x['frame_id'] for x in association['frame_summaries']] == gt.metadata['observation_metadata']['frame_ids'] == list(range(0, 2000, 5))
    fsha = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
    assert fsha == gt.metadata['observation_metadata']['frame_list_sha256']
    config = Path(f'/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/{SCENE}/configs/raw.json')
    assert sha256_file(config) == material['config_sha256'] == association['config_sha256']
    cache_root=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache') if SCENE in ['room0','room2'] else Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/geometry_cache')
    cache=cache_root/SCENE/'fixed_gt_to_tsdf_r2.npz'
    assert sha256_file(cache)==read(SOURCE/'evaluation_r2/protocol_freeze.json')['fixed_correspondence_sha256']
    dump(OUT / 'protocol_freeze.json', {'current_protocol': CURRENT_PROTOCOL,
        'registry_sha256': sha256_file(REGISTRY_PATH), 'default_main_CLI': True,
        'entry_commit': subprocess.check_output(['git', '-C', str(MAIN), 'rev-parse', 'HEAD'], text=True).strip(),
        'fixed_correspondence_sha256': sha256_file(cache), 'predictions_frozen_before_GT': True})
    baseline = read(SOURCE / 'original_repair/predictions_freeze.json')['scenes'][SCENE]['predictions']
    roots = {'old_auto_v2': SOURCE / 'e1/E1_expanded_history' / SCENE,
             'whole_to_partial': R / 'whole_to_partial' / SCENE}
    variants = ('baseline', *roots)
    rows, manifests = [], []
    for stage in ('native', 'final'):
        reference = np.load(baseline['strict_baseline_' + stage]['path'])
        for variant in variants:
            surface = Path(baseline['strict_baseline_' + stage]['path']) if variant == 'baseline' else roots[variant] / stage / 'instance_surface.npz'
            z = np.load(surface)
            np.testing.assert_array_equal(z['xyz_m'], reference['xyz_m'])
            np.testing.assert_array_equal(z['rgb'], reference['rgb'])
            load_fixed_surface_correspondence(cache, z['xyz_m'], gt.xyz_ref)
            provenance = {'input_scope_verified': True, 'frame_list_sha256': fsha, 'frame_count': 400,
                'input_config_sha256': sha256_file(config), 'materialization_report_sha256': sha256_file(materialpath),
                'association_report_sha256': sha256_file(associationpath),
                'prediction_freeze_sha256': sha256_file(freeze_path), 'prediction_frozen_before_GT': True,
                'GT_used_for_mapping': False, 'condition': variant, 'selected_primary_masks': 0 if variant == 'baseline' else COUNT,
                'batch': str(R), 'surface_sha256': sha256_file(surface)}
            pp = OUT / 'provenance' / (variant + '_' + stage + '.json')
            dump(pp, provenance)
            dest = OUT / (variant + '_' + stage)
            cli(['adapt-surface', '--gt', gtpath, '--surface', surface, '--fixed-correspondence', cache,
                 '--source-provenance', pp, '--method-name', 'autov2_partial_' + SCENE + '_' + variant + '_' + stage,
                 '--method-commit', sha256_file(C / 'run_partial_replay.py'), '--out', dest],
                 OUT / (variant + '_' + stage + '.log'))
            manifests.append(read(dest / 'adapter_manifest.json'))
            sm = read(dest / 'metrics.json')['summary']
            labels = z['instance_id']
            epath = Path(baseline['strict_baseline_final']['path']).parent.parent/'surface_evidence.npz' if variant == 'baseline' else roots[variant] / 'native/surface_evidence.npz'
            evidence = np.load(epath)
            row = {'variant': variant, 'stage': stage, 'F1': sm['CA_PRF1_0_5']['F1'],
                   'PQ': sm['CA_PQ']['PQ'], 'AP': sm['CA_AP_uniform'], 'AP50': sm['CA_AP50_uniform'],
                   'mCov': sm['CA_mCov'], 'TP': sm['CA_PRF1_0_5']['TP'], 'FP': sm['CA_PRF1_0_5']['FP'],
                   'FN': sm['CA_PRF1_0_5']['FN'], 'gray_total': int((labels <= 0).sum()),
                   'gray_states': {n: int(((labels <= 0) & (evidence['state'] == i)).sum()) for n, i in [('U', 0), ('T', 1), ('C', 3)]},
                   'owner_surface': sm['owner_surface']}
            rows.append(row)
            print('SCORED', json.dumps(row), flush=True)
        for before, after in (('baseline', 'whole_to_partial'), ('old_auto_v2', 'whole_to_partial')):
            label = before + '_to_' + after + '_' + stage
            cli(['eval-repair-pair', '--gt', gtpath, '--before', OUT / (before + '_' + stage) / 'canonical_prediction.npz',
                 '--after', OUT / (after + '_' + stage) / 'canonical_prediction.npz', '--out', OUT / 'paired' / label],
                 OUT / ('paired_' + label + '.log'))
    for m in manifests[1:]:
        for key in ('geometry_xyz_sha256', 'correspondence_sha256', 'gt_scope_sha256',
                    'gt_observed_support_sha256', 'profile_config_sha256'):
            assert m[key] == manifests[0][key], key
    old_rows = read(SOURCE / 'evaluation_r2/comparison.json')['rows']
    for stage in ('native', 'final'):
        for current, source_variant in (('baseline', 'before'), ('old_auto_v2', 'after')):
            new = next(x for x in rows if x['variant'] == current and x['stage'] == stage)
            old = next(x for x in old_rows if x['variant'] == source_variant and x['stage'] == stage)
            for key in ('F1', 'PQ', 'AP', 'AP50', 'mCov', 'TP', 'FP', 'FN', 'gray_total'):
                assert abs(new[key] - old[key]) < 1e-12, (stage, current, key)
    scope = {int(x['raw_gt_id']): x for x in read(gtpath.parent / 'gt_scope.json')['objects']}
    overlaps = {v: dict(np.load(OUT / (v + '_final') / 'overlap_matrix.npz')) for v in variants}
    gids = overlaps['baseline']['gt_ids']
    pergt = []
    for gid in gids:
        row = {'GT_ID': int(gid), 'semantic_class': scope[int(gid)]['semantic_class']}
        for v, z in overlaps.items():
            assert set(z['gt_ids']) == set(gids)
            index = int(np.flatnonzero(z['gt_ids'] == gid)[0])
            row[v] = float(z['iou'][:, index].max(initial=0))
        row['partial_minus_old'] = row['whole_to_partial'] - row['old_auto_v2']
        pergt.append(row)
    dump(OUT / 'per_GT.json', pergt)
    for name, values in [('metrics', rows), ('per_GT', pergt)]:
        flat = [{k: v for k, v in x.items() if not isinstance(v, dict)} for x in values]
        with (OUT / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    result = {'status': 'PASS', 'protocol': 'object_observed_repair / revision 2',
              'development_protocol_frozen': False, 'scene': SCENE, 'GT_count': len(gids),
              'rows': rows, 'same_geometry_GT_scope_and_correspondence': True,
              'baseline_and_old_E1_scores_reproduced_exactly': True,
              'GT_changes': {key: {'improved': sum(x[key] > 1e-12 for x in pergt),
                                    'worsened': sum(x[key] < -1e-12 for x in pergt),
                                    'unchanged': sum(abs(x[key]) <= 1e-12 for x in pergt)}
                             for key in ('partial_minus_old',)}}
    dump(OUT / 'comparison.json', result)
    dump(R / 'evaluation_complete.json', result)
    verify_sources()
    print('EVALUATION_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

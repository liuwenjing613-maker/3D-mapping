"""Frozen gap-only ablation: exact saved-mask truncation, unchanged repair and v3."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import traceback
import numpy as np
from PIL import Image
from continuity_policy import track_window, in_window

CODE = Path(__file__).resolve().parent
OUT = Path('/data/chenkejun/CVPR/revisable_instance_map/continuous_tracking_ablation_20261008')
LEGACY = Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
SOURCE = {
    'room0': ('room0_repair_20261007', 'batch_386cdcff710d'),
    'room2': ('room2_top30_repair_20261008', 'batch_6b3f32f5dc1c'),
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_engine(scene):
    dirname, batch = SOURCE[scene]
    path = Path('/home/chenkejun/CVPR/experiments') / dirname / batch / 'run_batch.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('frozen_gap_engine_' + scene, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == Path('/data/chenkejun/CVPR/revisable_instance_map') / dirname / batch
    assert (module.r.ROOT / 'repair/baseline_controls' / scene / 'complete.json').exists(), 'Require the existing no-op control; never rewrite the source experiment'
    return module


def validate_sources(engine, cases):
    freeze = read(engine.ROOT / 'input_freeze.json')
    for key in ['seed_and_config_hashes', 'original_baseline_source_hashes', 'raw_rgb_hashes']:
        engine.check_hashes(freeze[key])
    protected = dict(read(engine.ROOT / 'validated/full_track/source_freeze.json')['source_hashes'])
    protected.update(read(engine.ROOT / 'validated/full_track/complete.json')['output_sha256'])
    engine.check_hashes(protected)
    tracking = {}
    verified = 0
    for uid, data in cases.items():
        report_path = engine.ROOT / 'cases' / uid / 'tracking/complete.json'
        report = read(report_path)
        assert report['status'] == 'PASS' and report['completed_frames'] == report['total_frames'] == 2000
        assert not report['GT_used'] and report['directions_use_independent_seed_only_states']
        assert report['case_config_sha256'] == engine.r.sha(data['case']['config_path'])
        assert report['code_sha256'] == freeze['tracking_code_sha256'] == engine.r.sha(freeze['tracking_code_path'])
        assert report['seed_frame'] == data['config']['seed_frame'] and report['track_ids'] == data['ids']
        frames = {row['frame']: row for row in report['frames']}
        assert sorted(frames) == list(range(2000))
        assert report['seed_sha256'] == engine.r.sha(engine.ROOT / 'cases' / uid / 'prepared_seed.png')
        for fid, row in frames.items():
            assert engine.r.sha(row['label_file']) == row['label_sha256']
            if fid != report['seed_frame']:
                assert row['direction'] == ('reverse' if fid < report['seed_frame'] else 'forward')
            verified += 1
        protected[str(report_path)] = engine.r.sha(report_path)
        tracking[uid] = frames
    for name in ['global_association.json', 'input_freeze.json', 'alias_review.json',
                 'seed_manifest.json', 'tracking_validation.json', 'evaluation_freeze.json']:
        protected[str(engine.ROOT / name)] = engine.r.sha(engine.ROOT / name)
    return tracking, protected, verified


def freeze_windows(engine, scene, cases, tracking):
    result = {}
    for gap in (0, 1):
        expected = read(CODE / f'{scene}_gap{gap}_windows.json')
        by_track = {w['track']: w for w in expected['tracks']}
        assert expected['max_empty_gap'] == gap and not expected['GT_used']
        windows = {}
        for uid, data in cases.items():
            for oid in data['ids']:
                areas = [tracking[uid][fid]['areas'][str(oid)] for fid in range(2000)]
                window = track_window(areas, data['config']['seed_frame'], gap)
                assert all(by_track[oid][key] == value for key, value in window.items())
                windows[oid] = window
        result[f'gap{gap}'] = windows
        dump(OUT / scene / f'gap{gap}/windows.json', expected)
    for oid in result['gap0']:
        assert all(not in_window(result['gap0'][oid], f) or in_window(result['gap1'][oid], f) for f in range(2000))
    return result


def materialize_mapping_masks(engine, scene, cases, tracking, windows, mapping_frames):
    adjusted = {name: {uid: dict(frames) for uid, frames in tracking.items()} for name in windows}
    hashes = {name: {} for name in windows}
    for uid, data in cases.items():
        for fid in sorted(mapping_frames):
            record = tracking[uid][fid]
            labels = np.asarray(Image.open(record['label_file']), dtype=np.uint16)
            assert labels.shape == data['seed'].shape
            assert set(np.unique(labels)).issubset({0, *data['ids']})
            for oid in data['ids']:
                assert int(np.sum(labels == oid)) == record['areas'][str(oid)]
            for name, per_track in windows.items():
                retained = [oid for oid in data['ids'] if in_window(per_track[oid], fid)]
                clipped = np.where(np.isin(labels, retained), labels, 0).astype(np.uint16)
                for oid in retained:
                    np.testing.assert_array_equal(clipped == oid, labels == oid)
                assert not np.any(np.isin(clipped, [oid for oid in data['ids'] if oid not in retained]))
                if fid == data['config']['seed_frame']:
                    np.testing.assert_array_equal(clipped, data['seed'])
                if np.array_equal(clipped, labels):
                    path = Path(record['label_file'])
                else:
                    path = OUT / scene / name / 'mapping_masks' / uid / f'f{fid:06d}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        np.testing.assert_array_equal(np.asarray(Image.open(path), dtype=np.uint16), clipped)
                    else:
                        Image.fromarray(clipped).save(path)
                digest = engine.r.sha(path)
                adjusted[name][uid][fid] = {**record, 'label_file': str(path), 'label_sha256': digest,
                    'areas': {str(oid): int(np.sum(clipped == oid)) for oid in data['ids']}}
                hashes[name][str(path)] = digest
    return adjusted, hashes


def export_ply(engine, scene, name, surface):
    arrays = engine.f.load_arrays(surface)
    colors = read(engine.ROOT / 'review/instance_colors.json')['id_colors']
    ids = arrays['instance_id']
    known = set(map(int, colors))
    assert set(map(int, np.unique(ids[ids > 0]))).issubset(known), 'Require the original complete identity palette'
    rgb = np.full((len(ids), 3), [105, 105, 105], dtype=np.uint8)
    for pid in np.unique(ids[ids > 0]):
        rgb[ids == pid] = colors[str(int(pid))]
    dtype = np.dtype([(n, '<f4') for n in ['x', 'y', 'z']] + [(n, 'u1') for n in ['red', 'green', 'blue']] + [('instance_id', '<i4')])
    cloud = np.empty(len(ids), dtype=dtype)
    for i, n in enumerate(['x', 'y', 'z']): cloud[n] = arrays['xyz_m'][:, i]
    for i, n in enumerate(['red', 'green', 'blue']): cloud[n] = rgb[:, i]
    cloud['instance_id'] = ids
    header = '\n'.join(['ply', 'format binary_little_endian 1.0',
        'comment Exact original TSDF geometry; unchanged instance colors; no downsampling',
        f'element vertex {len(ids)}', 'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue', 'property int instance_id', 'end_header', '']).encode('ascii')
    path = OUT / scene / name / f'{scene}_{name}_full_instance.ply'
    with path.open('wb') as stream: stream.write(header);cloud.tofile(stream)
    restored = np.memmap(path, mode='r', dtype=dtype, offset=len(header), shape=(len(ids),))
    for n in dtype.names: np.testing.assert_array_equal(restored[n], cloud[n])
    return {'path':str(path), 'sha256':engine.r.sha(path), 'points':len(ids), 'bytes':path.stat().st_size,
            'original_ID_colors_preserved':True, 'geometry_unchanged':True, 'downsampled':False}


def run_scene(scene):
    started = time.monotonic()
    engine = load_engine(scene)
    data_scene, cases, objects, canonical, policy = engine.inputs()
    association = read(engine.ROOT / 'global_association.json')
    assert association['status'] == 'PASS' and not association['GT_used']
    assert {oid:row['canonical_track_id'] for row in association['objects'] for oid in row['member_track_ids']} == canonical
    tracking, protected, verified = validate_sources(engine, cases)
    windows = freeze_windows(engine, scene, cases, tracking)
    own_sources = {str(CODE/n):engine.r.sha(CODE/n) for n in
                   ['run_ablation.py', 'continuity_policy.py', f'{scene}_gap0_windows.json', f'{scene}_gap1_windows.json']}
    freeze = {'status':'FROZEN_BEFORE_NEW_REPAIR_AND_GT', 'scene':scene, 'source_hashes':protected,
        'own_code_and_policy_hashes':own_sources, 'engine_source_sha256':engine.r.sha(engine.__file__),
        'only_intervention':'Set original saved track pixels to background outside their independently seed-derived continuity intervals',
        'SAM_rerun':False, 'GT_used_for_filter_or_repair':False,
        'unchanged':['human seeds','forward and reverse seed-only propagation prefixes','per-frame largest-logit partition',
                     'canonical identity and aliases','original observation retirement gates','one frame/surface/identity vote',
                     'confirmation thresholds','geodesic diffusion settings','bounded commit','unified v3 protocol and flags'],
        'verified_original_tracking_PNGs':verified}
    dump(OUT / scene / 'source_freeze.json', freeze)
    adjusted, label_hashes = materialize_mapping_masks(engine, scene, cases, tracking, windows, data_scene.records)
    outcomes = {}
    for name in ['gap0', 'gap1']:
        engine.check_hashes(protected);engine.check_hashes(own_sources);engine.check_hashes(label_hashes[name])
        engine.REPAIR_ROOT = OUT / scene / name
        result = engine.repair('full_track', data_scene, cases, objects, canonical, policy, association, adjusted[name])
        assert result['status'] == 'PASS' and result['complete_frame_replay_bit_identical_to_delta_ledger'] and result['exact_rollback_pass']
        assert result['strict_commit']['final_outside_changes'] == 0
        decisions = read(engine.REPAIR_ROOT / 'full_track/frame_decisions.json')
        for row in decisions['frames']:
            assert all(in_window(windows[name][oid], row['frame']) for oid in row['accepted_track_ids'])
        engine.check_hashes(protected);engine.check_hashes(own_sources);engine.check_hashes(label_hashes[name])
        surface = engine.REPAIR_ROOT / 'full_track/final/instance_surface.npz'
        outcomes[name] = {'surface':str(surface),'sha256':engine.r.sha(surface),
                          'repair_receipt':str(engine.REPAIR_ROOT / 'full_track/complete.json'),
                          'mapping_masks_source_hashes':label_hashes[name],
                          'PLY':export_ply(engine,scene,name,surface)}
    data_scene.verify_unchanged();engine.check_hashes(protected)
    dump(OUT / scene / 'predictions_frozen.json', {'status':'PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT',
         'scene':scene,'conditions':outcomes,'source_freeze_sha256':engine.r.sha(OUT/scene/'source_freeze.json'),
         'GT_used_for_filter_or_repair':False,'old_outputs_preserved':True,'seconds':round(time.monotonic()-started,2)})
    return outcomes


def evaluate_scene(scene):
    engine = load_engine(scene)
    source_freeze = read(OUT / scene / 'source_freeze.json')
    engine.check_hashes(source_freeze['source_hashes'])
    engine.check_hashes(source_freeze['own_code_and_policy_hashes'])
    frozen = read(OUT / scene / 'predictions_frozen.json')
    assert frozen['status'] == 'PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT'
    for row in frozen['conditions'].values(): assert engine.r.sha(row['surface']) == row['sha256']
    sys.path.insert(0, str(LEGACY))
    import evaluate_pilot_v3 as e
    original_freeze = read(engine.ROOT / 'evaluation_freeze.json')
    assert original_freeze['evaluator_sha256'] == engine.r.sha(e.__file__)
    assert original_freeze['flags'] == e.FLAGS and original_freeze['protocol_sha256'] == engine.r.sha(e.PROTOCOL)
    parser = e.ArgumentParser();e.add_protocol_args(parser)
    protocol, raw, debug = e.load_protocol(e.PROTOCOL,parser.parse_args(['--config',str(e.PROTOCOL),*e.FLAGS]))
    assert raw == original_freeze['effective_protocol']
    dump(OUT/scene/'evaluation_freeze.json', {'status':'FROZEN_BEFORE_NEW_GT_EVALUATION',
        'predictions_frozen_sha256':engine.r.sha(OUT/scene/'predictions_frozen.json'),
        'evaluator_sha256':engine.r.sha(e.__file__),'flags':e.FLAGS,'protocol_sha256':engine.r.sha(e.PROTOCOL),
        'supplementary_reporting_wrapper':str(Path(__file__).resolve()),'supplementary_reporting_wrapper_sha256':engine.r.sha(__file__),
        'effective_protocol':raw,'GT_used_for_filter_or_repair':False})
    reference = read(engine.ROOT / 'evaluation_summary.json')
    gt = e.load_gt(e.INPUT / scene / 'ground_truth/gt.npz')
    base_map = engine.f.load_arrays(original_freeze['predictions']['baseline']['path'])
    original = engine.f.load_arrays(engine.ROOT/'validated/full_track/final/instance_surface.npz')
    mapping_path = engine.ROOT / 'fixed_surface_mapping.npz'
    assert mapping_path.exists(), 'Reuse the existing fixed evaluation mapping'
    mapping_hash = engine.r.sha(mapping_path)
    mapping = engine.f.fixed_mapping(mapping_path,base_map['xyz_m'],gt.xyz_ref,.01)
    assert mapping['cache_sha256'] == mapping_hash
    valid = gt.valid_vertex_mask & ~gt.ignore_vertex_mask
    target_ids = set(reference['target_GT_ids'])
    identity_gt = {row['persistent_id']:row['GT_id'] for row in reference['fixed_surface_diagnostics'] if row.get('GT_id') is not None}
    base_v3 = read(e.ROOT / 'v3/baseline' / scene / 'complete.json')
    unrestricted_v3 = read(engine.ROOT/'validated/full_track/v3/complete.json')
    unrestricted_gt = e.per_gt(e.load_overlap(engine.ROOT/'validated/full_track/v3/overlap.npz'))
    results = {}
    for name, item in frozen['conditions'].items():
        arrays = engine.f.load_arrays(item['surface'])
        engine.f.require_same_geometry(base_map,arrays)
        score = e.evaluate(scene,Path(item['surface']),OUT/scene/name/'v3',protocol)
        for key in ['flags','protocol_sha256','GT_sha256','evaluation_code_sha256']:
            assert score[key] == base_v3[key] == unrestricted_v3[key]
        per_gt = e.per_gt(e.load_overlap(OUT/scene/name/'v3/overlap.npz'))
        lost = [g for g in unrestricted_gt if g not in target_ids and unrestricted_gt[g]['matched_IoU_gt_0_5'] and not per_gt[g]['matched_IoU_gt_0_5']]
        changed = arrays['instance_id'] != original['instance_id']
        unknown = arrays['instance_id'] <= 0
        report = read(OUT/scene/name/'full_track/complete.json')
        results[name] = {'metrics':score['metrics'],'repair':{k:report[k] for k in
           ['counts','whole_scene_unknown_before','whole_scene_unknown_after','assigned_to_unassigned_inside','unassigned_to_assigned_inside','removed_frame_surface_identity_votes','added_frame_surface_identity_votes','strict_commit']},
           'vs_unrestricted':{'changed_surface_points':int(changed.sum()),
                'new_unknown_points':int(np.sum((original['instance_id']>0)&unknown)),
                'restored_assigned_points':int(np.sum((original['instance_id']<=0)&~unknown)),
                'non_target_previously_correct_lost':lost,
                'target_mean_best_IoU_delta':float(np.mean([per_gt[g]['best_IoU']-unrestricted_gt[g]['best_IoU'] for g in target_ids])),
                'target_completeness_mean_delta':float(np.mean([per_gt[g]['completeness_best_single_instance']-unrestricted_gt[g]['completeness_best_single_instance'] for g in target_ids]))},
           'targets':[{'unrestricted':unrestricted_gt[g],'after':per_gt[g]} for g in sorted(target_ids)],
           'fixed_identity_diagnostics':{'targets':[engine.f.target_diagnostics(mapping,arrays['instance_id'],gt.instance_id,valid,{pid:gid})['targets'][0] for pid,gid in sorted(identity_gt.items())], 'multiple_selected_parts_of_same_GT_preserved':True, 'same_single_identity_diagnostics_as_original_evaluate_batch':True},
           'PLY':item['PLY']}
        assert engine.r.sha(item['surface']) == item['sha256']
    assert engine.r.sha(mapping_path) == mapping_hash
    engine.check_hashes(source_freeze['source_hashes'])
    engine.check_hashes(source_freeze['own_code_and_policy_hashes'])
    dump(OUT/scene/'evaluation_summary.json',{'status':'PASS','scene':scene,
        'protocol_frozen':bool(raw.get('frozen',False)),'formal_benchmark_result':False,
        'unified_v3_code_protocol_and_flags_unchanged':True,'GT_used_for_filter_or_repair':False,
        'fixed_original_target_GT_ids':sorted(target_ids),'baseline_metrics':base_v3['metrics'],
        'unrestricted_metrics':unrestricted_v3['metrics'],'conditions':results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--scene',choices=SOURCE,required=True)
    parser.add_argument('--stage',choices=['evaluate'],required=True)
    args = parser.parse_args()
    try:
        if args.stage == 'repair': run_scene(args.scene)
        else: evaluate_scene(args.scene)
    except Exception:
        dump(OUT/args.scene/(args.stage+'_failure.json'),{'status':'FAIL','traceback':traceback.format_exc()})
        raise

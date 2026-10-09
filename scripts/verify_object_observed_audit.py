"""Verify frozen baseline, committed scoring bytes and every revision-2 artifact."""
from pathlib import Path
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_eval.io import load_gt, load_prediction, repair_evaluator_code_hashes, sha256_file
from unified_eval.repair_profile import write_json

CODE = Path(__file__).resolve().parents[1]
OUT = Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_audit_20261009')
OLD = Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009')
SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']


def main():
    checked = {}
    def verify(path, expected):
        path = str(path)
        if path not in checked:
            checked[path] = sha256_file(path)
        assert checked[path] == expected, 'Artifact changed: ' + path
    lock = json.loads((OUT / 'development_baseline_lock.json').read_text())
    for path, expected in lock['file_sha256'].items():
        verify(path, expected)
    code_hashes = repair_evaluator_code_hashes()
    for name, expected in code_hashes.items():
        blob = subprocess.check_output(['git', '-C', str(CODE), 'show', 'cd26056:unified_eval/' + name])
        import hashlib
        assert hashlib.sha256(blob).hexdigest() == expected, 'Scoring differs from committed code bytes'
    config = CODE / 'unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json'
    blob = subprocess.check_output(['git', '-C', str(CODE), 'show', 'cd26056:unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json'])
    assert config.read_bytes() == blob
    progress = json.loads((OUT / 'progress.json').read_text())
    assert progress['stage'] == 'complete' and set(progress['completed_scenes']) == set(SCENES)
    observation = json.loads((OUT / 'observation_audit_complete.json').read_text())
    assert len(observation['per_scene']) == 8 and all(x['support_reproduced'] for x in observation['per_scene'])
    scene_rows, run_count = [], 0
    for scene in SCENES:
        paths = [OUT / scene / name / 'adapter_manifest.json' for name in
                 ['P1-A1_native', 'P1-A1_holes_geodesic', 'P1-A1_auto_v2', 'OVI-MAP_full400']]
        if scene in ('room0', 'room2'):
            paths += [OUT / scene / name / 'adapter_manifest.json' for name in ('P1-A1_human_full_track', 'P1-A1_human_seed_only')]
        manifests = []
        gt = load_gt(OLD / 'mesh' / scene / 'gt.npz')
        for key in ('source_reference', 'source_manifest', 'source_mesh', 'source_instance_labels', 'source_semantic_labels'):
            verify(gt.metadata[key], gt.metadata[key + '_sha256'])
        gt_scope = json.loads((OLD / 'mesh' / scene / 'gt_scope.json').read_text())
        verify(gt_scope['official_annotation_source'], gt_scope['official_annotation_sha256'])
        for path in paths:
            manifest = json.loads(path.read_text())
            assert manifest['profile_revision'] == 2 and manifest['evaluator_code_sha256'] == code_hashes
            assert manifest['profile_config_sha256'] == sha256_file(config)
            assert manifest['gt_scope_sha256'] == gt.metadata['gt_scope_sha256']
            assert manifest['gt_observed_support_sha256'] == gt.metadata['gt_observed_support_sha256']
            for path_key, hash_key in [('source_surface', 'source_surface_sha256'), ('gt_file', 'gt_file_sha256'),
                                      ('correspondence_cache', 'correspondence_file_sha256')]:
                verify(manifest[path_key], manifest[hash_key])
            verify(path.parent / 'canonical_prediction.npz', manifest['canonical_prediction_sha256'])
            verify(path.parent / 'diagnostic_support_prediction.npz', manifest['diagnostic_prediction_sha256'])
            prediction = load_prediction(path.parent / 'canonical_prediction.npz')
            with np.load(manifest['source_surface'], allow_pickle=False) as source:
                if 'xyz_m' in source:
                    inventory = source['native_instance_ids'] if 'native_instance_ids' in source else np.unique(source['instance_id'][source['instance_id'] > 0])
                    expected_uids = {str(int(raw)) for raw in inventory}
                else:
                    expected_uids = {f'ovi-id:{int(raw)}' for raw in source['native_instance_ids']}
            assert {instance.instance_uid for instance in prediction.instances} == expected_uids, 'Exported instance inventory changed'
            baseline = json.loads((OLD / 'mesh' / scene / ('OVI-MAP_full400' if path.parent.name.startswith('OVI') else 'P1-A1_holes_geodesic') / 'adapter_manifest.json').read_text())
            assert prediction.metadata['geometry_xyz_sha256'] == baseline['geometry_xyz_sha256']
            overlap = np.load(path.parent / 'overlap_matrix.npz', allow_pickle=False)
            assert len(overlap['pred_uids']) == len(prediction.instances)
            manifests.append(manifest)
            run_count += 1
        p1 = [x for x in manifests if not x['method_name'].startswith('OVI')]
        assert len({x['geometry_xyz_sha256'] for x in p1}) == 1
        assert len({x['correspondence_sha256'] for x in p1}) == 1
        assert len({x['gt_scope_sha256'] for x in manifests}) == 1
        with np.load(OUT / 'GT_review' / scene / 'depth_tolerance_counts.npz', allow_pickle=False) as counts:
            assert np.array_equal(counts['count_2cm'], gt.observation_count)
        scene_rows.append({'scene_id': scene, 'GT_scope_preserved': True, 'depth_count_reproduced_exactly': True,
                           'all_P1_versions_share_geometry_and_correspondence': True, 'prediction_inventories_retained': True})
    tests = ET.parse(OUT / 'unified_eval_tests.xml').getroot().find('testsuite').attrib
    assert int(tests['tests']) == 91 and all(int(tests[key]) == 0 for key in ('skipped', 'errors', 'failures'))
    report = {'status': 'PASS_TECHNICAL_AUDIT_NOT_OFFICIAL_PROTOCOL_FREEZE', 'baseline_commit': lock['code_commit'],
        'baseline_locked_files_verified_unchanged': lock['locked_file_count'], 'method_scene_runs': run_count,
        'scoring_commit': progress['audit_code_commit'], 'profile_config_sha256': sha256_file(config),
        'evaluator_code_sha256': code_hashes, 'verified_file_count': len(checked), 'per_scene': scene_rows,
        'tests': tests, 'GT_review_driver_sha256': sha256_file(CODE / 'scripts/audit_object_observed_gt_support.py'),
        'structure_review_driver_sha256': sha256_file(CODE / 'scripts/render_repair_structure_review.py'),
        'pending_for_official_freeze': ['6 tiny GT quality decisions', '20 real structure case final review',
            'predeclared geometry/observation thresholds approval independent of method ranking', 'background-only/empty-boundary policy final decision']}
    write_json(OUT / 'audit_integrity_verification.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'method_scene_runs', 'baseline_locked_files_verified_unchanged', 'verified_file_count')}))


if __name__ == '__main__':
    main()

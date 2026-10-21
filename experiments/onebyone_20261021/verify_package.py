"""Read-only verification of the archived onebyone experiment; standard library only."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
CODE_ROOT = '/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021/'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT / 'package_manifest.json')
    for name, item in manifest['files'].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve()), name
        assert path.is_file(), name
        assert path.stat().st_size == item['bytes'], name
        assert sha(path) == item['sha256'], name
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'package_manifest.json'}, 'Unexpected or missing package files'
    for path in (ROOT / 'sources').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cfg = read(ROOT / 'sources/config.json')
    source_lock = read(ROOT / 'evidence/input_lock.json')
    assert sha(ROOT / 'sources/config.json') == source_lock['config_sha256']
    comparison = read(ROOT / 'evidence/comparison.json')
    assert comparison['status'] == 'PASS' and comparison['frozen'] is False
    assert comparison['GT_count'] == 350 and all(comparison['verified'].values())
    assert len(comparison['scenes']) == 16
    scenes = {'room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4'}
    for group, count in [('automatic', 196), ('manual', 76)]:
        assert set(cfg[group]) == scenes
        assert sum(x['selected'] for x in cfg[group].values()) == count
        rows = [x for x in comparison['scenes'] if x['group'] == group]
        assert {x['scene'] for x in rows} == scenes
        assert sum(x['GT_count'] for x in rows) == 350
        for row in rows:
            folder = ROOT / 'evidence' / group / row['scene']
            freeze = read(folder / 'predictions_freeze.json')
            assert freeze['status'] == 'PREDICTIONS_FROZEN_BEFORE_GT'
            assert freeze['GT_read_by_mapper'] is False
            for origin, digest in freeze['code_sha256'].items():
                assert origin.startswith(CODE_ROOT), origin
                assert sha(ROOT / 'sources' / origin[len(CODE_ROOT):]) == digest, origin
            assert read(folder / 'mapping_complete.json')['status'] == 'PASS'
            assert read(folder / 'evaluation_complete.json')['status'] == 'PASS'
            assert row['same_geometry_GT_scope_correspondence_and_scorer'] is True
            assert row['direct_total_votes_never_decrease_at_any_TSDF_point'] is True
            assert all(x['valid_to_zero_frame_points'] == 0 for x in row['mapping']['stages'].values())
    final = read(ROOT / 'evidence/final_complete.json')
    assert final['status'] == 'PASS' and final['main_runs'] == 16
    assert final['new_stage_metric_records'] == 96
    assert final['phase_matched_control_metric_records'] == 64
    assert final['same_R2_protocol'] and final['original_scorer_unchanged']
    assert sha(ROOT / 'reports/逐帧逐点替换对比.html') == final['report_sha256']
    tests = read(ROOT / 'evidence/tests_complete.json')
    assert tests['status'] == 'PASS' and tests['independent_set_oracle_cases'] == 504
    print(json.dumps({'status': 'PASS', 'files': len(manifest['files']), 'runs': 16,
                      'GT_count': 350, 'new_stage_metric_records': 96,
                      'direct_control_metric_records': 64, 'archived_kernel_cases': 504}))


if __name__ == '__main__':
    main()

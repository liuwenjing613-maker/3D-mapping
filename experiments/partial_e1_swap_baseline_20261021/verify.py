"""Verify the selected partial+E1+swap baseline without changing any result."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-sources', action='store_true',
                        help='Also hash deployed sources, seed masks, models and result files')
    args = parser.parse_args()
    release = read(HERE / 'release_manifest.json')
    for name, expected in release['files'].items():
        require(sha(HERE / name) == expected, 'Release file changed: ' + name)
    for name, expected in release['repository_files'].items():
        require(sha(ROOT / name) == expected, 'Registry/evaluation file changed: ' + name)
    config = read(HERE / 'config.json')
    registry = read(ROOT / 'instance_repair/configs/current_auto_pipeline.json')
    require(registry['pipeline'] == config['name'] and registry['frozen'] is True,
            'Current algorithm is not this frozen baseline')
    require(registry['config'] == str((HERE / 'config.json').relative_to(ROOT)).replace('\\', '/'),
            'Current configuration path changed')
    require(registry['default_repair'] == 'partial_plus_E1'
            and registry['result_variant'] == 'whole_to_partial', 'Default repair changed')
    require(config['source_commit'] == registry['source_commit']
            == '1103d8196a980874ef743f572d3b4c0748a4fdab', 'Upstream commit changed')
    require(config['frozen'] is True and config['evaluation']['frozen'] is False,
            'Algorithm/evaluation freeze distinction changed')
    protocol = read(ROOT / 'unified_eval/configs/current_protocol.json')
    require(protocol['frozen'] is False and protocol['profile_revision'] == 2
            and protocol['evaluable_GT_count'] == 350, 'Current R2 protocol changed')
    require(protocol['config_sha256'] == config['evaluation']['config_sha256'],
            'R2 configuration changed')
    for name, expected in protocol['evaluator_code_sha256'].items():
        require(sha(ROOT / 'unified_eval' / name) == expected, 'R2 scoring source changed: ' + name)
    from instance_repair.cli import verify_sources
    verify_sources(root=ROOT, frozen=ROOT / 'instance_repair/frozen_e1_20261012')
    cmd = [sys.executable, '-B', str(ROOT / 'experiments/auto_v3_swap/verify.py')]
    if args.server_sources:
        cmd.append('--server-sources')
    result = subprocess.run(cmd, cwd=ROOT, text=True, encoding='utf-8', capture_output=True)
    require(result.returncode == 0, result.stdout + result.stderr)
    upstream = json.loads(result.stdout)
    source = read(ROOT / 'experiments/auto_v3_swap/evidence/comparison.json')
    locked = read(HERE / 'result_lock.json')
    receipt = read(HERE / 'server_verification.json')
    require(receipt['status'] == 'PASS' and receipt['algorithm_frozen'] is True,
            'Server verification did not pass')
    require(receipt['scenes'] == locked['scenes'] and receipt['summary'] == locked['summary'],
            'Result lock differs from verified server artifacts')
    require(len(locked['scenes']) == 8 and sum(s['selected_seeds'] for s in locked['scenes']) == 196
            and sum(s['GT_count'] for s in locked['scenes']) == 350, 'Scene/seed/GT scope changed')
    require({s['scene']: s['selected_seeds'] for s in locked['scenes']} == config['counts'],
            'Scene selection changed')
    expected_summary = [r for r in source['summary'] if r['variant'] == 'new_partial']
    require(locked['summary'] == expected_summary, 'Results no longer match the archived comparison')
    for scene in locked['scenes']:
        expected = next(s for s in source['scenes'] if s['scene'] == scene['scene'])
        for stage, row in scene['stages'].items():
            old = expected['source_locks']['new_partial_' + stage]
            require(row['source'] == old['source'] and row['source_sha256'] == old['surface_sha256'],
                    'Wrong result selected: ' + scene['scene'] + '/' + stage)
            require(row['metrics'] == next(r for r in expected['rows']
                    if r['variant'] == 'new_partial' and r['stage'] == stage),
                    'Metric drift: ' + scene['scene'] + '/' + stage)
    for name, row in receipt['copies'].items():
        require(sha(HERE / name) == row['sha256'], 'Copied provenance changed: ' + name)
    if args.server_sources:
        for name, row in receipt['files'].items():
            require(sha(name) == row['sha256'], 'External artifact changed: ' + name)
    print(json.dumps({'status': 'PASS', 'baseline': config['name'],
                      'algorithm_frozen': True, 'evaluation_frozen': False,
                      'scenes': 8, 'selected_seeds': 196, 'evaluable_GT': 350,
                      'source_files': upstream['locked_files'],
                      'release_files': len(release['files']),
                      'external_files': len(receipt['files']),
                      'server_sources_checked': args.server_sources,
                      'summary': locked['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()

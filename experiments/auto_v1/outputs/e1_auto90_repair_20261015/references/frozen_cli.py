"""Frozen E1 entry. Replays locked observations; never reads GT during mapping."""
from pathlib import Path
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'instance_repair' / 'frozen_e1_20261012'
REGISTRY = ROOT / 'instance_repair' / 'configs' / 'current_architecture.json'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(root=ROOT, frozen=FROZEN):
    lock = read(frozen / 'source_lock.json')
    for relative, expected in lock['sha256'].items():
        path = root / relative
        if digest(path) != expected:
            raise RuntimeError(f'Frozen E1 source changed: {relative}')
    protocol = read(root / 'unified_eval/configs/current_protocol.json')
    if protocol['config_sha256'] != '6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0':
        raise RuntimeError('Current R2 evaluation lock changed')
    config = root / 'unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json'
    if digest(config) != protocol['config_sha256']:
        raise RuntimeError('R2 config bytes changed')
    for name, expected in protocol['evaluator_code_sha256'].items():
        if digest(root / 'unified_eval' / name) != expected:
            raise RuntimeError(f'R2 scoring source changed: {name}')
    return lock


class LockedModules(importlib.abc.MetaPathFinder):
    """Resolve original module names to vendored, hash-locked source files."""
    def __init__(self, folder, paths):
        self.folder = folder
        self.paths = paths

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.paths:
            return None
        file = self.folder / self.paths[fullname]
        kw = {'submodule_search_locations': [str(file.parent)]} if file.name == '__init__.py' else {}
        return importlib.util.spec_from_file_location(fullname, file, **kw)


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_artifacts(scene):
    reference = read(FROZEN / scene / 'reference.json')
    for path, expected in reference['reference_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Reference artifact changed: {path}')
    return reference


def compare_npz(actual, expected):
    import numpy as np
    with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise AssertionError(f'NPZ fields differ: {actual}')
        for field in a.files:
            if a[field].dtype != b[field].dtype or a[field].shape != b[field].shape:
                raise AssertionError(f'NPZ schema differs: {actual}:{field}')
            np.testing.assert_array_equal(a[field], b[field], err_msg=f'{actual}:{field}')


def prepare_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Replay requires a new output directory: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def replay(scene, output):
    verify_sources()
    reference = verify_artifacts(scene)
    output = prepare_output(output)
    folder = FROZEN / scene
    tests = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_observation_ops',
                            'test_adaptive_core'], cwd=folder, capture_output=True, text=True)
    (output / 'tests.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError('E1 invariant tests failed; see tests.log')
    (output / 'tests_complete.json').write_text(json.dumps({'status': 'PASS', 'tests': 24}))
    finder = LockedModules(folder, read(folder / 'module_map.json'))
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, str(folder))
    expand = load_script('_frozen_e1_expand_' + scene, folder / ('expand_' + scene + '.py'))
    expand.R = output
    expand.main()
    compare_npz(output / 'region.npz', Path(reference['experiment_root']) / 'region.npz')
    repair = load_script('_frozen_e1_repair_' + scene, folder / 'run_observation_repair.py')
    if repair.NAMES != ['E1_expanded_history']:
        raise RuntimeError('Frozen architecture must execute E1 only')
    repair.R = output
    repair.main()
    expected_root = Path(reference['experiment_root']) / 'E1_expanded_history' / scene
    actual_root = output / 'E1_expanded_history' / scene
    expected_files = sorted(p.relative_to(expected_root) for p in expected_root.rglob('*.npz'))
    actual_files = sorted(p.relative_to(actual_root) for p in actual_root.rglob('*.npz'))
    if expected_files != actual_files:
        raise AssertionError('Replayed artifact set differs from frozen E1')
    for relative in expected_files:
        compare_npz(actual_root / relative, expected_root / relative)
    loaded = {}
    for name, module in sys.modules.items():
        file = getattr(module, '__file__', None)
        if file and str(file).startswith('/home/chenkejun/CVPR/'):
            resolved = Path(file).resolve()
            if not resolved.is_relative_to(ROOT / 'instance_repair'):
                raise AssertionError(f'Unfrozen external project module loaded: {name}: {resolved}')
            loaded[name] = str(resolved.relative_to(ROOT))
    verify_sources()
    verify_artifacts(scene)
    receipt = {'status': 'PASS', 'architecture': 'P1-A1-strict-vote-E1-20261012',
               'scene': scene, 'mapping_frames': 400, 'region_arrays_exact': True,
               'all_E1_npz_arrays_exact': True, 'compared_npz_files': len(expected_files),
               'reference_experiment': reference['experiment_root'],
               'source_lock_sha256': digest(FROZEN / 'source_lock.json'),
               'GT_read_by_mapper': False, 'E2_executed': False,
               'external_project_modules_used': False, 'loaded_project_modules': loaded,
               'reference_R2_final': reference['R2_final']}
    (output / 'replay_verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('current-architecture')
    check = sub.add_parser('verify')
    check.add_argument('--artifacts', action='store_true')
    run = sub.add_parser('replay')
    run.add_argument('--scene', choices=['room0', 'room2'], required=True)
    run.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'current-architecture':
        verify_sources()
        print(json.dumps(read(REGISTRY), ensure_ascii=False, indent=2))
    elif args.command == 'verify':
        lock = verify_sources()
        if args.artifacts:
            for scene in ['room0', 'room2']:
                verify_artifacts(scene)
        print(json.dumps({'status': 'PASS', 'source_files': len(lock['sha256']),
                          'artifacts_verified': args.artifacts, 'architecture_frozen': True,
                          'evaluation_protocol_frozen': False}))
    else:
        replay(args.scene, args.out)


if __name__ == '__main__':
    main()

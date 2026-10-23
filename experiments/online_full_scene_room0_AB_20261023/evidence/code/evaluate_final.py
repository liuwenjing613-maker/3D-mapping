"""Run only main's locked R2 evaluator, on complete 400-frame surfaces."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time
import numpy as np

MAIN = Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
GROUP = sys.argv[1]
assert GROUP in ('A', 'B')
root = OUT / '400' / GROUP
dest = OUT / 'evaluation_r2' / GROUP
dest.mkdir(parents=True, exist_ok=True)

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run(stage, args):
    start = time.perf_counter()
    command = [sys.executable, '-B', '-m', 'unified_eval.cli', *map(str, args)]
    with (dest / (stage + '.log')).open('w') as log:
        subprocess.run(command, cwd=MAIN, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {'stage': stage, 'seconds': time.perf_counter() - start, 'command': command}

def main():
    mapping = read(root / 'mapping/mapping_complete.json')
    commit = read(root / 'repair/commit_complete.json')
    assert mapping['status'] == commit['status'] == 'PASS'
    assert mapping['frame_count'] == commit['processed_frames'] == 400
    frames = mapping['frame_ids']
    assert frames == list(range(0, 2000, 5)) and commit['last_raw_frame'] == 1995
    protocol = read(MAIN / 'unified_eval/configs/current_protocol.json')
    assert protocol['profile_revision'] == 2 and protocol['evaluable_GT_count'] == 350
    assert protocol['evaluation_profile'] == 'object_observed_repair'
    assert protocol['scoring_commit'] == 'cd260569e89de1e9525f82c45ee10842b80616b0'
    assert protocol['audit_archive_commit'] == 'd035f68770cefab27c9751e97062c72faa07367c'
    config = MAIN / 'unified_eval/configs' / protocol['config_file']
    assert sha(config) == protocol['config_sha256']
    for name, expected in protocol['evaluator_code_sha256'].items():
        assert sha(MAIN / 'unified_eval' / name) == expected
    gt = Path(protocol['canonical_GT']['room0']['file'])
    assert sha(gt) == protocol['canonical_GT']['room0']['file_sha256']
    before = root / 'mapping/baseline/final/instance_surface.npz'
    after = Path(commit['final_map'])
    assert sha(after) == commit['final_map_sha256']
    with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
        assert a['xyz_m'].dtype == b['xyz_m'].dtype and np.array_equal(a['xyz_m'], b['xyz_m'])
        geometry = hashlib.sha256(a['xyz_m'].tobytes()).hexdigest()
        point_count = len(a['xyz_m'])
        changed = int(np.count_nonzero(a['instance_id'] != b['instance_id']))
    provenance = {'input_scope_verified': True,
                  'frame_list_sha256': hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest(),
                  'frame_count': 400, 'frame_ids': frames, 'scene': 'room0', 'branch': GROUP,
                  'mapping_complete': str(root / 'mapping/mapping_complete.json'),
                  'mapping_complete_sha256': sha(root / 'mapping/mapping_complete.json'),
                  'commit_complete': str(root / 'repair/commit_complete.json'),
                  'commit_complete_sha256': sha(root / 'repair/commit_complete.json'),
                  'numerical_source_commit': '4e7d3c62c0c6876c1e6c08e988f85d6d53e7604c',
                  'adapter_audit': str(OUT / 'adapter_audit.json'), 'adapter_audit_sha256': sha(OUT / 'adapter_audit.json'),
                  'scope': 'Same final400 geometry before versus after checkpoint400 repair; earlier committed repairs already present.'}
    source = dest / 'source_provenance.json'
    source.write_text(json.dumps(provenance, indent=2) + '\n')
    steps = []
    for stage, surface in [('before', before), ('after', after)]:
        steps.append(run(stage, ['adapt-surface', '--gt', gt, '--surface', surface,
                     '--fixed-correspondence', dest / 'fixed_correspondence.npz', '--source-provenance', source,
                     '--method-name', f'online_checkpoint_room0_{GROUP}_400_{stage}',
                     '--method-commit', provenance['numerical_source_commit'], '--out', dest / stage]))
    steps.append(run('pair', ['eval-repair-pair', '--gt', gt,
                 '--before', dest / 'before/canonical_prediction.npz', '--after', dest / 'after/canonical_prediction.npz',
                 '--out', dest / 'pair']))
    result = {'status': 'PASS', 'branch': GROUP, 'checkpoint': 400,
              'evaluator_worktree': str(MAIN), 'evaluation_profile': protocol['evaluation_profile'],
              'profile_revision': 2, 'frozen': protocol['frozen'], 'scene_evaluable_GT': 72,
              'identical_geometry_verified': True, 'surface_points': point_count,
              'surface_label_changes_at400': changed, 'xyz_sha256': geometry,
              'before_surface': str(before), 'before_surface_sha256': sha(before),
              'after_surface': str(after), 'after_surface_sha256': sha(after), 'timings': steps}
    (dest / 'evaluation_complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()

"""Run full checkpoint continuations sequentially with immutable input commits."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

REPO = Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
CODE = REPO / 'experiments/online_room0_repair50_to100_20261022/continuation300'
HERE = Path(__file__).resolve().parent
OUT = Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
PY = '/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
BASE = '20cb6a164d9aba465e020bc026e086357b88ceea'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_reference():
    ref = HERE / 'reference'
    ref.mkdir(exist_ok=True)
    original = subprocess.check_output(['git', '-C', str(REPO), 'show',
        BASE + ':experiments/online_room0_repair50_to100_20261022/continuation300/map_continue.py'])
    (ref / 'map_continue.py').write_bytes(original)
    # The sole shared adapter redirects output; reference algorithm is byte-identical to git.
    (ref / 'map_support.py').write_bytes((CODE / 'map_support.py').read_bytes())
    old_ast = ast.parse(original)
    new_ast = ast.parse((CODE / 'map_continue.py').read_text(encoding='utf-8-sig'))
    old_functions = {n.name: ast.dump(n) for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n) for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    unchanged = [name for name in old_functions if name != 'main']
    assert all(old_functions[name] == new_functions[name] for name in unchanged)
    result = dict(status='PASS', base_commit=BASE, unchanged_functions=unchanged,
        original_map_sha256=digest(ref/'map_continue.py'), modified_map_sha256=digest(CODE/'map_continue.py'),
        shared_support_sha256=digest(CODE/'map_support.py'),
        guard_sha256=digest(CODE/'map_guard/sitecustomize.py'))
    write(OUT/'source_audit.json', result)
    return ref, result

def run(name, count, group, code, policy):
    root = OUT/name
    assert not root.exists(), 'Validation outputs must be fresh: ' + str(root)
    root.mkdir(parents=True)
    env = dict(os.environ, CVPR_CHECKPOINT=str(count), CVPR_TRACK_GROUP=group,
        CVPR_MAPPING_RUN_ROOT=str(root), CVPR_SURFACE_POLICY=policy,
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    env.pop('CVPR_ONLINE_GUARD', None)
    env.pop('PYTHONPATH', None)
    setup = subprocess.run([PY, '-B', str(code/'map_support.py')], env=env, capture_output=True, text=True)
    assert setup.returncode == 0, setup.stdout + setup.stderr
    env.update(PYTHONPATH=str(CODE/'map_guard'), CVPR_ONLINE_GUARD=str(root/'mapping/input_guard.json'))
    manifest = dict(status='RUNNING', name=name, count=count, group=group, policy=policy,
        code=str(code/'map_continue.py'), code_sha256=digest(code/'map_continue.py'),
        prior_commit=json.loads((root/'mapping/prior_commit.json').read_text()),
        threads=dict(OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4), started_unix=time.time())
    started = time.perf_counter()
    with (root/'execution.log').open('wb') as log:
        proc = subprocess.Popen([PY, '-B', '-u', str(code/'map_continue.py')],
            cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest['pid'] = proc.pid
        write(root/'launch.json', manifest)
        write(OUT/'validation_status.json', dict(status='RUNNING', current=name, pid=proc.pid))
        ret = proc.wait()
    manifest.update(returncode=ret, ended_unix=time.time(), process_wall_seconds=time.perf_counter()-started)
    manifest['status'] = 'PASS' if ret == 0 else 'FAILED'
    write(root/'execution.json', manifest)
    assert ret == 0, str(root/'execution.log')
    assert digest(code/'map_continue.py') == manifest['code_sha256']
    return manifest

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref, source = prepare_reference()
    completed = []
    for name, count, group, code, policy in [
        ('checkpoint_200_A', 200, 'A', CODE, 'checkpoint'),
        ('reference_200_A', 200, 'A', ref, 'every_frame'),
        ('checkpoint_300_A', 300, 'A', CODE, 'checkpoint'),
        ('checkpoint_200_B', 200, 'B', CODE, 'checkpoint'),
        ('checkpoint_300_B', 300, 'B', CODE, 'checkpoint'),
    ]:
        completed.append(run(name, count, group, code, policy))
    write(OUT/'validation_status.json', dict(status='RUNS_COMPLETE', completed=completed, source_audit=source))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(OUT/'validation_status.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise

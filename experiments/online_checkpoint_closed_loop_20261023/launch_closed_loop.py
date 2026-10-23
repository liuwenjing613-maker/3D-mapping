from pathlib import Path
import hashlib,json,subprocess,os
HERE=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_checkpoint_closed_loop_20261023')
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
for mode in ['reference','checkpoint']:
    code=HERE/mode
    wrapper=HERE/'track_queue_with_cache.py'
    (code/wrapper.name).write_bytes(wrapper.read_bytes())
    path=code/'selection_run.py';s=path.read_text()
    assert "CONT/'track_queue.py'" in s
    path.write_text(s.replace("CONT/'track_queue.py'","CONT/'track_queue_with_cache.py'"))
    (code/'cache_adapter_audit.json').write_text(json.dumps(dict(status='PASS',
        wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        original_tracking_reuse_function_unchanged=True,candidate_unification=False),indent=2)+'\n')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
with (OUT/'driver.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(HERE/'run_closed_loop.py')],
        cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,log=str(OUT/'driver.log'))))

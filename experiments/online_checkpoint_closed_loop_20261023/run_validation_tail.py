"""Complete frame-201 inheritance, real-output comparisons, and a separate common-surface control."""
from pathlib import Path
import json,os,subprocess,time,traceback,threading
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
MAP_CODE=HERE.parents[1]/'experiments/online_room0_repair50_to100_20261022/continuation300'
STATUS_LOCK=threading.Lock()
def read(p):return json.loads(p.read_text())
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def command(name,script,args=(),guard=None,code=None):
    env={**os.environ,'CVPR_CHECKPOINT':'200','CVPR_TRACK_GROUP':'A','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
    if guard:env.update(CVPR_ONLINE_GUARD=str(guard),PYTHONPATH=str(code))
    start=time.perf_counter()
    with (OUT/(name+'.log')).open('ab',buffering=0) as log:
        proc=subprocess.Popen([PY,'-B','-u',str(script),*map(str,args)],cwd=script.parent,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        with STATUS_LOCK:write(OUT/'validation_tail_status.json',dict(status='RUNNING',stage=name,pid=proc.pid))
        ret=proc.wait()
    write(OUT/(name+'_execution.json'),dict(returncode=ret,seconds=time.perf_counter()-start))
    assert ret==0,(name,ret)
def main():
    for mode in ['reference','checkpoint']:
        root=OUT/mode
        while not (root/'repair/commit_complete.json').exists():
            assert read(OUT/'closed_loop_status.json')['status']!='FAILED'
            write(OUT/'validation_tail_status.json',dict(status='WAITING_COMMIT',mode=mode))
            time.sleep(10)
        assert read(root/'repair/commit_complete.json')['status']=='PASS'
        if not (root/'frame201/mapping/mapping_complete.json').exists():
            if not (root/'frame201/mapping/mapping_plan.json').exists():
                command(mode+'_frame201_setup',HERE/'run_frame201.py',[mode,'setup'])
            command(mode+'_frame201',HERE/'run_frame201.py',[mode,'run'],
                root/'frame201/mapping/input_guard.json',MAP_CODE/'map_guard')
        assert read(root/'frame201/mapping/mapping_complete.json')['status']=='PASS'
    command('compare_tracking',HERE/'compare_tracking.py')
    command('compare_repair',HERE/'compare_repair.py')
    if not (OUT/'fixed_control_setup.json').exists():command('fixed_setup',HERE/'prepare_fixed_control.py')
    def fixed_run(mode):
        code=HERE/mode;root=OUT/mode;guard=root/'input_guard.json'
        for stage,script,marker in [('fence','selection_fence.py','input_fence_audit.json'),
                ('prepare','track_prepare.py','tracking/prepare_complete.json'),
                ('tracking','track_queue_with_cache.py','tracking/cache_fingerprint_audit.json'),
                ('gate','track_gate.py','tracking/gate_complete.json'),
                ('repair','repair_continue.py','repair/repair_complete.json'),
                ('commit','commit_continue.py','repair/commit_complete.json')]:
            if not (root/marker).exists():
                command(mode+'_'+stage,code/script,guard=guard,code=str(code/'selection_guard')+':'+str(code))
            assert read(root/marker)['status']=='PASS'
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in [pool.submit(fixed_run,mode) for mode in ['fixed_reference','fixed_checkpoint']]:f.result()
    command('fixed_compare_tracking',HERE/'compare_tracking.py',['--left','fixed_reference','--right','fixed_checkpoint','--name','fixed_tracking_comparison'])
    command('fixed_compare_repair',HERE/'compare_repair.py',['--left','fixed_reference','--right','fixed_checkpoint','--name','fixed_repair_comparison','--fixed'])
    write(OUT/'validation_tail_status.json',dict(status='COMPLETE'))
if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'validation_tail_status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

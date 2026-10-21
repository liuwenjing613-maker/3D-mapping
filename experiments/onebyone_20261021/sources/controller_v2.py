"""Resume existing mappings; use corrected main-R2 output filename adapter."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,time,traceback
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def save(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def live(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.exists() and bool(p.read_bytes())
def run(group,scene):
    out=R/group/scene;env={**os.environ,'POINT_GROUP':group,'POINT_SCENE':scene,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    if not (out/'mapping_complete.json').exists() and (out/'active_process.json').exists():
        active=json.loads((out/'active_process.json').read_text())
        if active['script']=='replay.py':
            while live(active['pid']) and not (out/'mapping_complete.json').exists():
                assert not (out/'mapping_failure.json').exists(),(group,scene,'mapping failure')
                time.sleep(10)
            assert (out/'mapping_complete.json').exists(),(group,scene,'existing replay exited')
    for script,complete,logname in [('replay.py','mapping_complete.json','mapping.log'),('evaluate_v2.py','evaluation_complete.json','evaluation_v2.log')]:
        if (out/complete).exists():assert json.loads((out/complete).read_text())['status']=='PASS';continue
        assert not (out/'mapping_failure.json').exists(),(group,scene,'mapping failure')
        with (out/logname).open('w') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',str(C/script)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(out/'active_process.json',dict(pid=p.pid,script=script,unix=time.time()));rc=p.wait()
        assert rc==0,(group,scene,script,rc)
    if (out/'evaluation_failure.json').exists():
        save(out/'evaluation_recovery.json',dict(status='PASS',reason='Corrected result filename to paired_repair_metrics.json; reused verified mapped surfaces and scores',old_failure=str(out/'evaluation_failure.json')))
    return dict(group=group,scene=scene,status='PASS')
def main():
    tasks=[('automatic','room0'),('manual','office0')]+[(g,s) for s in SCENES for g in ['automatic','manual'] if (g,s) not in [('automatic','room0'),('manual','office0')]]
    completed=[];failures=[]
    save(R/'controller_status.json',dict(stage='running',version=2,total=16,completed=[],unix=time.time()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(run,g,s):(g,s) for g,s in tasks}
        for f in as_completed(futures):
            try:completed.append(f.result());print('DONE',completed[-1],flush=True)
            except Exception:failures.append(dict(task=futures[f],traceback=traceback.format_exc()))
            save(R/'controller_status.json',dict(stage='running',version=2,completed=completed,failures=failures,total=16,unix=time.time()))
    assert not failures,failures
    save(R/'all_runs_complete.json',dict(status='PASS',completed=completed,unix=time.time()))
    p=subprocess.run([sys.executable,'-B','-u',str(C/'collect.py')],cwd=C);assert p.returncode==0
    save(R/'controller_status.json',dict(stage='complete',status='PASS',version=2,completed=completed,unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:save(R/'controller_v2_failure.json',dict(traceback=traceback.format_exc()));raise

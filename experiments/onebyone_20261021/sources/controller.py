from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,time,traceback
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4'];PILOTS=[('automatic','room0'),('manual','office0')]
def save(p,v):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def run(group,scene):
    out=R/group/scene;env={**os.environ,'POINT_GROUP':group,'POINT_SCENE':scene,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    for script,complete,logname in [('replay.py','mapping_complete.json','mapping.log'),('evaluate.py','evaluation_complete.json','evaluation.log')]:
        if (out/complete).exists():
            assert json.loads((out/complete).read_text())['status']=='PASS';continue
        with (out/logname).open('w') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',str(C/script)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(out/'active_process.json',dict(pid=p.pid,script=script,unix=time.time()))
            rc=p.wait()
        assert rc==0,(group,scene,script,rc)
    return dict(group=group,scene=scene,status='PASS')
def main():
    save(R/'controller_status.json',dict(stage='waiting_for_two_pilots',unix=time.time()))
    while True:
        for group,scene in PILOTS:
            out=R/group/scene
            assert not (out/'mapping_failure.json').exists(),str(out/'mapping_failure.json')
            if not (out/'mapping_complete.json').exists():
                pid=json.loads((out/'launch.json').read_text())['pid'];assert Path(f'/proc/{pid}').exists(),(group,scene,'pilot exited without completion')
        if all((R/g/s/'mapping_complete.json').exists() for g,s in PILOTS):break
        time.sleep(10)
    tasks=PILOTS+[(g,s) for s in SCENES for g in ['automatic','manual'] if (g,s) not in PILOTS]
    complete=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(run,g,s) for g,s in tasks]
        for f in as_completed(futures):
            complete.append(f.result());save(R/'controller_status.json',dict(stage='running',completed=complete,total=16,unix=time.time()))
            print('DONE',complete[-1],flush=True)
    save(R/'all_runs_complete.json',dict(status='PASS',completed=complete,unix=time.time()))
    p=subprocess.run([sys.executable,'-B','-u',str(C/'collect.py')],cwd=C);assert p.returncode==0
    save(R/'controller_status.json',dict(stage='complete',status='PASS',completed=complete,unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:save(R/'controller_failure.json',dict(traceback=traceback.format_exc()));raise

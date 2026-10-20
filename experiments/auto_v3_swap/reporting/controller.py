from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, subprocess, sys, time, traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
def save(n,v):
    p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(scene):
    with (R/scene/'controller.log').open('x') as f:
        p=subprocess.Popen([sys.executable,'-B','-u',str(C/scene/'controller.py')],cwd=C/scene,
            env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'},
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(scene+'/launch.json',dict(pid=p.pid,unix=time.time()))
        code=p.wait()
        return dict(scene=scene,returncode=code,status='PASS' if code==0 else 'FAIL')
def main():
    with (R/'selection.log').open('x') as f:
        subprocess.run([sys.executable,'-B','-u',str(C/'select_all.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    save('status.json',dict(status='RUNNING',stage='E1_and_partial',scenes=['office1','office3','office4']))
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,s) for s in ['office1','office3','office4']]):
            results.append(future.result());save('scene_progress.json',results)
    assert all(r['status']=='PASS' for r in results),results
    save('complete.json',dict(status='PASS',scenes=8,rerun=results,unix=time.time()))
    save('status.json',dict(status='PASS',stage='complete',unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:
        save('failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

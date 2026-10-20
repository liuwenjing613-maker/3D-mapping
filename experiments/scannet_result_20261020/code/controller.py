"""Sequential scene execution with restartable, independently audited stages."""
from pathlib import Path
import os,sys,json,subprocess,time,traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
SCENES=['scene0011_00','scene0608_00']
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def stage(scene,name,args,cwd=None):
    folder=R/scene;folder.mkdir(exist_ok=True);receipt=folder/(name+'_stage_complete.json')
    if receipt.exists():return
    d=dict(status='RUNNING',scene=scene,stage=name,time=time.time())
    save(R/'status.json',d);save(folder/'status.json',d)
    log=folder/(name+'.controller.log')
    with log.open('ab') as f:
        env={**os.environ,'PYTHONPATH':str(C/'shared'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
        q=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=cwd or C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(folder/'active_process.json',dict(pid=q.pid,stage=name))
        assert q.wait()==0,(scene,name,str(log))
    save(receipt,dict(status='PASS',time=time.time()))
def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode=='all' and (R/'baseline_launch.json').exists():
        while not (R/'baseline_complete.json').exists():
            pid=json.loads((R/'baseline_launch.json').read_text())['pid']
            os.kill(pid,0)
            assert not (R/'failure.json').exists(),(R/'failure.json').read_text() if (R/'failure.json').exists() else ''
            time.sleep(15)
    for scene in SCENES:
        for sub in ['frontend','overmerge','intersection','versions','semantic_route','grouping_r2','grouping_r3','v1','v2','runtime_cache','cuda_cache','e1_repair','partial']:
            (R/scene/sub).mkdir(parents=True,exist_ok=True)
        stage(scene,'prepare',[C/'prepare.py',scene])
        stage(scene,'baseline',[C/'baseline.py',scene])
        if mode=='baseline':continue
        stage(scene,'metadata',[C/'metadata.py',scene])
        stage(scene,'baseline_evaluation',[C/'evaluate.py',scene])
        stage(scene,'catalog',[C/'catalog_code/build_candidates.py',scene])
        stage(scene,'upstream',[C/scene/'run_v1.py'],C/scene)
        stage(scene,'E1',[C/scene/'run_remaining.py'],C/scene)
        stage(scene,'partial',[C/scene/'partial/controller.py'],C/scene/'partial')
        save(R/scene/'status.json',dict(status='PASS',stage='all_complete'))
    save(R/(mode+'_complete.json'),dict(status='PASS',scenes=SCENES,time=time.time()))
    if mode=='all':
        save(R/'complete.json',dict(status='PASS',scenes=SCENES,time=time.time()))
    save(R/'status.json',dict(status='PASS',stage=mode+'_complete'))
if __name__=='__main__':
    try:main()
    except Exception:save(R/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

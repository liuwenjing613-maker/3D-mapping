"""Eight-scene online-B trial with checked shared-prefix reuse and resource admission."""
from pathlib import Path
import fcntl,json,os,subprocess,time,traceback
import psutil
C=Path('/home/chenkejun/CVPR/experiments/trusted_full_online_20261104')
O=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
EVALPY='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
SCENES=['room0','room2','room1','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def passed(s):
    p=O/'tasks/online_B'/s/'status.json'
    return p.exists() and read(p).get('status')=='PASS'
def pilot():
    p=O/'online/room0/100/B/repair/commit_complete.json'
    q=O/'online/room2/100/B/repair/commit_complete.json'
    mapping=[O/'online'/s/'100/B/mapping_smoke_status.json' for s in ['room0','room2']]
    return any(x.exists() and read(x).get('status')=='PASS' for x in [p,q]) or all(x.exists() and read(x).get('status')=='PASS' for x in mapping)
def main():
    O.mkdir(parents=True,exist_ok=True)
    with (O/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[s for s in SCENES if not passed(s)];active={};done=[s for s in SCENES if passed(s)];failures=[]
        for s in pending:
            status=O/'tasks/online_B'/s/'status.json'
            if status.exists():
                row=read(status)
                if row.get('status')=='RUNNING' and psutil.pid_exists(row.get('pid',-1)):
                    raise RuntimeError('Existing live scene driver; do not duplicate: '+s)
        while pending or active:
            for s,p in list(active.items()):
                rc=p.poll()
                if rc is None:continue
                del active[s]
                if rc or not passed(s):failures.append(dict(scene=s,returncode=rc))
                else:done.append(s)
            available=psutil.virtual_memory().available/1024**3
            max_scenes=4 if pilot() else 2
            eligible=[]
            for scene in pending:
                smoke=O/'online'/scene/'100/B/mapping_smoke_status.json'
                state=read(smoke).get('status') if smoke.exists() else None
                if state=='FAILED':
                    failures.append(dict(scene=scene,stage='mapping_pilot'));break
                if state!='RUNNING' and (pilot() or scene in ['room0','room2']):eligible.append(scene)
            if not failures and eligible and len(active)<max_scenes and available>=80:
                s=eligible[0];pending.remove(s)
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CVPR_GPU_POOL':'0,1,2,3,4,5,6,7'}
                for k in ['CVPR_ONLINE_GUARD','PYTHONPATH']:env.pop(k,None)
                log=O/'run_logs'/f'online_B_{s}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as f:
                    p=subprocess.Popen([PY,'-B','-u',str(C/'run_one.py'),'online_B',s],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active[s]=p;print('DISPATCH',s,p.pid,flush=True)
            write(O/'status.json',dict(status='FAILED_STOP_NEW_DISPATCH' if failures else 'RUNNING',running=[dict(scene=s,pid=p.pid) for s,p in active.items()],completed=done,pending=pending,failures=failures,total=8,max_scene_parallelism=max_scenes,available_ram_gib=available,unix=time.time()))
            if failures and not active:raise RuntimeError(failures)
            if pending or active:time.sleep(10)
        write(O/'status.json',dict(status='EVALUATING',completed=done,total=8,unix=time.time()))
        env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
        with (O/'comparison.log').open('a') as f:
            subprocess.run([EVALPY,'-B','-u',str(C/'compare_results.py')],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        if (C/'analyze_mechanisms.py').exists():
            with (O/'mechanism_audit.log').open('a') as f:subprocess.run([EVALPY,'-B',str(C/'analyze_mechanisms.py')],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        write(O/'status.json',dict(status='PASS',completed=done,total=8,comparison=str(O/'comparison/comparison.json'),unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:
        write(O/'coordinator_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()))
        raise

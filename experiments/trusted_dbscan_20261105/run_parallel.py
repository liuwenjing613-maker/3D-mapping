"""Run exact-control replay first, then trusted maps, then fixed-main comparison."""
from pathlib import Path
import os,sys,json,time,subprocess,concurrent.futures,traceback,fcntl
import psutil
C=Path('/home/chenkejun/CVPR/experiments/trusted_dbscan_20261105')
O=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def worker(mode,s):
 dest=O/mode/s;dest.mkdir(parents=True,exist_ok=True)
 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
 env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
 with (dest/'run.log').open('a') as log:
  subprocess.run([sys.executable,'-B','-u',str(C/'run_scene.py'),mode,s],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=read(dest/'final_scene_complete.json');assert d['status']=='PASS';return s
def main():
 O.mkdir(parents=True,exist_ok=True)
 with (O/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert psutil.virtual_memory().available/2**30>60
  for mode in ['control','candidate']:
   done=[]
   dump(O/'status.json',dict(status='RUNNING',stage=mode,completed=done,total=8,max_parallel=4,unix=time.time()))
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    fs=[pool.submit(worker,mode,s) for s in SCENES]
    for f in concurrent.futures.as_completed(fs):
     done.append(f.result());dump(O/'status.json',dict(status='RUNNING',stage=mode,completed=done,total=8,max_parallel=4,unix=time.time()))
   dump(O/(mode+'_complete.json'),dict(status='PASS',scenes=sorted(done),unix=time.time()))
  dump(O/'status.json',dict(status='EVALUATING',stage='four_way_comparison',completed=SCENES,total=8,unix=time.time()))
  assert (C/'compare_results.py').exists()
  with (O/'comparison.log').open('a') as log:
   subprocess.run(['/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python','-B','-u',str(C/'compare_results.py')],cwd=C,stdout=log,stderr=subprocess.STDOUT,check=True)
  dump(O/'status.json',dict(status='PASS',stage='complete',completed=SCENES,total=8,comparison=str(O/'comparison/comparison.json'),unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(O/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise

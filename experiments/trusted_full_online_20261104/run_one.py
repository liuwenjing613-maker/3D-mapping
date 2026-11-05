"""Whole-scene driver: producer first, then one final B and R2 evaluation."""
from alignment_common import *
import sys,subprocess,traceback,fcntl
variant,scene=sys.argv[1:];dest=O/'tasks'/variant/scene;dest.mkdir(parents=True,exist_ok=True)
def main():
 with (dest/'task.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  frozen=read(C/'source_lock.json')
  # Immutable policy + adapter files. Generated case/config files are excluded from the source lock.
  for p,h in frozen['sha256'].items():assert sha(p)==h,p
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CVPR_GPU_POOL':os.environ.get('CVPR_GPU_POOL','0,1,2,3,4,5,6,7')}
  env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
  dump(dest/'status.json',dict(status='RUNNING',variant=variant,scene=scene,pid=os.getpid(),stage='producer',unix=time.time()))
  py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
  if variant!='no_repair':
   method,group=('main','A') if variant=='main' else ('online',variant[-1])
   branch=C/method/scene/'run_branch.py'
   subprocess.run([py,'-B','-u',str(branch),group],cwd=branch.parent,env=env,check=True)
  dump(dest/'status.json',dict(status='RUNNING',variant=variant,scene=scene,pid=os.getpid(),stage='final_B_and_evaluation',unix=time.time()))
  subprocess.run([py,'-B','-u',str(C/'finish_scene.py'),variant,scene],cwd=C,env=env,check=True)
  dump(dest/'status.json',dict(status='PASS',variant=variant,scene=scene,unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:dump(dest/'failure.json',dict(status='FAILED',variant=variant,scene=scene,traceback=traceback.format_exc(),unix=time.time()));raise

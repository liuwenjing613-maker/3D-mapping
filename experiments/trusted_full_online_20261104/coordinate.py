"""Persistent priority scheduler with pilot gates and per-scene failure isolation."""
from alignment_common import *
import sys,subprocess,fcntl,traceback
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
VARIANTS=['no_repair','main','online_A','online_B']
def passed(v,s):
 p=O/'tasks'/v/s/'status.json'
 return p.exists() and read(p).get('status')=='PASS'
def failed(v,s):return (O/'tasks'/v/s/'failure.json').exists()
def pilot_online():
 for g in ['A','B']:
  p=O/'online/room1/50'/g/'repair/commit_complete.json'
  q=O/'online/room1/100'/g/'mapping/frame51_state_audit.json'
  if not p.exists() or not q.exists():return False
  assert read(p)['status']=='PASS'
  assert read(q)['status']=='PASS' and read(q)['repaired_state_used']
 return True
class Adopted:
 def __init__(self,pid,variant,scene):
  self.pid=pid;self.variant=variant;self.scene=scene
  self.cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
 def poll(self):
  try:
   p=Path('/proc')/str(self.pid)
   if (p/'cmdline').read_bytes()==self.cmd and (p/'stat').read_text().split(') ',1)[1].split()[0]!='Z':return None
  except (OSError,ProcessLookupError):pass
  return 0 if passed(self.variant,self.scene) else 1

def main():
 with (O/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=[(v,s) for s in ['room1']+[x for x in SCENES if x!='room1'] for v in VARIANTS]
  active={};done=[];failures=[]
  for job in list(jobs):
   v,scene=job;status=O/'tasks'/v/scene/'status.json'
   if not status.exists() or passed(*job) or failed(*job):continue
   state=read(status);pid=state.get('pid')
   try:
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if str(C/'run_one.py').encode() not in cmd or v.encode() not in cmd or scene.encode() not in cmd:continue
    active[job]=Adopted(pid,v,scene);jobs.remove(job)
    print('ADOPT',v,scene,pid,flush=True)
   except (OSError,ProcessLookupError):pass
  while jobs or active:
   for job,proc in list(active.items()):
    rc=proc.poll()
    if rc is None:continue
    del active[job]
    if rc or not passed(*job):failures.append(dict(variant=job[0],scene=job[1],returncode=rc))
    else:done.append(job)
   for job in list(jobs):
    if passed(*job):jobs.remove(job);done.append(job);continue
    if failed(*job):jobs.remove(job);failures.append(dict(variant=job[0],scene=job[1],existing_failure=True));continue
    v,s=job
    if failures:break
    if v=='no_repair':
     if sum(j[0]=='no_repair' for j in active)>=2:continue
    else:
     if sum(j[0]!='no_repair' for j in active)>=6:continue
     if s!='room1' and (not passed('main','room1') if v=='main' else not pilot_online()):continue
    log=O/'run_logs'/(v+'_'+s+'.log');log.parent.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CVPR_GPU_POOL':'0,1,2,3,4,5,6,7'}
    with log.open('a') as f:
     proc=subprocess.Popen([PY,'-B','-u',str(C/'run_one.py'),v,s],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    active[job]=proc;jobs.remove(job);print('DISPATCH',v,s,proc.pid,flush=True)
   dump(O/'status.json',dict(status='FAILED_STOP_NEW_DISPATCH' if failures else 'RUNNING',running=[dict(variant=v,scene=s,pid=p.pid) for (v,s),p in active.items()],completed=[list(x) for x in done],pending=[list(x) for x in jobs],failures=failures,total=32,main_pilot_passed=passed('main','room1'),online_pilots_passed=pilot_online(),unix=time.time()))
   if failures and not active:raise RuntimeError(failures)
   time.sleep(10)
  subprocess.run([PY,'-B','-u',str(C/'report_results.py')],cwd=C,check=True)
  dump(O/'status.json',dict(status='PASS',total=32,completed=done,unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'coordinator_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

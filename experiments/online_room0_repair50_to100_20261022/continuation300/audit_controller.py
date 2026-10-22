"""Validate each completed branch checkpoint exactly once, without modifying maps."""
from pathlib import Path
import os,json,time,subprocess
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def ready(p):return p.exists() and read(p).get('status')=='PASS'
def dump(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
running={};done={};failed={}
while len(done)<4:
 for n in (200,300):
  for g in ('A','B'):
   k=f'{n}{g}';root=T/str(n)/g
   if k in running or k in done or k in failed:continue
   if ready(root/'final_audit.json'):done[k]={'preexisting':True};continue
   if not ready(root/'repair/commit_complete.json'):continue
   env=os.environ.copy();env.update(CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=g,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
   env.pop('CVPR_ONLINE_GUARD',None)
   log=(root/'final_audit.log').open('ab',buffering=0)
   proc=subprocess.Popen([PY,'-B',str(C/'final_audit.py')],cwd=str(C),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   running[k]=(proc,time.perf_counter(),log,root)
 for k,(proc,t,log,root) in list(running.items()):
  rc=proc.poll()
  if rc is None:continue
  log.close();row=dict(status='PASS' if rc==0 and ready(root/'final_audit.json') else 'FAILED',exit_code=rc,seconds=time.perf_counter()-t)
  (done if row['status']=='PASS' else failed)[k]=row;del running[k]
 dump(T/'audit_controller_status.json',dict(status='PASS' if len(done)==4 else ('NEEDS_ATTENTION' if failed else 'RUNNING'),completed=done,failed=failed,running={k:v[0].pid for k,v in running.items()},unix=time.time()))
 if len(done)<4:time.sleep(5)

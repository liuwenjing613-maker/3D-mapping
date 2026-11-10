"""Run only ready repair stages; each branch remains paused until its separate commit."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
CONT=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def ready(p):return p.exists() and read(p).get('status')=='PASS'
def main():
 TOP.mkdir(parents=True,exist_ok=True);running={};finished={};failed={}
 for n in (200,300):
  for g in ('A','B'):
   root=TOP/str(n)/g
   if ready(root/'repair/repair_complete.json'):finished[f'{n}{g}']={'preexisting_complete':True}
 while len(finished)<4:
  for n in (200,300):
   for g in ('A','B'):
    key=f'{n}{g}';root=TOP/str(n)/g
    if key in running or key in finished or key in failed:continue
    if not ready(root/'tracking/gate_complete.json'):continue
    assert ready(root/'mapping/mapping_complete.json') and ready(root/'tracking/tracking_queue_complete.json')
    guard=root/'input_guard.json';assert read(guard)['raw_cutoff']==(n-1)*5
    destination=root/'repair';destination.mkdir(exist_ok=True)
    assert not (destination/'repair_failure.json').exists(),'Explicit review required after failed repair attempt'
    env=os.environ.copy();env.update(CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=g,CVPR_ONLINE_GUARD=str(guard),
     PYTHONPATH=str(CONT/'selection_guard')+':'+str(CONT),
     PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    log=(destination/'execution.log').open('ab',buffering=0)
    proc=subprocess.Popen([PY,'-B',str(CONT/'repair_continue.py')],cwd=str(CONT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    running[key]=(proc,time.perf_counter(),time.time(),log,n,g)
    dump(destination/'process.json',dict(status='RUNNING',pid=proc.pid,checkpoint=n,group=g,started_unix=time.time()))
  for key,(proc,t0,u0,log,n,g) in list(running.items()):
   rc=proc.poll()
   if rc is None:continue
   log.close();root=TOP/str(n)/g
   row=dict(status='PASS' if rc==0 and ready(root/'repair/repair_complete.json') else 'FAILED',
    exit_code=rc,seconds=time.perf_counter()-t0,started_unix=u0,ended_unix=time.time(),pid=proc.pid,checkpoint=n,group=g)
   dump(root/'repair/execution_timing.json',row)
   (finished if row['status']=='PASS' else failed)[key]=row;del running[key]
  dump(TOP/'repair_controller_status.json',dict(status='PASS' if len(finished)==4 else ('NEEDS_ATTENTION' if failed else 'RUNNING'),
    completed=finished,failed=failed,running={k:dict(pid=v[0].pid,seconds=time.perf_counter()-v[1]) for k,v in running.items()},unix=time.time()))
  if len(finished)==4:break
  time.sleep(5)
if __name__=='__main__':
 try:main()
 except Exception:dump(TOP/'repair_controller_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise

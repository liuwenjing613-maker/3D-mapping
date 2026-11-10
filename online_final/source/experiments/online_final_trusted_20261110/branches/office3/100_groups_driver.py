from common_continue import *
from selection_runtime import run
import time,traceback,fcntl
out=CANDIDATE_ROOT/'group_runtime';out.mkdir(exist_ok=True);sc=CONT/'selection_code/100/B';slotdir=ROOT.parent.parent.parent/'cpu_evidence_slots';slotdir.mkdir(exist_ok=True);lock=None
try:
 p=CANDIDATE_ROOT/'semantic_runtime/driver_status.json'
 while True:
  st=read(p).get('status') if p.exists() else 'WAITING'
  if st=='PASS':break
  if st=='FAILED':raise RuntimeError('semantic route failed')
  dump(out/'driver_status.json',dict(status='WAITING_SEMANTIC',not_medoids=True));time.sleep(20)
 while lock is None:
  for i in range(3):
   f=(slotdir/(str(i)+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=f;break
   except BlockingIOError:f.close()
  if not lock:dump(out/'driver_status.json',dict(status='WAITING_CPU_SLOT',not_medoids=True));time.sleep(15)
 for stage in ['grouping_r2','grouping_r3','v1']:
  dump(out/'driver_status.json',dict(status='RUNNING',stage=stage,unix=time.time(),not_medoids=True));run(stage,sc/stage/'compute.py',out=out)
 dump(out/'driver_status.json',dict(status='PASS',stage='original_groups_only',not_medoids=True,not_tracking_or_repair=True))
except Exception:dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc(),not_medoids=True));raise
finally:
 if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

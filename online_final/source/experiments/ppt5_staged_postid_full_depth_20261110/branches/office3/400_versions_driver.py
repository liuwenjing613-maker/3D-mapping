from common_continue import *
from selection_runtime import run
import time,traceback,fcntl
out=CANDIDATE_ROOT/'versions_runtime';out.mkdir(parents=True,exist_ok=True);slotdir=ROOT.parent.parent.parent/'cpu_evidence_slots';slotdir.mkdir(exist_ok=True);lock=None
try:
 assert read(CANDIDATE_ROOT/'upstream_evidence/driver_status.json')['status']=='PASS'
 while lock is None:
  for i in range(3):
   f=(slotdir/(str(i)+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=f;break
   except BlockingIOError:f.close()
  if lock is None:dump(out/'driver_status.json',dict(status='WAITING_CPU_SLOT',not_fixedH=True));time.sleep(15)
 dump(out/'driver_status.json',dict(status='RUNNING',stage='original_input_version_selection',unix=time.time(),not_fixedH=True))
 row=run('versions',CONT/'selection_code/400/B/versions/compute.py',out=out)
 dump(out/'driver_status.json',dict(status='PASS',process=row,not_fixedH=True,not_tracking_or_repair=True))
except Exception:dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc(),not_fixedH=True));raise
finally:
 if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

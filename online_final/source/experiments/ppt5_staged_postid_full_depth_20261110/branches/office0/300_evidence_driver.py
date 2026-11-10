from common_continue import *
from selection_runtime import run
import time,traceback,fcntl
out=CANDIDATE_ROOT/'upstream_evidence';out.mkdir(parents=True,exist_ok=True);sc=CONT/'selection_code/300/B';slotdir=ROOT.parent.parent.parent/'cpu_evidence_slots';slotdir.mkdir(exist_ok=True);lock=None
try:
 summary=CANDIDATE_ROOT/'frontend/run_summary/driver_status.json'
 while True:
  status=read(summary).get('status') if summary.exists() else 'WAITING'
  if status=='PASS':break
  if status=='FAILED':raise RuntimeError('frontend summary failed')
  dump(out/'driver_status.json',dict(status='WAITING_FRONTEND',tracking_or_repair=False));time.sleep(30)
 while lock is None:
  for i in range(2):
   f=(slotdir/(str(i)+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=f;break
   except BlockingIOError:f.close()
  if lock is None:dump(out/'driver_status.json',dict(status='WAITING_CPU_SLOT',tracking_or_repair=False));time.sleep(15)
 for stage,rel in [('overmerge','overmerge/score.py'),('intersection','intersection/compute.py')]:
  dump(out/'driver_status.json',dict(status='RUNNING',stage=stage,unix=time.time(),tracking_or_repair=False));run(stage,sc/rel,out=out)
 dump(out/'driver_status.json',dict(status='PASS',stage='upstream_evidence_only',tracking_or_repair=False))
except Exception:dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc(),tracking_or_repair=False));raise
finally:
 if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

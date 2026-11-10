"""Run the independent protocol audit as checkpoint groups complete."""
from pathlib import Path
import json,subprocess,time,os,traceback
C=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def write(name,d):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 last=set()
 while True:
  ready=set()
  for n in [200,300]:
   for g in ['A','B']:
    p=ROOT/str(n)/g/'tracking/complete.json'
    if p.exists() and json.loads(p.read_text()).get('status')=='PASS':ready.add((n,g))
  if ready!=last:
   start=time.perf_counter()
   with (ROOT/'selection_protocol_audit.log').open('a') as log:
    proc=subprocess.run([PY,'-B','-u',str(C/'selection_protocol_audit.py')],cwd=C,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   assert proc.returncode==0,'Protocol audit failed; see selection_protocol_audit.log'
   last=ready;write('selection_audit_watch_status.json',dict(status='PASS' if len(ready)==4 else 'WAITING_REMAINING',audited_groups=sorted(ready),audit_process_seconds=time.perf_counter()-start,unix=time.time()))
  if len(ready)==4:break
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception:write('selection_audit_watch_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
